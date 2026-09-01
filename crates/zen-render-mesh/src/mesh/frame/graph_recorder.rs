use super::{MeshGraphResources, MeshRenderTargets};
use crate::mesh::{
    draw::MeshPassSet,
    scene::MeshGpuScene,
    stats::MeshStatsReadback,
    visibility::{HiZPyramidDesc, HiZStage, MeshVisibilityState},
};
use zen_frame_graph::{BufferRange, ClearBufferOp, Frame, FrameGraphError, RootReason};

/// Single-use ticket returned by [`crate::MeshRenderer::prepare_frame`].
///
/// The ticket is borrowed while recording and then consumed by exactly one terminal hook.
#[derive(Debug)]
pub struct PreparedMeshFrame {
    pub(crate) enable_occlusion_culling: bool,
    pub(crate) debug_camera: bool,
    pub(crate) readback_index: Option<usize>,
    pub(crate) extent: wgpu::Extent3d,
}

pub(crate) struct MeshGraphRecorder<'frame> {
    scene: &'frame MeshGpuScene,
    visibility: &'frame MeshVisibilityState,
    hiz_stage: &'frame HiZStage,
    passes: &'frame MeshPassSet,
    stats: &'frame MeshStatsReadback,
}

impl<'frame> MeshGraphRecorder<'frame> {
    pub(crate) const fn new(
        scene: &'frame MeshGpuScene,
        visibility: &'frame MeshVisibilityState,
        hiz_stage: &'frame HiZStage,
        passes: &'frame MeshPassSet,
        stats: &'frame MeshStatsReadback,
    ) -> Self {
        Self {
            scene,
            visibility,
            hiz_stage,
            passes,
            stats,
        }
    }

    pub(crate) fn record(
        self,
        frame: &mut Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        prepared: &PreparedMeshFrame,
    ) -> Result<(), FrameGraphError> {
        let Self {
            scene,
            visibility,
            hiz_stage,
            passes,
            stats,
        } = self;
        let hiz = HiZPyramidDesc::new(prepared.extent.width, prepared.extent.height);
        let readback_buffer = prepared
            .readback_index
            .map(|index| stats.staging_buffer(index));
        let resources = MeshGraphResources::register(
            frame,
            &visibility.list_a,
            &visibility.list_b,
            &visibility.history,
            prepared.enable_occlusion_culling.then_some(hiz),
            readback_buffer,
        )?;

        let max_instance_count = scene.instances().instance_count();
        Self::record_counter_clears(frame, &resources)?;
        passes
            .main_cull
            .record(frame, &resources, max_instance_count)?;
        passes.indirect_dispatch_prepare.record(
            frame,
            "dispatch-prepare-a",
            resources.list_a,
            &visibility.list_a,
        )?;
        passes.indirect_draw_prepare.record(
            frame,
            "draw-prepare-a",
            &resources,
            resources.list_a,
            &visibility.list_a,
        )?;
        passes.draw.record(
            frame,
            "draw-a",
            targets,
            resources.list_a,
            &visibility.list_a,
            scene,
            max_instance_count,
            0,
            true,
        )?;

        if prepared.enable_occlusion_culling {
            hiz_stage.record(
                frame,
                "Initial Hi-Z Pyramid",
                "hiz-initial",
                targets,
                &resources,
                hiz,
            )?;
            passes.indirect_dispatch_prepare.record(
                frame,
                "dispatch-prepare-b",
                resources.list_b,
                &visibility.list_b,
            )?;
            passes.occlusion_cull.record(
                frame,
                "occlusion-cull-b",
                &resources,
                resources.list_b,
                &visibility.list_b,
                scene,
                &visibility.history,
            )?;
            passes.indirect_draw_prepare.record(
                frame,
                "draw-prepare-b",
                &resources,
                resources.list_b,
                &visibility.list_b,
            )?;
            passes.draw.record(
                frame,
                "draw-b",
                targets,
                resources.list_b,
                &visibility.list_b,
                scene,
                max_instance_count,
                0,
                false,
            )?;

            hiz_stage.record(
                frame,
                "Final Hi-Z Pyramid",
                "hiz-final",
                targets,
                &resources,
                hiz,
            )?;
            passes.occlusion_cull.record(
                frame,
                "occlusion-cull-a-history",
                &resources,
                resources.list_a,
                &visibility.list_a,
                scene,
                &visibility.history,
            )?;
        }

        if prepared.debug_camera {
            frame.with_debug_group("Debug View", |frame| {
                passes.draw.record(
                    frame,
                    "debug-draw-a",
                    targets,
                    resources.list_a,
                    &visibility.list_a,
                    scene,
                    max_instance_count,
                    1,
                    true,
                )?;
                passes.draw.record(
                    frame,
                    "debug-draw-b",
                    targets,
                    resources.list_b,
                    &visibility.list_b,
                    scene,
                    max_instance_count,
                    1,
                    false,
                )
            })?;
        }

        let readback = if prepared.readback_index.is_some() {
            Some(frame.with_debug_group("Stats Readback", |frame| {
                stats.record_copy(frame, &resources)
            })?)
        } else {
            None
        };
        frame.mark_buffer_root(
            resources.history,
            BufferRange::whole(),
            RootReason::PersistentState,
        )?;
        if let Some(readback) = readback {
            frame.mark_readback(readback, BufferRange::whole())?;
        }
        Ok(())
    }

    fn record_counter_clears(
        frame: &mut Frame<'frame>,
        resources: &MeshGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        frame.clear_buffers(
            "clear-visibility-counters",
            [
                ClearBufferOp::new(resources.list_a.visible_count, BufferRange::whole()),
                ClearBufferOp::new(resources.list_a.draw_count, BufferRange::whole()),
                ClearBufferOp::new(resources.list_b.visible_count, BufferRange::whole()),
                ClearBufferOp::new(resources.list_b.draw_count, BufferRange::whole()),
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{
        draw::MeshPassSet,
        scene::{Instance, Material, Mesh, MeshGpuScene, Texture},
        stats::MeshStatsReadback,
        visibility::{HiZStage, MeshVisibilityState},
    };
    use zen_frame_graph::{
        AccessRole, CompileOptions, DependencyKind, FullCompilationReport, HazardKind, NodeKind,
        ReportLevel, ResourceOrigin, ResourceUsage, TextureDesc, UsagePolicy,
    };

    fn device_and_queue() -> (wgpu::Device, wgpu::Queue) {
        let bindless = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: bindless
                | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 16,
                max_binding_array_sampler_elements_per_shader_stage: 4,
                ..Default::default()
            },
            ..Default::default()
        })
    }

    fn record_topology(occlusion: bool, debug: bool, with_stats: bool) -> FullCompilationReport {
        let (device, queue) = device_and_queue();
        let meshes = [Mesh::create_triangle()];
        let materials = [Material {
            albedo_factor: glam::Vec4::ONE,
            emissive_ao: glam::Vec4::W,
            albedo: Default::default(),
            emissive: Default::default(),
            occlusion: Default::default(),
            _padding: [0; 2],
        }];
        let instances = [Instance {
            transform: glam::Mat4::IDENTITY,
            mesh_id: 0,
            material_id: 0,
            _pad: [0; 2],
        }];
        let textures = [Texture::white_1x1()];
        let scene = MeshGpuScene::new(
            &device,
            &queue,
            &meshes,
            &materials,
            &instances,
            &textures,
            &[],
            crate::TextureSamplingConfig::default(),
        )
        .unwrap();
        let visibility = MeshVisibilityState::new(&device, 1);
        let hiz_stage = HiZStage::new(&device);
        let passes = MeshPassSet::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            &scene,
            &visibility,
        );
        let mut stats = MeshStatsReadback::new(&device);
        if with_stats {
            stats.request();
        }

        let surface = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("topology.surface"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut graph = zen_frame_graph::FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let targets = frame
            .with_debug_group("Frame Targets", |frame| {
                let color = frame.import_surface_texture(
                    TextureDesc {
                        label: "surface-color".into(),
                        size: surface.size(),
                        mip_level_count: surface.mip_level_count(),
                        sample_count: surface.sample_count(),
                        dimension: surface.dimension(),
                        format: surface.format(),
                        view_formats: vec![],
                        usage: UsagePolicy::Fixed(surface.usage()),
                    },
                    Some(surface.usage()),
                )?;
                frame.bind_imported_texture(color, &surface)?;
                let depth = frame.create_texture(TextureDesc {
                    label: "depth-transient".into(),
                    size: surface.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    view_formats: vec![],
                    usage: UsagePolicy::Infer,
                })?;
                Ok(MeshRenderTargets::new(color, depth))
            })
            .unwrap();
        let prepared = PreparedMeshFrame {
            enable_occlusion_culling: occlusion,
            debug_camera: debug,
            readback_index: stats.planned_buffer_index(),
            extent: surface.size(),
        };
        MeshGraphRecorder::new(&scene, &visibility, &hiz_stage, &passes, &stats)
            .record(&mut frame, targets, &prepared)
            .unwrap();
        frame.mark_present(targets.color).unwrap();
        let compiled = frame
            .compile(CompileOptions {
                report_level: ReportLevel::Full,
            })
            .unwrap();
        compiled.report().unwrap().full.as_ref().unwrap().clone()
    }

    #[test]
    fn basic_topology_preserves_domain_boundaries_and_resource_origins() {
        let report = record_topology(false, false, false);
        assert_eq!(
            report
                .nodes
                .iter()
                .map(|node| node.label.as_str())
                .collect::<Vec<_>>(),
            [
                "clear-visibility-counters",
                "main-cull",
                "dispatch-prepare-a",
                "draw-prepare-a",
                "draw-a",
            ]
        );
        assert_eq!(
            report
                .nodes
                .iter()
                .map(|node| node.kind)
                .collect::<Vec<_>>(),
            [
                NodeKind::ClearBuffer,
                NodeKind::Compute,
                NodeKind::Compute,
                NodeKind::Compute,
                NodeKind::Render,
            ]
        );
        assert!(
            report
                .nodes
                .iter()
                .all(|node| node.kind != NodeKind::Command)
        );
        assert_eq!(report.execution_segments.len(), 1);
        assert_eq!(
            report
                .debug_groups
                .iter()
                .map(|group| group.label.as_str())
                .collect::<Vec<_>>(),
            ["Frame Targets"]
        );
        let group = |label: &str| {
            report
                .debug_groups
                .iter()
                .find(|group| group.label == label)
                .unwrap()
        };
        assert!(report.nodes.iter().all(|node| node.debug_group.is_none()));

        let resource = |label: &str| {
            report
                .resources
                .iter()
                .find(|resource| resource.label == label)
                .unwrap()
        };
        assert_eq!(resource("surface-color").origin, ResourceOrigin::Surface);
        assert_eq!(
            resource("depth-transient").origin,
            ResourceOrigin::Transient
        );
        assert_eq!(
            resource("depth-transient").effective_usage,
            ResourceUsage::Texture(wgpu::TextureUsages::RENDER_ATTACHMENT)
        );
        assert_eq!(
            resource("surface-color").debug_group,
            Some(group("Frame Targets").id)
        );
        assert!(
            report
                .resources
                .iter()
                .all(|resource| resource.label != "hiz-transient")
        );
        for renderer_owned in [
            "meshes.vertices",
            "meshes.indices",
            "meshes.table",
            "materials",
            "instances",
            "scene-texture-0",
            "main-cull.uniform",
            "occlusion.uniform",
            "draw.camera",
        ] {
            assert!(
                report
                    .resources
                    .iter()
                    .all(|resource| resource.label != renderer_owned),
                "renderer-owned resource {renderer_owned} must not be registered with the graph"
            );
        }
    }

    #[test]
    fn full_topology_preserves_hiz_attachment_and_root_dependencies() {
        let report = record_topology(true, true, true);
        assert_eq!(report.nodes.len(), 21);
        assert_eq!(
            report
                .nodes
                .iter()
                .map(|node| node.label.as_str())
                .collect::<Vec<_>>(),
            [
                "clear-visibility-counters",
                "main-cull",
                "dispatch-prepare-a",
                "draw-prepare-a",
                "draw-a",
                "hiz-initial-depth-to-mip0",
                "hiz-initial-mip0-to-mip1",
                "hiz-initial-mip1-to-mip2",
                "hiz-initial-mip2-to-mip3",
                "dispatch-prepare-b",
                "occlusion-cull-b",
                "draw-prepare-b",
                "draw-b",
                "hiz-final-depth-to-mip0",
                "hiz-final-mip0-to-mip1",
                "hiz-final-mip1-to-mip2",
                "hiz-final-mip2-to-mip3",
                "occlusion-cull-a-history",
                "debug-draw-a",
                "debug-draw-b",
                "stats-readback",
            ]
        );
        assert!(
            report
                .nodes
                .iter()
                .enumerate()
                .all(|(index, node)| node.recording_order == index as u32)
        );
        assert_eq!(report.nodes.last().unwrap().kind, NodeKind::Copy);
        assert!(
            report
                .nodes
                .iter()
                .all(|node| node.kind != NodeKind::Command)
        );
        assert_eq!(report.execution_segments.len(), 1);
        assert_eq!(
            report
                .debug_groups
                .iter()
                .map(|group| group.label.as_str())
                .collect::<Vec<_>>(),
            [
                "Frame Targets",
                "Initial Hi-Z Pyramid",
                "Final Hi-Z Pyramid",
                "Debug View",
                "Stats Readback",
            ]
        );
        let group = |label: &str| {
            report
                .debug_groups
                .iter()
                .find(|group| group.label == label)
                .unwrap()
        };
        assert!(
            report
                .debug_groups
                .iter()
                .all(|group| group.parent.is_none())
        );
        for (prefix, group_label) in [
            ("hiz-initial-", "Initial Hi-Z Pyramid"),
            ("hiz-final-", "Final Hi-Z Pyramid"),
        ] {
            let group_id = group(group_label).id;
            let nodes = report
                .nodes
                .iter()
                .filter(|node| node.label.starts_with(prefix))
                .collect::<Vec<_>>();
            assert_eq!(nodes.len(), 4);
            assert!(nodes.iter().all(|node| node.debug_group == Some(group_id)));
        }
        for label in [
            "clear-visibility-counters",
            "main-cull",
            "dispatch-prepare-a",
            "draw-prepare-a",
            "draw-a",
            "dispatch-prepare-b",
            "occlusion-cull-b",
            "draw-prepare-b",
            "draw-b",
            "occlusion-cull-a-history",
        ] {
            assert_eq!(
                report
                    .nodes
                    .iter()
                    .find(|node| node.label == label)
                    .unwrap()
                    .debug_group,
                None,
                "{label} should remain in the continuous top-level flow"
            );
        }
        for label in ["debug-draw-a", "debug-draw-b"] {
            assert_eq!(
                report
                    .nodes
                    .iter()
                    .find(|node| node.label == label)
                    .unwrap()
                    .debug_group,
                Some(group("Debug View").id)
            );
        }
        assert_eq!(
            report
                .nodes
                .iter()
                .find(|node| node.label == "stats-readback")
                .unwrap()
                .debug_group,
            Some(group("Stats Readback").id)
        );

        let resource = |label: &str| {
            report
                .resources
                .iter()
                .find(|resource| resource.label == label)
                .unwrap()
        };
        let depth = resource("depth-transient");
        let hiz = resource("hiz-transient");
        assert_eq!(hiz.debug_group, None);
        assert_eq!(
            depth.effective_usage,
            ResourceUsage::Texture(
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            )
        );
        assert_eq!(
            hiz.effective_usage,
            ResourceUsage::Texture(
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING
            )
        );
        assert_ne!(depth.allocation.unwrap(), hiz.allocation.unwrap());

        let node_id = |label: &str| {
            report
                .nodes
                .iter()
                .find(|node| node.label == label)
                .unwrap()
                .id
        };
        let clear = node_id("clear-visibility-counters");
        assert_eq!(
            report
                .accesses
                .iter()
                .filter(|access| {
                    access.pass == clear && access.role == AccessRole::BufferCopyDst
                })
                .count(),
            4
        );
        for (draw_label, hiz_label) in [
            ("draw-a", "hiz-initial-depth-to-mip0"),
            ("draw-b", "hiz-final-depth-to-mip0"),
        ] {
            assert!(report.dependencies.iter().any(|dependency| {
                dependency.from == node_id(draw_label)
                    && dependency.to == node_id(hiz_label)
                    && dependency.resource == depth.id
                    && dependency.kind == DependencyKind::Value
                    && dependency.hazard == HazardKind::Raw
            }));
        }
        for phase in ["initial", "final"] {
            let mut previous = node_id(&format!("hiz-{phase}-depth-to-mip0"));
            for mip in 1..4 {
                let current = node_id(&format!("hiz-{phase}-mip{}-to-mip{mip}", mip - 1));
                assert!(report.dependencies.iter().any(|dependency| {
                    dependency.from == previous
                        && dependency.to == current
                        && dependency.kind == DependencyKind::Value
                        && dependency.hazard == HazardKind::Raw
                }));
                previous = current;
            }
        }

        let debug_a = node_id("debug-draw-a");
        let debug_b = node_id("debug-draw-b");
        let is_attachment = |role| {
            matches!(
                role,
                AccessRole::ColorAttachment | AccessRole::DepthAttachment
            )
        };
        let debug_a_accesses = report
            .accesses
            .iter()
            .filter(|access| access.pass == debug_a && is_attachment(access.role))
            .collect::<Vec<_>>();
        assert_eq!(debug_a_accesses.len(), 2);
        assert!(
            debug_a_accesses
                .iter()
                .all(|access| !access.consumes_previous)
        );
        assert_eq!(
            report
                .dependencies
                .iter()
                .filter(|dependency| dependency.from == debug_a
                    && dependency.to == debug_b
                    && dependency.hazard == HazardKind::Preserve)
                .count(),
            2
        );

        let roots = report
            .roots
            .iter()
            .map(|root| root.reason)
            .collect::<Vec<_>>();
        assert!(roots.contains(&RootReason::Present));
        assert!(roots.contains(&RootReason::PersistentState));
        assert!(roots.contains(&RootReason::Readback));
    }

    #[test]
    fn optional_branches_are_owned_by_the_recorder_for_every_combination() {
        for occlusion in [false, true] {
            for debug in [false, true] {
                for stats in [false, true] {
                    let report = record_topology(occlusion, debug, stats);
                    let has_node =
                        |label: &str| report.nodes.iter().any(|node| node.label == label);
                    assert_eq!(has_node("occlusion-cull-b"), occlusion);
                    assert_eq!(has_node("hiz-final-depth-to-mip0"), occlusion);
                    assert_eq!(has_node("debug-draw-a"), debug);
                    assert_eq!(has_node("debug-draw-b"), debug);
                    assert_eq!(has_node("stats-readback"), stats);
                    assert!(
                        report
                            .nodes
                            .iter()
                            .all(|node| node.kind != NodeKind::Command)
                    );
                    assert_eq!(report.execution_segments.len(), 1);
                    let has_group =
                        |label: &str| report.debug_groups.iter().any(|group| group.label == label);
                    assert!(has_group("Frame Targets"));
                    assert!(!has_group("Mesh"));
                    assert!(!has_group("Main View"));
                    assert!(!has_group("Occlusion Refinement"));
                    assert!(!has_group("Visibility History"));
                    assert_eq!(has_group("Initial Hi-Z Pyramid"), occlusion);
                    assert_eq!(has_group("Final Hi-Z Pyramid"), occlusion);
                    assert_eq!(has_group("Debug View"), debug);
                    assert_eq!(has_group("Stats Readback"), stats);
                    assert!(
                        report
                            .debug_groups
                            .iter()
                            .all(|group| group.parent.is_none())
                    );
                    assert!(
                        report
                            .nodes
                            .iter()
                            .filter(|node| {
                                !node.label.starts_with("hiz-")
                                    && !node.label.starts_with("debug-draw-")
                                    && node.label != "stats-readback"
                            })
                            .all(|node| node.debug_group.is_none())
                    );

                    let roots = report
                        .roots
                        .iter()
                        .map(|root| root.reason)
                        .collect::<Vec<_>>();
                    assert!(roots.contains(&RootReason::Present));
                    assert!(roots.contains(&RootReason::PersistentState));
                    assert_eq!(roots.contains(&RootReason::Readback), stats);
                }
            }
        }
    }
}

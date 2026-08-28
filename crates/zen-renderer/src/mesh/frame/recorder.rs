use super::{FrameTargets, MeshFrameResources};
use crate::mesh::{
    draw::MeshDrawStage,
    scene::MeshSceneResources,
    stats::MeshStatsReadback,
    visibility::{HiZPyramidDesc, VisibilityStage},
};
use zen_frame_graph::{BufferRange, Frame, FrameGraphError, RootReason};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedMeshFrame {
    pub enable_occlusion_culling: bool,
    pub debug_camera: bool,
    pub readback_index: Option<usize>,
    pub extent: wgpu::Extent3d,
}

pub(crate) struct MeshFrameRecorder;

impl MeshFrameRecorder {
    pub(crate) fn record<'frame>(
        frame: &mut Frame<'frame>,
        targets: FrameTargets<'frame>,
        prepared: PreparedMeshFrame,
        scene: &'frame MeshSceneResources,
        visibility: &'frame VisibilityStage,
        draw: &'frame MeshDrawStage,
        stats: &'frame MeshStatsReadback,
    ) -> Result<(), FrameGraphError> {
        let hiz = HiZPyramidDesc::new(prepared.extent.width, prepared.extent.height);
        let readback_buffer = prepared
            .readback_index
            .map(|index| stats.staging_buffer(index));
        let resources = MeshFrameResources::register(
            frame,
            scene,
            &visibility.list_a,
            &visibility.list_b,
            &visibility.history,
            hiz,
            visibility.main_cull.uniform_buffer(),
            visibility.occlusion_cull.uniform_buffer(),
            draw.draw.camera_buffer(),
            readback_buffer,
        )?;

        let max_instance_count = scene.instances().instance_count();
        frame.with_debug_group("Main View", |frame| {
            visibility.record_counter_clears(frame, &resources)?;
            visibility
                .main_cull
                .record(frame, &resources, max_instance_count)?;
            visibility.dispatch_prepare.record(
                frame,
                "dispatch-prepare-a",
                resources.list_a,
                &visibility.list_a,
            )?;
            draw.prepare.record(
                frame,
                "draw-prepare-a",
                &resources,
                resources.list_a,
                &visibility.list_a,
            )?;
            draw.draw.record(
                frame,
                "draw-a",
                targets,
                &resources,
                resources.list_a,
                &visibility.list_a,
                scene,
                max_instance_count,
                0,
                true,
            )
        })?;

        if prepared.enable_occlusion_culling {
            frame.with_debug_group("Occlusion Refinement", |frame| {
                visibility.hiz_generator.record_depth_to_mip0(
                    frame,
                    "hiz-initial-depth-to-mip0",
                    targets,
                    &resources,
                    hiz,
                )?;
                for mip in 1..hiz.mip_level_count() {
                    visibility.hiz_generator.record_mip_to_mip(
                        frame,
                        format!("hiz-initial-mip{}-to-mip{mip}", mip - 1),
                        &resources,
                        hiz,
                        mip,
                    )?;
                }
                visibility.dispatch_prepare.record(
                    frame,
                    "dispatch-prepare-b",
                    resources.list_b,
                    &visibility.list_b,
                )?;
                visibility.occlusion_cull.record(
                    frame,
                    "occlusion-cull-b",
                    &resources,
                    resources.list_b,
                    &visibility.list_b,
                    scene,
                    &visibility.history,
                )?;
                draw.prepare.record(
                    frame,
                    "draw-prepare-b",
                    &resources,
                    resources.list_b,
                    &visibility.list_b,
                )?;
                draw.draw.record(
                    frame,
                    "draw-b",
                    targets,
                    &resources,
                    resources.list_b,
                    &visibility.list_b,
                    scene,
                    max_instance_count,
                    0,
                    false,
                )
            })?;

            frame.with_debug_group("Visibility History", |frame| {
                visibility.hiz_generator.record_depth_to_mip0(
                    frame,
                    "hiz-final-depth-to-mip0",
                    targets,
                    &resources,
                    hiz,
                )?;
                for mip in 1..hiz.mip_level_count() {
                    visibility.hiz_generator.record_mip_to_mip(
                        frame,
                        format!("hiz-final-mip{}-to-mip{mip}", mip - 1),
                        &resources,
                        hiz,
                        mip,
                    )?;
                }
                visibility.occlusion_cull.record(
                    frame,
                    "occlusion-cull-a-history",
                    &resources,
                    resources.list_a,
                    &visibility.list_a,
                    scene,
                    &visibility.history,
                )
            })?;
        }

        if prepared.debug_camera {
            frame.with_debug_group("Debug View", |frame| {
                draw.draw.record(
                    frame,
                    "debug-draw-a",
                    targets,
                    &resources,
                    resources.list_a,
                    &visibility.list_a,
                    scene,
                    max_instance_count,
                    1,
                    true,
                )?;
                draw.draw.record(
                    frame,
                    "debug-draw-b",
                    targets,
                    &resources,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{
        draw::MeshDrawStage,
        scene::{Instance, Material, Mesh, MeshSceneResources, Texture},
        stats::MeshStatsReadback,
        visibility::VisibilityStage,
    };
    use zen_frame_graph::{
        AccessRole, CompileOptions, DependencyKind, FullCompilationReport, HazardKind, NodeKind,
        ReportLevel, ResourceOrigin, ResourceUsage,
    };

    fn device_and_queue() -> (wgpu::Device, wgpu::Queue) {
        let bindless = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: bindless | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 16,
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
            tex_ids: [0; 4],
        }];
        let instances = [Instance {
            transform: glam::Mat4::IDENTITY,
            mesh_id: 0,
            material_id: 0,
            _pad: [0; 2],
        }];
        let textures = [Texture::white_1x1()];
        let scene =
            MeshSceneResources::new(&device, &queue, &meshes, &materials, &instances, &textures);
        let visibility = VisibilityStage::new(&device, &scene, 1);
        let draw = MeshDrawStage::new(
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
                FrameTargets::register(frame, &surface)
            })
            .unwrap();
        frame
            .with_debug_group("Mesh", |frame| {
                MeshFrameRecorder::record(
                    frame,
                    targets,
                    PreparedMeshFrame {
                        enable_occlusion_culling: occlusion,
                        debug_camera: debug,
                        readback_index: stats.planned_buffer_index(),
                        extent: surface.size(),
                    },
                    &scene,
                    &visibility,
                    &draw,
                    &stats,
                )
            })
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
            ["Frame Targets", "Mesh", "Main View"]
        );
        let group = |label: &str| {
            report
                .debug_groups
                .iter()
                .find(|group| group.label == label)
                .unwrap()
        };
        assert_eq!(group("Main View").parent, Some(group("Mesh").id));
        assert!(
            report
                .nodes
                .iter()
                .all(|node| { node.debug_group == Some(group("Main View").id) })
        );

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
        assert_eq!(resource("hiz-transient").origin, ResourceOrigin::Transient);
        assert_eq!(
            resource("depth-transient").effective_usage,
            ResourceUsage::Texture(wgpu::TextureUsages::RENDER_ATTACHMENT)
        );
        assert_eq!(resource("hiz-transient").lifetime, None);
        assert_eq!(resource("hiz-transient").allocation, None);
        assert_eq!(
            resource("surface-color").debug_group,
            Some(group("Frame Targets").id)
        );
        assert_eq!(
            resource("hiz-transient").debug_group,
            Some(group("Mesh").id)
        );
    }

    #[test]
    fn full_topology_preserves_hiz_attachment_and_root_dependencies() {
        let report = record_topology(true, true, true);
        assert_eq!(report.nodes.len(), 21);
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
                "Mesh",
                "Main View",
                "Occlusion Refinement",
                "Visibility History",
                "Debug View",
                "Stats Readback",
            ]
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
                    assert!(has_group("Mesh"));
                    assert!(has_group("Main View"));
                    assert_eq!(has_group("Occlusion Refinement"), occlusion);
                    assert_eq!(has_group("Visibility History"), occlusion);
                    assert_eq!(has_group("Debug View"), debug);
                    assert_eq!(has_group("Stats Readback"), stats);

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

use crate::{
    mesh::frame::FrameTargets,
    mesh::{MeshFrameInput, MeshRenderer},
};
#[cfg(feature = "snapshot")]
use zen_frame_graph::snapshot::{
    CreateFrameGraphSnapshotOptions, FrameGraphSnapshotV1, SnapshotExportError,
    create_frame_graph_snapshot,
};
#[cfg(feature = "snapshot")]
use zen_frame_graph::{CompilationReport, ResourcePoolStats};
use zen_frame_graph::{
    CompileOptions, ExecutionOptions, FrameGraph, FrameGraphError, GpuTimingReadback,
    GpuTimingReport,
};

pub struct FrameInput<'a> {
    pub frame_index: u64,
    pub surface_texture: &'a wgpu::Texture,
    pub mesh: MeshFrameInput,
}

pub struct Renderer {
    mesh: MeshRenderer,
    frame_graph: FrameGraph,
    surface_format: wgpu::TextureFormat,
    last_surface_extent: Option<(u32, u32)>,
    gpu_debug_groups_enabled: bool,
    gpu_timing: GpuTimingRequestState,
    #[cfg(feature = "snapshot")]
    snapshot: FrameGraphSnapshotRequestState,
}

#[derive(Debug, Default)]
struct GpuTimingRequestState {
    requested: bool,
    readback: Option<GpuTimingReadback>,
}

impl GpuTimingRequestState {
    fn request(&mut self) {
        self.requested = true;
    }

    fn should_execute(&self) -> bool {
        self.requested && self.readback.is_none()
    }

    fn commit(&mut self, readback: GpuTimingReadback) {
        debug_assert!(self.should_execute());
        self.readback = Some(readback);
        self.requested = false;
    }

    fn take(&mut self) -> Option<GpuTimingReport> {
        let report = self.readback.as_mut()?.try_take()?;
        self.readback = None;
        Some(report)
    }
}

#[cfg(feature = "snapshot")]
#[derive(Debug, Default)]
struct FrameGraphSnapshotRequestState {
    requested: bool,
    pending: Option<PendingFrameGraphSnapshot>,
}

#[cfg(feature = "snapshot")]
#[derive(Debug)]
struct PendingFrameGraphSnapshot {
    frame_index: u64,
    report: CompilationReport,
    pool_stats: ResourcePoolStats,
    timing: GpuTimingReadback,
}

#[cfg(feature = "snapshot")]
impl FrameGraphSnapshotRequestState {
    fn request(&mut self) {
        if self.pending.is_none() {
            self.requested = true;
        }
    }

    fn should_execute(&self, ordinary_timing_pending: bool) -> bool {
        self.requested && self.pending.is_none() && !ordinary_timing_pending
    }

    fn commit(
        &mut self,
        frame_index: u64,
        report: CompilationReport,
        pool_stats: ResourcePoolStats,
        timing: GpuTimingReadback,
    ) {
        debug_assert!(self.requested && self.pending.is_none());
        self.pending = Some(PendingFrameGraphSnapshot {
            frame_index,
            report,
            pool_stats,
            timing,
        });
        self.requested = false;
    }

    fn take(&mut self) -> Option<Result<FrameGraphSnapshotV1, SnapshotExportError>> {
        let timing = self.pending.as_mut()?.timing.try_take()?;
        let pending = self.pending.take().expect("pending snapshot disappeared");
        let mut options = CreateFrameGraphSnapshotOptions::new(pending.frame_index);
        options.gpu_timing = Some(&timing);
        options.pool_stats = Some(pending.pool_stats);
        Some(create_frame_graph_snapshot(&pending.report, options))
    }
}

impl Renderer {
    pub fn new(device: &wgpu::Device, mesh: MeshRenderer) -> Self {
        let surface_format = mesh.color_format();
        Self {
            mesh,
            frame_graph: FrameGraph::with_device(device),
            surface_format,
            last_surface_extent: None,
            gpu_debug_groups_enabled: false,
            gpu_timing: GpuTimingRequestState::default(),
            #[cfg(feature = "snapshot")]
            snapshot: FrameGraphSnapshotRequestState::default(),
        }
    }

    pub fn mesh(&self) -> &MeshRenderer {
        &self.mesh
    }

    pub fn mesh_mut(&mut self) -> &mut MeshRenderer {
        &mut self.mesh
    }

    pub fn set_gpu_debug_groups_enabled(&mut self, enabled: bool) {
        self.gpu_debug_groups_enabled = enabled;
    }

    /// Requests one GPU timing sample from the next eligible successful frame.
    pub fn request_gpu_timing(&mut self) {
        self.gpu_timing.request();
    }

    /// Non-blockingly polls and takes the current GPU timing result.
    pub fn take_gpu_timing(&mut self) -> Option<GpuTimingReport> {
        self.gpu_timing.take()
    }

    /// Requests a Snapshot 1.0 object from the next eligible successful frame.
    /// Repeated requests coalesce while a request or capture is pending.
    #[cfg(feature = "snapshot")]
    pub fn request_frame_graph_snapshot(&mut self) {
        self.snapshot.request();
    }

    /// Non-blockingly polls and takes the current Snapshot result.
    #[cfg(feature = "snapshot")]
    pub fn take_frame_graph_snapshot(
        &mut self,
    ) -> Option<Result<FrameGraphSnapshotV1, SnapshotExportError>> {
        self.snapshot.take()
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: FrameInput<'_>,
    ) -> Result<(), FrameGraphError> {
        if input.surface_texture.format() != self.surface_format {
            return Err(FrameGraphError::InvalidResourceDescriptor {
                message: format!(
                    "surface texture format {:?} does not match renderer format {:?}",
                    input.surface_texture.format(),
                    self.surface_format
                ),
            });
        }

        let surface_extent = input.surface_texture.size();
        let extent = (surface_extent.width, surface_extent.height);
        if should_clear_resource_pool(self.last_surface_extent, extent) {
            self.frame_graph.clear_resource_pool();
        }

        let prepared = self.mesh.prepare_frame(queue, input.mesh, surface_extent);
        let mut frame = self.frame_graph.begin_frame();
        let targets = frame.with_debug_group("Frame Targets", |frame| {
            FrameTargets::register(frame, input.surface_texture)
        })?;
        self.mesh
            .record_frame_graph(&mut frame, targets, prepared)?;
        frame.mark_present(targets.color)?;

        let execution_options = ExecutionOptions::default()
            .with_gpu_debug_groups(self.gpu_debug_groups_enabled)
            .with_frame_index(input.frame_index);

        #[cfg(feature = "snapshot")]
        {
            let should_snapshot = self
                .snapshot
                .should_execute(self.gpu_timing.readback.is_some());
            let compile_options = if should_snapshot {
                CompileOptions::full_report()
            } else {
                CompileOptions::default()
            };
            let mut compiled = frame.compile(compile_options)?;
            if should_snapshot {
                let report = compiled
                    .take_report()
                    .expect("full report requested for snapshot capture");
                let readback = compiled.execute_with_gpu_timing(queue, execution_options)?;
                let pool_stats = self.frame_graph.resource_pool_stats();
                self.snapshot
                    .commit(input.frame_index, report, pool_stats, readback);
            } else if self.gpu_timing.should_execute() {
                let readback = compiled.execute_with_gpu_timing(queue, execution_options)?;
                self.gpu_timing.commit(readback);
            } else {
                compiled.execute_with_options(queue, execution_options)?;
            }
        }

        #[cfg(not(feature = "snapshot"))]
        {
            let compiled = frame.compile(CompileOptions::default())?;
            if self.gpu_timing.should_execute() {
                let readback = compiled.execute_with_gpu_timing(queue, execution_options)?;
                self.gpu_timing.commit(readback);
            } else {
                compiled.execute_with_options(queue, execution_options)?;
            }
        }

        self.mesh.after_submit(device, prepared);
        self.last_surface_extent = Some(extent);
        Ok(())
    }
}

fn should_clear_resource_pool(previous: Option<(u32, u32)>, current: (u32, u32)) -> bool {
    previous.is_some_and(|previous| previous != current)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "snapshot")]
    use super::FrameGraphSnapshotRequestState;
    use super::{GpuTimingRequestState, should_clear_resource_pool};
    #[cfg(feature = "snapshot")]
    use crate::{
        FrameInput,
        camera::Camera,
        mesh::{Instance, Material, Mesh, MeshFrameInput, MeshRenderer, Texture},
    };
    #[cfg(feature = "snapshot")]
    use zen_frame_graph::snapshot::{SnapshotGpuTimings, SnapshotPoolReport};
    use zen_frame_graph::{CompileOptions, ExecutionOptions, FrameGraph, GpuTimingReport};

    #[test]
    fn pool_is_cleared_only_after_a_successful_extent_changes() {
        assert!(!should_clear_resource_pool(None, (800, 600)));
        assert!(!should_clear_resource_pool(Some((800, 600)), (800, 600)));
        assert!(should_clear_resource_pool(Some((800, 600)), (1280, 720)));
    }

    #[test]
    fn timing_requests_coalesce_and_queue_behind_an_unconsumed_result() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut state = GpuTimingRequestState::default();
        assert!(!state.should_execute());
        state.request();
        state.request();
        assert!(state.should_execute());

        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        frame
            .command_pass("command")
            .finish_command(|_| Ok(()))
            .unwrap();
        let readback = frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute_with_gpu_timing(&queue, ExecutionOptions::default())
            .unwrap();
        state.commit(readback);
        assert!(!state.should_execute());

        state.request();
        state.request();
        assert!(!state.should_execute());
        assert!(matches!(
            state.take(),
            Some(GpuTimingReport::Available { .. })
        ));
        assert!(state.should_execute());
    }

    #[cfg(feature = "snapshot")]
    #[test]
    fn snapshot_requests_coalesce_take_priority_and_move_the_full_report() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut snapshot = FrameGraphSnapshotRequestState::default();
        let mut ordinary = GpuTimingRequestState::default();
        ordinary.request();
        snapshot.request();
        snapshot.request();

        assert!(!snapshot.should_execute(true));
        assert!(snapshot.should_execute(false));
        assert!(ordinary.should_execute());

        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        frame
            .command_pass("snapshot-command")
            .finish_command(|_| Ok(()))
            .unwrap();
        let mut compiled = frame.compile(CompileOptions::full_report()).unwrap();
        let report = compiled.take_report().unwrap();
        assert!(
            compiled.report().is_none(),
            "the full report must be moved, not cloned"
        );
        let timing = compiled
            .execute_with_gpu_timing(&queue, ExecutionOptions::default().with_frame_index(9))
            .unwrap();
        snapshot.commit(9, report, graph.resource_pool_stats(), timing);

        snapshot.request();
        assert!(
            !snapshot.requested,
            "a pending capture absorbs duplicate requests"
        );
        assert!(
            ordinary.requested,
            "ordinary timing remains queued behind Snapshot"
        );
        let result = snapshot
            .take()
            .expect("noop readback is immediate")
            .unwrap();
        assert_eq!(result.capture.frame_index, 9);
        assert!(matches!(
            result.memory.pool_report,
            SnapshotPoolReport::Available { .. }
        ));
        assert!(matches!(
            result.timings.gpu,
            SnapshotGpuTimings::Available { .. } | SnapshotGpuTimings::Unavailable { .. }
        ));
    }

    #[cfg(feature = "snapshot")]
    #[test]
    fn renderer_snapshot_covers_the_real_mesh_frame_graph() {
        let bindless = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: bindless | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 16,
                ..Default::default()
            },
            ..Default::default()
        });
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let mesh = MeshRenderer::new(
            &device,
            &queue,
            format,
            &[Mesh::create_triangle()],
            &[Material {
                albedo_factor: glam::Vec4::ONE,
                emissive_ao: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
                tex_ids: [0; 4],
            }],
            &[Instance {
                transform: glam::Mat4::IDENTITY,
                mesh_id: 0,
                material_id: 0,
                _pad: [0; 2],
            }],
            &[Texture::white_1x1()],
        );
        let surface = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("snapshot-test-surface"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut renderer = super::Renderer::new(&device, mesh);
        renderer.request_frame_graph_snapshot();
        renderer
            .render(
                &device,
                &queue,
                FrameInput {
                    frame_index: 17,
                    surface_texture: &surface,
                    mesh: MeshFrameInput {
                        camera: Camera::default(),
                        debug_camera: None,
                        enable_occlusion_culling: true,
                    },
                },
            )
            .unwrap();

        let snapshot = renderer
            .take_frame_graph_snapshot()
            .expect("noop timing readback is immediate")
            .unwrap();
        assert_eq!(snapshot.capture.frame_index, 17);
        assert!(!snapshot.graph.nodes.is_empty());
        assert_eq!(
            snapshot
                .graph
                .groups
                .iter()
                .map(|group| group.label.as_str())
                .collect::<Vec<_>>(),
            [
                "Frame Targets",
                "Initial Hi-Z Pyramid",
                "Final Hi-Z Pyramid"
            ]
        );
        assert!(
            snapshot
                .graph
                .groups
                .iter()
                .all(|group| group.parent_id.is_none())
        );
        let group_id = |label: &str| {
            snapshot
                .graph
                .groups
                .iter()
                .find(|group| group.label == label)
                .unwrap()
                .id
                .as_str()
        };
        for (prefix, group_label) in [
            ("hiz-initial-", "Initial Hi-Z Pyramid"),
            ("hiz-final-", "Final Hi-Z Pyramid"),
        ] {
            let nodes = snapshot
                .graph
                .nodes
                .iter()
                .filter(|node| {
                    node.label
                        .as_deref()
                        .is_some_and(|label| label.starts_with(prefix))
                })
                .collect::<Vec<_>>();
            assert!(!nodes.is_empty());
            assert!(
                nodes
                    .iter()
                    .all(|node| { node.group_id.as_deref() == Some(group_id(group_label)) })
            );
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
                snapshot
                    .graph
                    .nodes
                    .iter()
                    .find(|node| node.label.as_deref() == Some(label))
                    .unwrap()
                    .group_id,
                None
            );
        }
        assert!(
            snapshot
                .graph
                .nodes
                .iter()
                .enumerate()
                .all(|(index, node)| node.recording_order == index as u64)
        );
        assert!(snapshot.graph.resources.iter().any(|resource| {
            resource.origin == zen_frame_graph::snapshot::SnapshotResourceOrigin::Surface
        }));
        assert_eq!(
            snapshot
                .graph
                .resources
                .iter()
                .find(|resource| resource.label.as_deref() == Some("surface-color"))
                .unwrap()
                .group_id
                .as_deref(),
            Some(group_id("Frame Targets"))
        );
        assert_eq!(
            snapshot
                .graph
                .resources
                .iter()
                .find(|resource| resource.label.as_deref() == Some("meshes.vertices"))
                .unwrap()
                .group_id,
            None
        );
    }
}

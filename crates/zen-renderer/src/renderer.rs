use crate::{
    mesh::frame::FrameTargets,
    mesh::{MeshFrameInput, MeshRenderer},
};
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
        frame.with_debug_group("Mesh", |frame| {
            self.mesh.record_frame_graph(frame, targets, prepared)
        })?;
        frame.mark_present(targets.color)?;

        let compiled = frame.compile(CompileOptions::default())?;
        let execution_options = ExecutionOptions::default()
            .with_gpu_debug_groups(self.gpu_debug_groups_enabled)
            .with_frame_index(input.frame_index);
        let should_time = self.gpu_timing.should_execute();
        if should_time {
            let readback = compiled.execute_with_gpu_timing(queue, execution_options)?;
            self.gpu_timing.commit(readback);
        } else {
            compiled.execute_with_options(queue, execution_options)?;
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
    use super::{GpuTimingRequestState, should_clear_resource_pool};
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
}

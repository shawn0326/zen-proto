use crate::{FrameComposeContext, FrameComposer, PresentTarget, RenderFrameInput};
#[cfg(feature = "snapshot")]
use zen_frame_graph::snapshot::{
    CreateFrameGraphSnapshotOptions, FrameGraphSnapshotV1, SnapshotExportError,
    create_frame_graph_snapshot,
};
#[cfg(feature = "snapshot")]
use zen_frame_graph::{CompilationReport, ResourcePoolStats};
use zen_frame_graph::{
    CompileOptions, ExecutionCpuTimings, ExecutionOptions, Frame, FrameGraph, FrameGraphError,
    GpuTimingReadback, GpuTimingReport, TextureDesc, UsagePolicy,
};

/// Domain-independent owner of FrameGraph recording, compilation, execution,
/// transient pooling, and diagnostics.
pub struct RenderHost<C> {
    composer: C,
    frame_graph: FrameGraph,
    last_surface_extent: Option<(u32, u32)>,
    gpu_debug_groups_enabled: bool,
    gpu_timing: GpuTimingRequestState,
    cpu_timings: Option<ExecutionCpuTimings>,
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

    fn try_request(&mut self) -> bool {
        if self.requested || self.readback.is_some() {
            return false;
        }
        self.requested = true;
        true
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

impl<C: FrameComposer> RenderHost<C> {
    pub fn new(device: &wgpu::Device, composer: C) -> Self {
        Self {
            composer,
            frame_graph: FrameGraph::with_device(device),
            last_surface_extent: None,
            gpu_debug_groups_enabled: false,
            gpu_timing: GpuTimingRequestState::default(),
            cpu_timings: None,
            #[cfg(feature = "snapshot")]
            snapshot: FrameGraphSnapshotRequestState::default(),
        }
    }

    pub const fn composer(&self) -> &C {
        &self.composer
    }

    pub fn composer_mut(&mut self) -> &mut C {
        &mut self.composer
    }

    pub fn set_gpu_debug_groups_enabled(&mut self, enabled: bool) {
        self.gpu_debug_groups_enabled = enabled;
    }

    /// Requests one GPU timing sample from the next eligible successful frame.
    pub fn request_gpu_timing(&mut self) {
        self.gpu_timing.request();
    }

    /// Attempts to reserve the next frame for timing without coalescing behind an in-flight
    /// readback. This is the handshake primitive for pairing timestamps with another one-shot
    /// frame resource such as an asynchronous counter copy.
    #[must_use]
    pub fn try_request_gpu_timing(&mut self) -> bool {
        self.gpu_timing.try_request()
    }

    /// Non-blockingly polls and takes the current GPU timing result.
    pub fn take_gpu_timing(&mut self) -> Option<GpuTimingReport> {
        self.gpu_timing.take()
    }

    /// Takes the CPU command-encoding and queue-submit timing for the last successful frame.
    pub fn take_cpu_timings(&mut self) -> Option<ExecutionCpuTimings> {
        self.cpu_timings.take()
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

    pub fn render_frame<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: RenderFrameInput<'a, C::FrameInput<'a>>,
    ) -> Result<(), FrameGraphError> {
        self.cpu_timings = None;
        let surface_extent =
            validate_present_texture(input.surface_texture, self.composer.present_format())?;
        let extent = (surface_extent.width, surface_extent.height);
        if should_clear_resource_pool(self.last_surface_extent, extent) {
            self.frame_graph.clear_resource_pool();
        }

        let prepared = self
            .composer
            .prepare_frame(queue, input.composer_input, surface_extent);
        let result = self.record_compile_execute(
            queue,
            input.frame_index,
            input.surface_texture,
            surface_extent,
            &prepared,
        );

        match result {
            Ok(cpu_timings) => {
                self.composer.after_submit(device, prepared);
                self.last_surface_extent = Some(extent);
                self.cpu_timings = Some(cpu_timings);
                Ok(())
            }
            Err(error) => {
                self.composer.after_discard(prepared);
                Err(error)
            }
        }
    }

    fn record_compile_execute(
        &mut self,
        queue: &wgpu::Queue,
        frame_index: u64,
        surface_texture: &wgpu::Texture,
        extent: wgpu::Extent3d,
        prepared: &C::PreparedFrame,
    ) -> Result<ExecutionCpuTimings, FrameGraphError> {
        let mut frame = self.frame_graph.begin_frame();
        let present_target = frame.with_debug_group("Frame Targets", |frame| {
            register_present_target(frame, surface_texture)
        })?;
        let mut context = FrameComposeContext::new(frame, present_target, frame_index, extent);
        self.composer.record_frame_graph(&mut context, prepared)?;
        let (mut frame, present_target) = context.into_parts();
        frame.mark_present(present_target.texture)?;

        let execution_options = ExecutionOptions::default()
            .with_gpu_debug_groups(self.gpu_debug_groups_enabled)
            .with_frame_index(frame_index);

        #[cfg(feature = "snapshot")]
        let cpu_timings = {
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
                let (readback, cpu_timings) =
                    compiled.execute_with_gpu_timing_profiled(queue, execution_options)?;
                let pool_stats = self.frame_graph.resource_pool_stats();
                self.snapshot
                    .commit(frame_index, report, pool_stats, readback);
                cpu_timings
            } else if self.gpu_timing.should_execute() {
                let (readback, cpu_timings) =
                    compiled.execute_with_gpu_timing_profiled(queue, execution_options)?;
                self.gpu_timing.commit(readback);
                cpu_timings
            } else {
                compiled.execute_profiled(queue, execution_options)?
            }
        };

        #[cfg(not(feature = "snapshot"))]
        let cpu_timings = {
            let compiled = frame.compile(CompileOptions::default())?;
            if self.gpu_timing.should_execute() {
                let (readback, cpu_timings) =
                    compiled.execute_with_gpu_timing_profiled(queue, execution_options)?;
                self.gpu_timing.commit(readback);
                cpu_timings
            } else {
                compiled.execute_profiled(queue, execution_options)?
            }
        };

        Ok(cpu_timings)
    }
}

fn validate_present_texture(
    texture: &wgpu::Texture,
    expected_format: wgpu::TextureFormat,
) -> Result<wgpu::Extent3d, FrameGraphError> {
    if texture.format() != expected_format {
        return Err(FrameGraphError::InvalidResourceDescriptor {
            message: format!(
                "present texture format {:?} does not match composer format {:?}",
                texture.format(),
                expected_format
            ),
        });
    }

    let extent = texture.size();
    if extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0 {
        return Err(FrameGraphError::InvalidResourceDescriptor {
            message: "present texture has a zero extent".into(),
        });
    }
    Ok(extent)
}

fn register_present_target<'frame>(
    frame: &mut Frame<'frame>,
    native: &wgpu::Texture,
) -> Result<PresentTarget<'frame>, FrameGraphError> {
    let texture = frame.import_surface_texture(
        TextureDesc {
            label: "surface-color".into(),
            size: native.size(),
            mip_level_count: native.mip_level_count(),
            sample_count: native.sample_count(),
            dimension: native.dimension(),
            format: native.format(),
            view_formats: vec![],
            usage: UsagePolicy::Fixed(native.usage()),
        },
        Some(native.usage()),
    )?;
    frame.bind_imported_texture(texture, native)?;
    Ok(PresentTarget::new(texture))
}

fn should_clear_resource_pool(previous: Option<(u32, u32)>, current: (u32, u32)) -> bool {
    previous.is_some_and(|previous| previous != current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameComposer;
    use zen_frame_graph::{ColorAttachmentOps, CompileOptions, ResourceOrigin, RootReason};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FailAt {
        Never,
        Record,
        Compile,
        Execute,
    }

    #[derive(Debug)]
    struct FakeComposer {
        format: wgpu::TextureFormat,
        fail_at: FailAt,
        prepare_count: u32,
        submit_count: u32,
        discard_count: u32,
        terminal_values: Vec<u32>,
    }

    impl FakeComposer {
        fn new(format: wgpu::TextureFormat, fail_at: FailAt) -> Self {
            Self {
                format,
                fail_at,
                prepare_count: 0,
                submit_count: 0,
                discard_count: 0,
                terminal_values: Vec::new(),
            }
        }
    }

    impl FrameComposer for FakeComposer {
        type FrameInput<'a> = u32;
        type PreparedFrame = u32;

        fn present_format(&self) -> wgpu::TextureFormat {
            self.format
        }

        fn prepare_frame(
            &mut self,
            _queue: &wgpu::Queue,
            input: Self::FrameInput<'_>,
            _extent: wgpu::Extent3d,
        ) -> Self::PreparedFrame {
            self.prepare_count += 1;
            input
        }

        fn record_frame_graph<'frame>(
            &'frame self,
            context: &mut FrameComposeContext<'frame>,
            _prepared: &Self::PreparedFrame,
        ) -> Result<(), FrameGraphError> {
            if self.fail_at == FailAt::Record {
                return Err(FrameGraphError::Internal {
                    message: "injected record failure".into(),
                });
            }
            if self.fail_at == FailAt::Compile {
                return Ok(());
            }

            let target = context.present_target().texture;
            let fail_at = self.fail_at;
            let mut pass = context.frame_mut().render_pass("fake-present");
            let _ =
                pass.color_attachment(target, ColorAttachmentOps::clear_store(wgpu::Color::BLACK))?;
            pass.finish_render(move |_| {
                if fail_at == FailAt::Execute {
                    Err(FrameGraphError::Internal {
                        message: "injected execute failure".into(),
                    })
                } else {
                    Ok(())
                }
            })?;
            Ok(())
        }

        fn after_submit(&mut self, _device: &wgpu::Device, prepared: Self::PreparedFrame) {
            self.submit_count += 1;
            self.terminal_values.push(prepared);
        }

        fn after_discard(&mut self, prepared: Self::PreparedFrame) {
            self.discard_count += 1;
            self.terminal_values.push(prepared);
        }
    }

    fn device_and_queue() -> (wgpu::Device, wgpu::Queue) {
        wgpu::Device::noop(&wgpu::DeviceDescriptor::default())
    }

    fn texture(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("host-test-surface"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    #[test]
    fn successful_frame_commits_exactly_once() {
        let (device, queue) = device_and_queue();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let surface = texture(&device, format, 8, 8);
        let mut host = RenderHost::new(&device, FakeComposer::new(format, FailAt::Never));

        host.render_frame(&device, &queue, RenderFrameInput::new(7, &surface, 41))
            .unwrap();

        assert!(host.take_cpu_timings().is_some());
        assert!(host.take_cpu_timings().is_none());

        let composer = host.composer();
        assert_eq!(composer.prepare_count, 1);
        assert_eq!(composer.submit_count, 1);
        assert_eq!(composer.discard_count, 0);
        assert_eq!(composer.terminal_values, [41]);
    }

    #[test]
    fn every_post_prepare_failure_discards_exactly_once() {
        for fail_at in [FailAt::Record, FailAt::Compile, FailAt::Execute] {
            let (device, queue) = device_and_queue();
            let format = wgpu::TextureFormat::Rgba8Unorm;
            let surface = texture(&device, format, 8, 8);
            let mut host = RenderHost::new(&device, FakeComposer::new(format, fail_at));

            assert!(
                host.render_frame(&device, &queue, RenderFrameInput::new(11, &surface, 73),)
                    .is_err(),
                "{fail_at:?} must fail"
            );

            let composer = host.composer();
            assert_eq!(composer.prepare_count, 1, "{fail_at:?}");
            assert_eq!(composer.submit_count, 0, "{fail_at:?}");
            assert_eq!(composer.discard_count, 1, "{fail_at:?}");
            assert_eq!(composer.terminal_values, [73], "{fail_at:?}");
        }
    }

    #[test]
    fn invalid_present_format_is_rejected_before_prepare() {
        let (device, queue) = device_and_queue();
        let surface = texture(&device, wgpu::TextureFormat::Rgba8Unorm, 8, 8);
        let mut host = RenderHost::new(
            &device,
            FakeComposer::new(wgpu::TextureFormat::Bgra8Unorm, FailAt::Never),
        );

        assert!(
            host.render_frame(&device, &queue, RenderFrameInput::new(0, &surface, 1),)
                .is_err()
        );
        assert_eq!(host.composer().prepare_count, 0);
        assert_eq!(host.composer().discard_count, 0);
    }

    #[test]
    fn present_target_has_surface_origin_and_present_root() {
        let (device, _) = device_and_queue();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let surface = texture(&device, format, 8, 8);
        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let target = register_present_target(&mut frame, &surface).unwrap();
        let mut clear = frame.render_pass("clear-present");
        let _ = clear
            .color_attachment(
                target.texture,
                ColorAttachmentOps::clear_store(wgpu::Color::BLACK),
            )
            .unwrap();
        clear.finish().unwrap();
        frame.mark_present(target.texture).unwrap();

        let compiled = frame.compile(CompileOptions::full_report()).unwrap();
        let report = compiled.report().unwrap().full.as_ref().unwrap();
        let resource = report
            .resources
            .iter()
            .find(|resource| resource.label == "surface-color")
            .unwrap();
        assert_eq!(resource.origin, ResourceOrigin::Surface);
        assert!(
            report
                .roots
                .iter()
                .any(|root| { root.resource == resource.id && root.reason == RootReason::Present })
        );
    }

    #[test]
    fn pool_is_cleared_only_after_a_successful_extent_changes() {
        assert!(!should_clear_resource_pool(None, (800, 600)));
        assert!(!should_clear_resource_pool(Some((800, 600)), (800, 600)));
        assert!(should_clear_resource_pool(Some((800, 600)), (1280, 720)));
    }

    #[test]
    fn timing_requests_coalesce_and_queue_behind_an_unconsumed_result() {
        let (device, queue) = device_and_queue();
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
        use zen_frame_graph::snapshot::{SnapshotGpuTimings, SnapshotPoolReport};

        let (device, queue) = device_and_queue();
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
        assert!(compiled.report().is_none());
        let timing = compiled
            .execute_with_gpu_timing(&queue, ExecutionOptions::default().with_frame_index(9))
            .unwrap();
        snapshot.commit(9, report, graph.resource_pool_stats(), timing);

        snapshot.request();
        assert!(!snapshot.requested);
        assert!(ordinary.requested);
        let result = snapshot.take().unwrap().unwrap();
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
    fn host_snapshot_covers_composer_graph_and_present_surface() {
        use zen_frame_graph::snapshot::SnapshotResourceOrigin;

        let (device, queue) = device_and_queue();
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let surface = texture(&device, format, 16, 16);
        let mut host = RenderHost::new(&device, FakeComposer::new(format, FailAt::Never));
        host.request_frame_graph_snapshot();
        host.render_frame(&device, &queue, RenderFrameInput::new(17, &surface, 5))
            .unwrap();

        let snapshot = host.take_frame_graph_snapshot().unwrap().unwrap();
        assert_eq!(snapshot.capture.frame_index, 17);
        assert!(
            snapshot
                .graph
                .nodes
                .iter()
                .any(|node| { node.label.as_deref() == Some("fake-present") })
        );
        let present = snapshot
            .graph
            .resources
            .iter()
            .find(|resource| resource.label.as_deref() == Some("surface-color"))
            .unwrap();
        assert_eq!(present.origin, SnapshotResourceOrigin::Surface);
        assert_eq!(
            present.group_id.as_deref(),
            snapshot
                .graph
                .groups
                .iter()
                .find(|group| group.label == "Frame Targets")
                .map(|group| group.id.as_str())
        );
    }
}

use zen_frame_graph::{Frame, FrameGraphError, Texture};

/// Domain-specific composition contract used by [`crate::RenderHost`].
///
/// Composition is static and typed: a host owns one concrete composer, and the
/// composer is free to connect any number of render domains through FrameGraph
/// handles without a dynamic registry or resource blackboard.
pub trait FrameComposer {
    /// Composer-owned input supplied for one frame.
    type FrameInput<'a>;

    /// Transaction ticket produced by preparation and consumed by exactly one
    /// terminal hook.
    type PreparedFrame;

    /// Format required by the final present texture.
    fn present_format(&self) -> wgpu::TextureFormat;

    /// Performs CPU work and queue uploads needed before FrameGraph recording.
    fn prepare_frame(
        &mut self,
        queue: &wgpu::Queue,
        input: Self::FrameInput<'_>,
        extent: wgpu::Extent3d,
    ) -> Self::PreparedFrame;

    /// Records the complete frame recipe into the supplied context.
    fn record_frame_graph<'frame>(
        &'frame self,
        context: &mut FrameComposeContext<'frame>,
        prepared: &Self::PreparedFrame,
    ) -> Result<(), FrameGraphError>;

    /// Commits preparation state after successful GPU submission.
    fn after_submit(&mut self, device: &wgpu::Device, prepared: Self::PreparedFrame);

    /// Rolls back or releases preparation state after a post-prepare failure.
    fn after_discard(&mut self, prepared: Self::PreparedFrame);
}

/// Host-owned frame values plus the concrete composer's input.
pub struct RenderFrameInput<'a, T> {
    pub frame_index: u64,
    pub surface_texture: &'a wgpu::Texture,
    pub composer_input: T,
}

impl<'a, T> RenderFrameInput<'a, T> {
    pub const fn new(
        frame_index: u64,
        surface_texture: &'a wgpu::Texture,
        composer_input: T,
    ) -> Self {
        Self {
            frame_index,
            surface_texture,
            composer_input,
        }
    }
}

/// Logical FrameGraph handle for the caller-acquired texture that will be
/// presented after the frame succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentTarget<'frame> {
    pub texture: Texture<'frame>,
}

impl<'frame> PresentTarget<'frame> {
    pub const fn new(texture: Texture<'frame>) -> Self {
        Self { texture }
    }
}

/// FrameGraph recording context shared with one concrete composer.
///
/// The context owns the recording so its FrameGraph handles remain tied to one
/// frame. Domain recorders can borrow the frame mutably and copy the present
/// target into their own typed input structures.
pub struct FrameComposeContext<'frame> {
    frame: Frame<'frame>,
    present_target: PresentTarget<'frame>,
    frame_index: u64,
    extent: wgpu::Extent3d,
}

impl<'frame> FrameComposeContext<'frame> {
    pub(crate) const fn new(
        frame: Frame<'frame>,
        present_target: PresentTarget<'frame>,
        frame_index: u64,
        extent: wgpu::Extent3d,
    ) -> Self {
        Self {
            frame,
            present_target,
            frame_index,
            extent,
        }
    }

    pub const fn frame(&self) -> &Frame<'frame> {
        &self.frame
    }

    pub fn frame_mut(&mut self) -> &mut Frame<'frame> {
        &mut self.frame
    }

    pub const fn present_target(&self) -> PresentTarget<'frame> {
        self.present_target
    }

    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub const fn extent(&self) -> wgpu::Extent3d {
        self.extent
    }

    pub(crate) fn into_parts(self) -> (Frame<'frame>, PresentTarget<'frame>) {
        (self.frame, self.present_target)
    }
}

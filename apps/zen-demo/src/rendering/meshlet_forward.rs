use zen_frame_graph::{FrameGraphError, TextureDesc};
use zen_render::{FrameComposeContext, FrameComposer, RenderHost};
use zen_render_mesh::{
    MeshRenderTargets, MeshletRenderInput, MeshletRenderer, PreparedMeshletFrame,
};

/// Forward-frame recipe for the independent Vulkan meshlet renderer.
pub struct MeshletForwardFrameComposer {
    meshlet: MeshletRenderer,
    present_format: wgpu::TextureFormat,
}

pub type MeshletForwardRenderHost = RenderHost<MeshletForwardFrameComposer>;

impl MeshletForwardFrameComposer {
    pub const fn new(meshlet: MeshletRenderer, present_format: wgpu::TextureFormat) -> Self {
        Self {
            meshlet,
            present_format,
        }
    }

    pub const fn meshlet(&self) -> &MeshletRenderer {
        &self.meshlet
    }

    pub fn meshlet_mut(&mut self) -> &mut MeshletRenderer {
        &mut self.meshlet
    }
}

impl FrameComposer for MeshletForwardFrameComposer {
    type FrameInput<'a> = MeshletRenderInput;
    type PreparedFrame = PreparedMeshletFrame;

    fn present_format(&self) -> wgpu::TextureFormat {
        self.present_format
    }

    fn prepare_frame(
        &mut self,
        queue: &wgpu::Queue,
        input: Self::FrameInput<'_>,
        extent: wgpu::Extent3d,
    ) -> Self::PreparedFrame {
        self.meshlet.prepare_frame(queue, input, extent)
    }

    fn record_frame_graph<'frame>(
        &'frame self,
        context: &mut FrameComposeContext<'frame>,
        prepared: &Self::PreparedFrame,
    ) -> Result<(), FrameGraphError> {
        let extent = context.extent();
        let color = context.present_target().texture;
        let depth = context.frame_mut().create_texture(TextureDesc::new_2d(
            "meshlet-depth-transient",
            extent.width,
            extent.height,
            wgpu::TextureFormat::Depth32Float,
        ))?;
        self.meshlet.record_frame_graph(
            context.frame_mut(),
            MeshRenderTargets::new(color, depth),
            prepared,
        )
    }

    fn after_submit(&mut self, device: &wgpu::Device, prepared: Self::PreparedFrame) {
        self.meshlet.after_submit(device, prepared);
    }

    fn after_discard(&mut self, prepared: Self::PreparedFrame) {
        self.meshlet.after_discard(prepared);
    }
}

#[cfg(test)]
mod tests {
    use zen_frame_graph::UsagePolicy;

    #[test]
    fn depth_target_contract_matches_forward_shading() {
        let descriptor = zen_frame_graph::TextureDesc::new_2d(
            "meshlet-depth-transient",
            1920,
            1080,
            wgpu::TextureFormat::Depth32Float,
        );
        assert_eq!(descriptor.size.width, 1920);
        assert_eq!(descriptor.size.height, 1080);
        assert_eq!(descriptor.format, wgpu::TextureFormat::Depth32Float);
        assert!(matches!(descriptor.usage, UsagePolicy::Infer));
    }
}

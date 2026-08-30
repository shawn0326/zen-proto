use zen_frame_graph::{FrameGraphError, TextureDesc};
use zen_render::{FrameComposeContext, FrameComposer, RenderHost};
use zen_render_mesh::{MeshRenderInput, MeshRenderTargets, MeshRenderer, PreparedMeshFrame};

/// Concrete forward-frame recipe shared by the interactive demos.
///
/// The host owns generic FrameGraph execution. This composer owns the Mesh
/// domain, creates frame-local scene targets, and connects the two layers.
pub struct ForwardFrameComposer {
    mesh: MeshRenderer,
    present_format: wgpu::TextureFormat,
}

pub type ForwardRenderHost = RenderHost<ForwardFrameComposer>;

fn depth_target_desc(extent: wgpu::Extent3d) -> TextureDesc {
    TextureDesc::new_2d(
        "depth-transient",
        extent.width,
        extent.height,
        wgpu::TextureFormat::Depth32Float,
    )
}

impl ForwardFrameComposer {
    pub const fn new(mesh: MeshRenderer, present_format: wgpu::TextureFormat) -> Self {
        Self {
            mesh,
            present_format,
        }
    }

    pub const fn mesh(&self) -> &MeshRenderer {
        &self.mesh
    }

    pub fn mesh_mut(&mut self) -> &mut MeshRenderer {
        &mut self.mesh
    }
}

impl FrameComposer for ForwardFrameComposer {
    type FrameInput<'a> = MeshRenderInput;
    type PreparedFrame = PreparedMeshFrame;

    fn present_format(&self) -> wgpu::TextureFormat {
        self.present_format
    }

    fn prepare_frame(
        &mut self,
        queue: &wgpu::Queue,
        input: Self::FrameInput<'_>,
        extent: wgpu::Extent3d,
    ) -> Self::PreparedFrame {
        self.mesh.prepare_frame(queue, input, extent)
    }

    fn record_frame_graph<'frame>(
        &'frame self,
        context: &mut FrameComposeContext<'frame>,
        prepared: &Self::PreparedFrame,
    ) -> Result<(), FrameGraphError> {
        let extent = context.extent();
        let color = context.present_target().texture;
        let depth = context
            .frame_mut()
            .create_texture(depth_target_desc(extent))?;
        let targets = MeshRenderTargets::new(color, depth);
        self.mesh
            .record_frame_graph(context.frame_mut(), targets, prepared)
    }

    fn after_submit(&mut self, device: &wgpu::Device, prepared: Self::PreparedFrame) {
        self.mesh.after_submit(device, prepared);
    }

    fn after_discard(&mut self, prepared: Self::PreparedFrame) {
        self.mesh.after_discard(prepared);
    }
}

#[cfg(test)]
mod tests {
    use super::{ForwardFrameComposer, depth_target_desc};
    use zen_frame_graph::{
        UsagePolicy,
        snapshot::{SnapshotResourceOrigin, SnapshotRootReason},
    };
    use zen_render::{RenderFrameInput, RenderHost};
    use zen_render_mesh::{Camera, MeshRenderInput, MeshRenderer};

    #[test]
    fn depth_target_preserves_the_forward_recipe_contract() {
        let desc = depth_target_desc(wgpu::Extent3d {
            width: 1280,
            height: 720,
            depth_or_array_layers: 1,
        });

        assert_eq!(desc.label, "depth-transient");
        assert_eq!(desc.size.width, 1280);
        assert_eq!(desc.size.height, 720);
        assert_eq!(desc.size.depth_or_array_layers, 1);
        assert_eq!(desc.mip_level_count, 1);
        assert_eq!(desc.sample_count, 1);
        assert_eq!(desc.dimension, wgpu::TextureDimension::D2);
        assert_eq!(desc.format, wgpu::TextureFormat::Depth32Float);
        assert!(matches!(desc.usage, UsagePolicy::Infer));
    }

    #[test]
    fn empty_mesh_frame_uses_composer_depth_and_one_host_surface() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: MeshRenderer::required_features(),
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 16,
                ..Default::default()
            },
            ..Default::default()
        });
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let mesh = MeshRenderer::new(&device, &queue, format, &[], &[], &[], &[]);
        let composer = ForwardFrameComposer::new(mesh, format);
        let mut host = RenderHost::new(&device, composer);
        let surface = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("forward-composer-test-surface"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        host.request_frame_graph_snapshot();
        host.render_frame(
            &device,
            &queue,
            RenderFrameInput::new(
                23,
                &surface,
                MeshRenderInput {
                    camera: Camera::default(),
                    debug_camera: None,
                    enable_occlusion_culling: true,
                },
            ),
        )
        .unwrap();

        let snapshot = host.take_frame_graph_snapshot().unwrap().unwrap();
        assert_eq!(snapshot.capture.frame_index, 23);
        assert!(!snapshot.graph.nodes.is_empty());

        let surface_resources = snapshot
            .graph
            .resources
            .iter()
            .filter(|resource| resource.origin == SnapshotResourceOrigin::Surface)
            .collect::<Vec<_>>();
        assert_eq!(surface_resources.len(), 1);
        let present = surface_resources[0];
        assert_eq!(present.label.as_deref(), Some("surface-color"));

        let depth = snapshot
            .graph
            .resources
            .iter()
            .find(|resource| resource.label.as_deref() == Some("depth-transient"))
            .unwrap();
        assert_eq!(depth.origin, SnapshotResourceOrigin::Transient);

        let present_roots = snapshot
            .graph
            .roots
            .iter()
            .filter(|root| root.reason == SnapshotRootReason::Present)
            .collect::<Vec<_>>();
        assert_eq!(present_roots.len(), 1);
        assert_eq!(
            present_roots[0].resource_id.as_deref(),
            Some(present.id.as_str())
        );
    }
}

use super::{DrawPreparePass, MeshDrawPass};
use crate::mesh::{scene::MeshSceneResources, visibility::VisibilityStage};

pub(crate) struct MeshDrawStage {
    pub prepare: DrawPreparePass,
    pub draw: MeshDrawPass,
}

impl MeshDrawStage {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        resources: &MeshSceneResources,
        visibility: &VisibilityStage,
    ) -> Self {
        let prepare = DrawPreparePass::new(device);
        prepare.prepare(device, resources, &visibility.history, &visibility.list_a);
        prepare.prepare(device, resources, &visibility.history, &visibility.list_b);

        Self {
            prepare,
            draw: MeshDrawPass::new(device, color_format, resources),
        }
    }
}

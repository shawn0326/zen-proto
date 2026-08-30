use super::{IndirectDrawPreparePass, MeshDrawPass};
use crate::mesh::{
    scene::MeshGpuScene,
    visibility::{
        IndirectDispatchPreparePass, MainCullPass, MeshVisibilityState, OcclusionCullPass,
    },
};

/// The stateless and pipeline-owning passes used by the Mesh graph contribution.
pub(crate) struct MeshPassSet {
    pub main_cull: MainCullPass,
    pub indirect_dispatch_prepare: IndirectDispatchPreparePass,
    pub occlusion_cull: OcclusionCullPass,
    pub indirect_draw_prepare: IndirectDrawPreparePass,
    pub draw: MeshDrawPass,
}

impl MeshPassSet {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        scene: &MeshGpuScene,
        visibility: &MeshVisibilityState,
    ) -> Self {
        let main_cull = MainCullPass::new(device);
        main_cull.prepare(
            device,
            scene,
            &visibility.history,
            &visibility.list_a,
            &visibility.list_b,
        );

        let indirect_dispatch_prepare = IndirectDispatchPreparePass::new(device);
        indirect_dispatch_prepare.prepare(device, &visibility.list_a);
        indirect_dispatch_prepare.prepare(device, &visibility.list_b);

        let indirect_draw_prepare = IndirectDrawPreparePass::new(device);
        indirect_draw_prepare.prepare(device, scene, &visibility.history, &visibility.list_a);
        indirect_draw_prepare.prepare(device, scene, &visibility.history, &visibility.list_b);

        Self {
            main_cull,
            indirect_dispatch_prepare,
            occlusion_cull: OcclusionCullPass::new(device),
            indirect_draw_prepare,
            draw: MeshDrawPass::new(device, color_format, scene),
        }
    }
}

use super::{
    DispatchPreparePass, HiZGenerator, MainCullPass, OcclusionCullPass, VisibilityHistory,
    VisibilityList,
};
use crate::mesh::frame::MeshFrameResources;
use crate::mesh::scene::MeshSceneResources;
use zen_frame_graph::{BufferRange, Frame, FrameGraphError};

pub(crate) struct VisibilityStage {
    pub list_a: VisibilityList,
    pub list_b: VisibilityList,
    pub history: VisibilityHistory,
    pub main_cull: MainCullPass,
    pub dispatch_prepare: DispatchPreparePass,
    pub occlusion_cull: OcclusionCullPass,
    pub hiz_generator: HiZGenerator,
}

impl VisibilityStage {
    pub fn new(
        device: &wgpu::Device,
        resources: &MeshSceneResources,
        max_instance_count: u32,
    ) -> Self {
        let list_a = VisibilityList::new(device, "list_a", max_instance_count);
        let list_b = VisibilityList::new(device, "list_b", max_instance_count);
        let history = VisibilityHistory::new(device, max_instance_count);

        let main_cull = MainCullPass::new(device);
        main_cull.prepare(device, resources, &history, &list_a, &list_b);

        let dispatch_prepare = DispatchPreparePass::new(device);
        dispatch_prepare.prepare(device, &list_a);
        dispatch_prepare.prepare(device, &list_b);

        Self {
            list_a,
            list_b,
            history,
            main_cull,
            dispatch_prepare,
            occlusion_cull: OcclusionCullPass::new(device),
            hiz_generator: HiZGenerator::new(device),
        }
    }

    pub(crate) fn record_counter_clears<'frame>(
        &self,
        frame: &mut Frame<'frame>,
        resources: &MeshFrameResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        for (label, buffer) in [
            ("clear-list-a-visible-count", resources.list_a.visible_count),
            ("clear-list-a-draw-count", resources.list_a.draw_count),
            ("clear-list-b-visible-count", resources.list_b.visible_count),
            ("clear-list-b-draw-count", resources.list_b.draw_count),
        ] {
            frame.clear_buffer(label, buffer, BufferRange::whole())?;
        }
        Ok(())
    }
}

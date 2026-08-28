use crate::camera::Camera;
use crate::mesh::{
    draw::MeshDrawStage,
    frame::{FrameTargets, MeshFrameRecorder, PreparedMeshFrame},
    scene::{Instance, Material, Mesh, MeshSceneResources, Texture},
    stats::{MeshRenderStats, MeshStatsReadback},
    visibility::VisibilityStage,
};
use zen_frame_graph::{Frame, FrameGraphError};

#[derive(Clone, Copy, Debug)]
pub struct MeshFrameInput {
    pub camera: Camera,
    pub debug_camera: Option<Camera>,
    pub enable_occlusion_culling: bool,
}

pub struct MeshRenderer {
    color_format: wgpu::TextureFormat,
    scene: MeshSceneResources,
    visibility: VisibilityStage,
    draw: MeshDrawStage,
    stats: MeshStatsReadback,
}

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        meshes: &[Mesh],
        materials: &[Material],
        instances: &[Instance],
        textures: &[Texture],
    ) -> Self {
        let scene = MeshSceneResources::new(device, queue, meshes, materials, instances, textures);
        let max_instance_count = scene.instances().instance_count();
        let visibility = VisibilityStage::new(device, &scene, max_instance_count);
        let draw = MeshDrawStage::new(device, color_format, &scene, &visibility);

        Self {
            color_format,
            scene,
            visibility,
            draw,
            stats: MeshStatsReadback::new(device),
        }
    }

    pub fn request_stats(&mut self) {
        self.stats.request();
    }

    pub fn take_stats(&mut self, device: &wgpu::Device) -> Option<MeshRenderStats> {
        self.stats
            .take_ready(device, self.scene.instances().instance_count())
    }

    pub(crate) fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub(crate) fn prepare_frame(
        &mut self,
        queue: &wgpu::Queue,
        input: MeshFrameInput,
        extent: wgpu::Extent3d,
    ) -> PreparedMeshFrame {
        self.visibility.main_cull.update(
            queue,
            &self.scene,
            &input.camera,
            input.enable_occlusion_culling,
        );
        self.draw.draw.update(queue, &input.camera, 0);
        if input.enable_occlusion_culling {
            self.visibility.occlusion_cull.update(
                queue,
                &input.camera,
                extent.width,
                extent.height,
            );
        }
        if let Some(debug_camera) = input.debug_camera {
            self.draw.draw.update(queue, &debug_camera, 1);
        }

        PreparedMeshFrame {
            enable_occlusion_culling: input.enable_occlusion_culling,
            debug_camera: input.debug_camera.is_some(),
            readback_index: self.stats.planned_buffer_index(),
            extent,
        }
    }

    pub(crate) fn record_frame_graph<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        targets: FrameTargets<'frame>,
        prepared: PreparedMeshFrame,
    ) -> Result<(), FrameGraphError> {
        MeshFrameRecorder::record(
            frame,
            targets,
            prepared,
            &self.scene,
            &self.visibility,
            &self.draw,
            &self.stats,
        )
    }

    pub(crate) fn after_submit(&mut self, device: &wgpu::Device, prepared: PreparedMeshFrame) {
        if let Some(buffer_index) = prepared.readback_index {
            self.stats
                .commit_submitted(buffer_index, prepared.enable_occlusion_culling);
        }
        self.stats.after_submit(device);
    }
}

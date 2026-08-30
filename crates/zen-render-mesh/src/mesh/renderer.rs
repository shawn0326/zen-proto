use crate::camera::Camera;
use crate::mesh::{
    draw::MeshPassSet,
    frame::{MeshGraphRecorder, MeshRenderTargets, PreparedMeshFrame},
    scene::{Instance, Material, Mesh, MeshGpuScene, Texture},
    stats::{MeshRenderStats, MeshStatsReadback},
    visibility::{HiZStage, MeshVisibilityState},
};
use zen_frame_graph::{Frame, FrameGraphError};

#[derive(Clone, Copy, Debug)]
pub struct MeshRenderInput {
    pub camera: Camera,
    pub debug_camera: Option<Camera>,
    pub enable_occlusion_culling: bool,
}

pub struct MeshRenderer {
    scene: MeshGpuScene,
    visibility: MeshVisibilityState,
    hiz_stage: HiZStage,
    passes: MeshPassSet,
    stats: MeshStatsReadback,
}

impl MeshRenderer {
    /// WebGPU features required by the Mesh renderer.
    pub fn required_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
    }

    /// WebGPU limits required by the Mesh renderer, clamped to the selected adapter.
    pub fn required_limits(adapter_limits: &wgpu::Limits) -> wgpu::Limits {
        wgpu::Limits {
            max_binding_array_elements_per_shader_stage: 1024
                .min(adapter_limits.max_binding_array_elements_per_shader_stage),
            ..Default::default()
        }
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        meshes: &[Mesh],
        materials: &[Material],
        instances: &[Instance],
        textures: &[Texture],
    ) -> Self {
        let scene = MeshGpuScene::new(device, queue, meshes, materials, instances, textures);
        let max_instance_count = scene.instances().instance_count();
        let visibility = MeshVisibilityState::new(device, max_instance_count);
        let passes = MeshPassSet::new(device, color_format, &scene, &visibility);

        Self {
            scene,
            visibility,
            hiz_stage: HiZStage::new(device),
            passes,
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

    pub fn prepare_frame(
        &mut self,
        queue: &wgpu::Queue,
        input: MeshRenderInput,
        extent: wgpu::Extent3d,
    ) -> PreparedMeshFrame {
        self.passes.main_cull.update(
            queue,
            &self.scene,
            &input.camera,
            input.enable_occlusion_culling,
        );
        self.passes.draw.update(queue, &input.camera, 0);
        if input.enable_occlusion_culling {
            self.passes
                .occlusion_cull
                .update(queue, &input.camera, extent.width, extent.height);
        }
        if let Some(debug_camera) = input.debug_camera {
            self.passes.draw.update(queue, &debug_camera, 1);
        }

        PreparedMeshFrame {
            enable_occlusion_culling: input.enable_occlusion_culling,
            debug_camera: input.debug_camera.is_some(),
            readback_index: self.stats.planned_buffer_index(),
            extent,
        }
    }

    pub fn record_frame_graph<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        prepared: &PreparedMeshFrame,
    ) -> Result<(), FrameGraphError> {
        MeshGraphRecorder::new(
            &self.scene,
            &self.visibility,
            &self.hiz_stage,
            &self.passes,
            &self.stats,
        )
        .record(frame, targets, prepared)
    }

    pub fn after_submit(&mut self, device: &wgpu::Device, prepared: PreparedMeshFrame) {
        if let Some(buffer_index) = prepared.readback_index {
            self.stats
                .commit_submitted(buffer_index, prepared.enable_occlusion_culling);
        }
        self.stats.after_submit(device);
    }

    /// Ends a prepared frame that was not submitted.
    ///
    /// No state is committed, so a pending stats request remains available to the next frame.
    pub fn after_discard(&mut self, _prepared: PreparedMeshFrame) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discarded_frame_keeps_the_stats_request_for_the_next_prepare() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: MeshRenderer::required_features(),
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 16,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut renderer = MeshRenderer::new(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
            &[],
            &[],
            &[],
            &[],
        );
        let input = MeshRenderInput {
            camera: Camera::default(),
            debug_camera: None,
            enable_occlusion_culling: false,
        };
        let extent = wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        };

        renderer.request_stats();
        let discarded = renderer.prepare_frame(&queue, input, extent);
        assert_eq!(discarded.readback_index, Some(0));
        renderer.after_discard(discarded);

        let retry = renderer.prepare_frame(&queue, input, extent);
        assert_eq!(retry.readback_index, Some(0));
    }
}

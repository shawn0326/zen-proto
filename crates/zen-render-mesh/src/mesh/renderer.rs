use crate::camera::Camera;
use crate::mesh::{
    draw::MeshPassSet,
    frame::{MeshGraphRecorder, MeshRenderTargets, PreparedMeshFrame},
    scene::{
        Instance, Material, MaterialTextureBinding, Mesh, MeshGpuScene, Texture,
        TextureResourceError, TextureSampler, TextureSamplingConfig,
    },
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

#[derive(Debug, thiserror::Error)]
pub enum MeshRendererError {
    #[error(transparent)]
    TextureResource(#[from] TextureResourceError),
    #[error(
        "material {material_index} {slot} texture index {texture_id} is out of range for {texture_count} textures"
    )]
    InvalidMaterialTexture {
        material_index: usize,
        slot: &'static str,
        texture_id: u32,
        texture_count: usize,
    },
    #[error(
        "material {material_index} {slot} sampler index {sampler_id} is out of range for {sampler_count} samplers"
    )]
    InvalidMaterialSampler {
        material_index: usize,
        slot: &'static str,
        sampler_id: u32,
        sampler_count: usize,
    },
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
            max_binding_array_sampler_elements_per_shader_stage: 32
                .min(adapter_limits.max_binding_array_sampler_elements_per_shader_stage),
            ..Default::default()
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "constructor keeps the legacy renderer resource inputs explicit"
    )]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        meshes: &[Mesh],
        materials: &[Material],
        instances: &[Instance],
        textures: &[Texture],
        samplers: &[TextureSampler],
        sampling: TextureSamplingConfig,
    ) -> Result<Self, MeshRendererError> {
        let texture_count = textures.len().max(1);
        let sampler_count = samplers.len().max(1);
        for (material_index, material) in materials.iter().enumerate() {
            for (slot, binding) in [
                ("albedo", material.albedo),
                ("emissive", material.emissive),
                ("occlusion", material.occlusion),
            ] {
                validate_binding(material_index, slot, binding, texture_count, sampler_count)?;
            }
        }
        let scene = MeshGpuScene::new(
            device, queue, meshes, materials, instances, textures, samplers, sampling,
        )?;
        let max_instance_count = scene.instances().instance_count();
        let visibility = MeshVisibilityState::new(device, max_instance_count);
        let passes = MeshPassSet::new(device, color_format, &scene, &visibility);

        Ok(Self {
            scene,
            visibility,
            hiz_stage: HiZStage::new(device),
            passes,
            stats: MeshStatsReadback::new(device),
        })
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

fn validate_binding(
    material_index: usize,
    slot: &'static str,
    binding: MaterialTextureBinding,
    texture_count: usize,
    sampler_count: usize,
) -> Result<(), MeshRendererError> {
    if binding.texture_id as usize >= texture_count {
        return Err(MeshRendererError::InvalidMaterialTexture {
            material_index,
            slot,
            texture_id: binding.texture_id,
            texture_count,
        });
    }
    if binding.sampler_id as usize >= sampler_count {
        return Err(MeshRendererError::InvalidMaterialSampler {
            material_index,
            slot,
            sampler_id: binding.sampler_id,
            sampler_count,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_resource_validation_reports_texture_and_sampler_slots() {
        assert!(matches!(
            validate_binding(
                3,
                "emissive",
                MaterialTextureBinding {
                    texture_id: 2,
                    sampler_id: 0,
                },
                2,
                1,
            ),
            Err(MeshRendererError::InvalidMaterialTexture {
                material_index: 3,
                slot: "emissive",
                texture_id: 2,
                ..
            })
        ));
        assert!(matches!(
            validate_binding(
                4,
                "occlusion",
                MaterialTextureBinding {
                    texture_id: 0,
                    sampler_id: 5,
                },
                1,
                5,
            ),
            Err(MeshRendererError::InvalidMaterialSampler {
                material_index: 4,
                slot: "occlusion",
                sampler_id: 5,
                ..
            })
        ));
    }

    #[test]
    fn discarded_frame_keeps_the_stats_request_for_the_next_prepare() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: MeshRenderer::required_features(),
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 16,
                max_binding_array_sampler_elements_per_shader_stage: 4,
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
            &[],
            TextureSamplingConfig::default(),
        )
        .unwrap();
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

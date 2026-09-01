//! GPU-driven Mesh domain rendering for `zen-frame-graph`.
//!
//! [`MeshRenderer`] owns Mesh GPU resources and contributes Mesh-specific work to a caller-owned
//! FrameGraph. Surface acquisition, target allocation, graph compilation, execution, and
//! presentation remain the responsibility of the application renderer.

pub mod camera;
pub mod mesh;
pub mod meshlet;

pub use camera::{Camera, OrthographicProjection, PerspectiveProjection};
pub use mesh::{
    Instance, Material, MaterialTextureBinding, Mesh, MeshRenderInput, MeshRenderStats,
    MeshRenderTargets, MeshRenderer, MeshRendererError, PreparedMeshFrame, Texture,
    TextureAddressMode, TextureMagFilter, TextureMinFilter, TextureResourceError, TextureSampler,
    TextureSamplingConfig, Vertex,
};
pub use meshlet::{
    BindlessTextureError, BoundsSphere, FallbackTextureHandles, LodTableEntry, MeshTableEntry,
    MeshletAssetError, MeshletAssetHash, MeshletBackend, MeshletBenchmarkProfile,
    MeshletBindlessCapacity, MeshletBindlessConfig, MeshletBuildConfig, MeshletCacheKey,
    MeshletCapabilities, MeshletCapabilityError, MeshletCapacityConfig, MeshletCapacityKind,
    MeshletConfigError, MeshletDeviceRequirements, MeshletDriverBlacklist,
    MeshletDriverBlacklistEntry, MeshletDriverKey, MeshletGpuFrameTimings, MeshletGpuPassTimings,
    MeshletGpuTimingError, MeshletLimitFailure, MeshletOverflowFlags, MeshletPsoBinStats,
    MeshletPsoClass, MeshletRenderInput, MeshletRenderStats, MeshletRenderer,
    MeshletRendererConfig, MeshletRendererError, MeshletSceneAsset, MeshletTableEntry, NormalCone,
    PackedVertexAttributes, ParseMeshletBackendError, PreparedMeshletFrame, RawStaticMesh,
    TextureHandle,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_requirements_cover_mesh_bindless_and_indirect_rendering() {
        let features = MeshRenderer::required_features();
        assert!(features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY));
        assert!(features.contains(wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY));
        assert!(features.contains(
            wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        ));
        assert!(features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE));
        assert!(!features.contains(wgpu::Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn device_limits_respect_the_adapter_binding_array_limit() {
        let adapter_limits = wgpu::Limits {
            max_binding_array_elements_per_shader_stage: 64,
            max_binding_array_sampler_elements_per_shader_stage: 12,
            ..Default::default()
        };
        let limits = MeshRenderer::required_limits(&adapter_limits);
        assert_eq!(limits.max_binding_array_elements_per_shader_stage, 64);
        assert_eq!(
            limits.max_binding_array_sampler_elements_per_shader_stage,
            12
        );

        let adapter_limits = wgpu::Limits {
            max_binding_array_elements_per_shader_stage: 2048,
            max_binding_array_sampler_elements_per_shader_stage: 64,
            ..Default::default()
        };
        let limits = MeshRenderer::required_limits(&adapter_limits);
        assert_eq!(limits.max_binding_array_elements_per_shader_stage, 1024);
        assert_eq!(
            limits.max_binding_array_sampler_elements_per_shader_stage,
            32
        );
    }
}

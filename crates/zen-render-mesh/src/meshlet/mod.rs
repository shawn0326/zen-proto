pub mod asset;

mod bindless;
mod capabilities;
mod config;
mod frame;
mod gpu_scene;
mod gpu_types;
mod graph_recorder;
mod passes;
mod renderer;
mod stats;
mod stats_readback;

pub use asset::{
    BoundsSphere, LodTableEntry, MeshTableEntry, MeshletAssetError, MeshletAssetHash,
    MeshletBuildConfig, MeshletCacheKey, MeshletPsoClass, MeshletSceneAsset, MeshletTableEntry,
    NormalCone, PackedVertexAttributes, RawStaticMesh,
};
pub use bindless::{
    BindlessSamplerError, BindlessTextureError, FallbackSamplerHandles, FallbackTextureHandles,
    SamplerHandle, TextureHandle,
};
pub use capabilities::{
    MeshletBindlessCapacity, MeshletCapabilities, MeshletCapabilityError,
    MeshletDeviceRequirements, MeshletDriverBlacklist, MeshletDriverBlacklistEntry,
    MeshletDriverKey, MeshletLimitFailure,
};
pub use config::{
    MeshletBackend, MeshletBenchmarkProfile, MeshletBindlessConfig, MeshletCapacityConfig,
    MeshletConfigError, MeshletRendererConfig, ParseMeshletBackendError,
};
pub use renderer::{
    MeshletRenderInput, MeshletRenderMode, MeshletRenderer, MeshletRendererError,
    PreparedMeshletFrame,
};
pub use stats::{
    MeshletCapacityKind, MeshletGpuFrameTimings, MeshletGpuPassTimings, MeshletGpuTimingError,
    MeshletOverflowFlags, MeshletPsoBinStats, MeshletRenderStats,
};

#[cfg(test)]
mod shader_tests {
    fn validate_and_emit_spirv(label: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{label} failed WGSL parsing: {error:?}"));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{label} failed Naga validation: {error:?}"));
        let pipeline_constants = naga::back::PipelineConstants::default();
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &module,
            &info,
            None,
            &pipeline_constants,
        )
        .unwrap_or_else(|error| panic!("{label} failed pipeline-override substitution: {error:?}"));
        let options = naga::back::spv::Options {
            lang_version: (1, 6),
            ..Default::default()
        };
        let words = naga::back::spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|error| panic!("{label} failed SPIR-V emission: {error:?}"));
        assert_eq!(words.first().copied(), Some(0x0723_0203));
    }

    #[test]
    fn every_meshlet_shader_parses_validates_and_emits_spirv() {
        for (label, source) in [
            (
                "classify",
                include_str!("../../shaders/meshlet/classify.wgsl"),
            ),
            (
                "prefix_scan",
                include_str!("../../shaders/meshlet/prefix_scan.wgsl"),
            ),
            (
                "candidate_scatter",
                include_str!("../../shaders/meshlet/candidate_scatter.wgsl"),
            ),
            ("cull", include_str!("../../shaders/meshlet/cull.wgsl")),
            (
                "indirect_prepare",
                include_str!("../../shaders/meshlet/indirect_prepare.wgsl"),
            ),
            (
                "indexed",
                include_str!("../../shaders/meshlet/indexed.wgsl"),
            ),
            ("mesh", include_str!("../../shaders/meshlet/mesh.wgsl")),
        ] {
            validate_and_emit_spirv(label, source);
        }
    }

    #[test]
    fn raster_shaders_forward_flat_meshlet_debug_state() {
        for (label, source) in [
            (
                "indexed",
                include_str!("../../shaders/meshlet/indexed.wgsl"),
            ),
            ("mesh", include_str!("../../shaders/meshlet/mesh.wgsl")),
        ] {
            assert!(
                source.contains("@interpolate(flat) meshlet_id: u32"),
                "{label} does not flat-interpolate the meshlet ID"
            );
            assert!(
                source.contains("@interpolate(flat) render_mode: u32"),
                "{label} does not flat-interpolate the render mode"
            );
            assert!(
                source.contains("meshlet_debug_color(input.meshlet_id)"),
                "{label} does not shade from the global meshlet ID"
            );
        }
    }

    #[test]
    fn every_counter_shader_matches_the_rust_counter_abi() {
        let expected_offsets: [(&str, u32); 13] = [
            ("candidate_count", 0),
            ("visible_count_backface", 4),
            ("visible_count_two_sided", 8),
            ("instances_visible", 12),
            ("culled_frustum", 16),
            ("culled_cone", 20),
            ("culled_hiz", 24),
            ("output_vertices", 28),
            ("output_primitives", 32),
            ("overflow", 36),
            ("lod_histogram", 40),
            ("lod_overflow_instances", 72),
            ("conservatively_visible_meshlets", 76),
        ];

        for (label, source) in [
            (
                "classify",
                include_str!("../../shaders/meshlet/classify.wgsl"),
            ),
            (
                "prefix_scan",
                include_str!("../../shaders/meshlet/prefix_scan.wgsl"),
            ),
            (
                "candidate_scatter",
                include_str!("../../shaders/meshlet/candidate_scatter.wgsl"),
            ),
            ("cull", include_str!("../../shaders/meshlet/cull.wgsl")),
            (
                "indirect_prepare",
                include_str!("../../shaders/meshlet/indirect_prepare.wgsl"),
            ),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{label} failed WGSL parsing: {error:?}"));
            let counter_type = module
                .types
                .iter()
                .map(|(_, ty)| ty)
                .find(|ty| ty.name.as_deref() == Some("Counters"))
                .unwrap_or_else(|| panic!("{label} has no Counters type"));
            let naga::TypeInner::Struct { members, span } = &counter_type.inner else {
                panic!("{label} Counters is not a struct");
            };
            assert_eq!(
                *span,
                std::mem::size_of::<super::gpu_types::GpuCounters>() as u32
            );
            let actual_offsets = members
                .iter()
                .map(|member| (member.name.as_deref().unwrap_or(""), member.offset))
                .collect::<Vec<_>>();
            assert_eq!(
                actual_offsets, expected_offsets,
                "{label} Counters ABI drifted"
            );
        }
    }

    #[test]
    fn counter_logic_keeps_lod_overflow_and_conservative_visibility_distinct() {
        let classify = include_str!("../../shaders/meshlet/classify.wgsl");
        assert!(classify.contains("if (selected_relative < 8u)"));
        assert!(classify.contains("lod_histogram[selected_relative]"));
        assert!(classify.contains("lod_overflow_instances"));

        let cull = include_str!("../../shaders/meshlet/cull.wgsl");
        assert!(cull.contains("CullResult(false, true)"));
        assert!(cull.contains("conservatively_visible_meshlets"));
        assert!(cull.contains("if (conservatively_visible)"));

        // All raster backends consume the compute-produced visible list; TaskMesh must not recount
        // cull reasons or conservative visibility in its raster shader.
        let mesh = include_str!("../../shaders/meshlet/mesh.wgsl");
        assert!(!mesh.contains("atomicAdd(&counters.culled_"));

        let prefix_scan = include_str!("../../shaders/meshlet/prefix_scan.wgsl");
        assert!(prefix_scan.contains("atomicOr(&counters.overflow, 8u)"));
        assert!(!prefix_scan.contains("atomicOr(&counters.overflow, 16u)"));
    }

    #[test]
    fn scatter_destination_check_cannot_wrap_at_u32_capacity() {
        const fn destination(offset: u32, local: u32, capacity: u32) -> Option<u32> {
            if offset < capacity && local < capacity - offset {
                Some(offset + local)
            } else {
                None
            }
        }

        assert_eq!(destination(u32::MAX - 1, 0, u32::MAX), Some(u32::MAX - 1));
        assert_eq!(destination(u32::MAX - 1, 1, u32::MAX), None);
        assert_eq!(destination(u32::MAX, 0, u32::MAX), None);
        assert_eq!(destination(7, u32::MAX, u32::MAX), None);

        let candidate_scatter = include_str!("../../shaders/meshlet/candidate_scatter.wgsl");
        assert!(
            candidate_scatter
                .contains("offset < frame.counts.z && local < frame.counts.z - offset")
        );
    }
}

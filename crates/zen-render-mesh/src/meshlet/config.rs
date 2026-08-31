use std::{error::Error, fmt, str::FromStr};

/// Minimum benchmark warm-up used before `TaskMesh` may be selected by [`MeshletBackend::Auto`].
pub const AUTO_BENCHMARK_MIN_WARMUP_FRAMES: u32 = 120;
/// Minimum benchmark sample count used before `TaskMesh` may be selected by [`MeshletBackend::Auto`].
pub const AUTO_BENCHMARK_MIN_SAMPLE_FRAMES: u32 = 600;
/// Required `TaskMesh` p95 improvement, expressed as basis points (10%).
pub const AUTO_TASK_MESH_MIN_SPEEDUP_BPS: u32 = 1_000;

/// Number of built-in texture slots reserved by the renderer (white, black, and flat normal).
pub const MESHLET_FALLBACK_TEXTURE_SLOTS: u32 = 3;
/// Number of built-in sampler slots reserved by the renderer.
pub const MESHLET_FALLBACK_SAMPLER_SLOTS: u32 = 1;

/// Default and currently supported maximum number of unique vertices in one meshlet.
pub const MESHLET_MAX_VERTICES: u32 = 64;
/// Default and currently supported maximum number of triangles in one meshlet.
pub const MESHLET_MAX_TRIANGLES: u32 = 64;
/// Number of candidate meshlets processed by one task workgroup.
pub const TASK_PACKET_MESHLET_COUNT: u32 = 32;

/// A concrete raster backend owned by `MeshletRenderer`, or its benchmark-gated automatic mode.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MeshletBackend {
    /// Resolve to `IndexedIndirect` unless a valid local benchmark profile approves `TaskMesh`.
    #[default]
    Auto,
    /// Compute-generated indexed draws submitted with `multi_draw_indexed_indirect_count`.
    IndexedIndirect,
    /// Mesh shaders consuming the same visible-work list as `IndexedIndirect`.
    MeshOnly,
    /// Task shaders compacting meshlets into payloads consumed by mesh shaders.
    TaskMesh,
}

impl MeshletBackend {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::IndexedIndirect => "indexed",
            Self::MeshOnly => "mesh",
            Self::TaskMesh => "task-mesh",
        }
    }

    /// Returns whether this is a resolved backend that uses the experimental mesh-shader feature.
    #[must_use]
    pub const fn uses_mesh_shaders(self) -> bool {
        matches!(self, Self::MeshOnly | Self::TaskMesh)
    }

    /// Returns whether a task stage is present in the selected pipeline.
    #[must_use]
    pub const fn uses_task_shaders(self) -> bool {
        matches!(self, Self::TaskMesh)
    }

    /// Returns whether this value names a concrete renderer backend.
    #[must_use]
    pub const fn is_resolved(self) -> bool {
        !matches!(self, Self::Auto)
    }
}

impl fmt::Display for MeshletBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for MeshletBackend {
    type Err = ParseMeshletBackendError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "indexed" | "indexed-indirect" => Ok(Self::IndexedIndirect),
            "mesh" | "mesh-only" => Ok(Self::MeshOnly),
            "task-mesh" | "task_mesh" => Ok(Self::TaskMesh),
            _ => Err(ParseMeshletBackendError {
                value: value.to_owned(),
            }),
        }
    }
}

/// Error returned when parsing a meshlet backend name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseMeshletBackendError {
    value: String,
}

impl ParseMeshletBackendError {
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for ParseMeshletBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown meshlet backend {:?}; expected auto, indexed, mesh, or task-mesh",
            self.value
        )
    }
}

impl Error for ParseMeshletBackendError {}

/// Local, adapter-specific benchmark result used by [`MeshletBackend::Auto`].
///
/// Persisted profiles must be keyed by adapter, driver, renderer version, scene, resolution, and
/// camera track by the application. This type deliberately contains only the measurements needed
/// to enforce the renderer's promotion gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletBenchmarkProfile {
    pub geometry_bound: bool,
    pub warmup_frames: u32,
    pub sample_frames: u32,
    pub indexed_gpu_p95_ns: u64,
    pub task_mesh_gpu_p95_ns: u64,
}

impl MeshletBenchmarkProfile {
    /// Returns whether this profile meets the agreed gate for automatic `TaskMesh` selection.
    #[must_use]
    pub fn approves_task_mesh(self) -> bool {
        if !self.geometry_bound
            || self.warmup_frames < AUTO_BENCHMARK_MIN_WARMUP_FRAMES
            || self.sample_frames < AUTO_BENCHMARK_MIN_SAMPLE_FRAMES
            || self.indexed_gpu_p95_ns == 0
            || self.task_mesh_gpu_p95_ns == 0
        {
            return false;
        }

        let required_ratio_bps = 10_000u128 - u128::from(AUTO_TASK_MESH_MIN_SPEEDUP_BPS);
        u128::from(self.task_mesh_gpu_p95_ns) * 10_000
            <= u128::from(self.indexed_gpu_p95_ns) * required_ratio_bps
    }
}

/// Requested capacities for the fixed bindless resource tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletBindlessConfig {
    pub max_textures: u32,
    pub max_samplers: u32,
}

impl Default for MeshletBindlessConfig {
    fn default() -> Self {
        Self {
            max_textures: 4_096,
            max_samplers: 32,
        }
    }
}

/// Fixed GPU buffer capacities. Exhaustion is clamped on the GPU and reported through sticky
/// overflow flags; it never authorizes an out-of-bounds access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletCapacityConfig {
    pub max_instances: u32,
    pub max_candidate_meshlets: u32,
    pub max_visible_meshlets: u32,
    pub max_task_packets: u32,
    pub max_indirect_draws_per_bin: u32,
}

impl Default for MeshletCapacityConfig {
    fn default() -> Self {
        Self {
            max_instances: 262_144,
            max_candidate_meshlets: 2_097_152,
            max_visible_meshlets: 1_048_576,
            max_task_packets: 65_536,
            max_indirect_draws_per_bin: 524_288,
        }
    }
}

impl MeshletCapacityConfig {
    pub fn validate(self) -> Result<(), MeshletConfigError> {
        for (name, value) in [
            ("max_instances", self.max_instances),
            ("max_candidate_meshlets", self.max_candidate_meshlets),
            ("max_visible_meshlets", self.max_visible_meshlets),
            ("max_task_packets", self.max_task_packets),
            (
                "max_indirect_draws_per_bin",
                self.max_indirect_draws_per_bin,
            ),
        ] {
            if value == 0 {
                return Err(MeshletConfigError::ZeroCapacity { name });
            }
        }

        if self.max_visible_meshlets > self.max_candidate_meshlets {
            return Err(
                MeshletConfigError::VisibleCapacityExceedsCandidateCapacity {
                    visible: self.max_visible_meshlets,
                    candidate: self.max_candidate_meshlets,
                },
            );
        }

        let total_indirect_capacity = self.max_indirect_draws_per_bin.checked_mul(2).ok_or(
            MeshletConfigError::CapacityAddressSpaceOverflow {
                name: "max_indirect_draws_per_bin",
                per_bin: self.max_indirect_draws_per_bin,
            },
        )?;
        if total_indirect_capacity != self.max_visible_meshlets {
            return Err(
                MeshletConfigError::IndirectCapacityDoesNotMatchVisibleCapacity {
                    indirect_per_bin: self.max_indirect_draws_per_bin,
                    visible: self.max_visible_meshlets,
                },
            );
        }
        if self
            .max_visible_meshlets
            .checked_mul(MESHLET_MAX_VERTICES.max(MESHLET_MAX_TRIANGLES))
            .is_none()
        {
            return Err(MeshletConfigError::StatisticsCounterCapacityOverflow {
                visible: self.max_visible_meshlets,
                maximum_per_meshlet: MESHLET_MAX_VERTICES.max(MESHLET_MAX_TRIANGLES),
            });
        }

        self.max_task_packets.checked_mul(2).ok_or(
            MeshletConfigError::CapacityAddressSpaceOverflow {
                name: "max_task_packets",
                per_bin: self.max_task_packets,
            },
        )?;

        Ok(())
    }
}

/// Construction policy for `MeshletRenderer`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletRendererConfig {
    pub backend: MeshletBackend,
    pub bindless: MeshletBindlessConfig,
    pub capacities: MeshletCapacityConfig,
    /// A profile for the exact local adapter/driver and current benchmark version.
    ///
    /// `None` is an unknown adapter and therefore resolves `Auto` to `IndexedIndirect`.
    pub auto_benchmark_profile: Option<MeshletBenchmarkProfile>,
}

impl Default for MeshletRendererConfig {
    fn default() -> Self {
        Self {
            backend: MeshletBackend::Auto,
            bindless: MeshletBindlessConfig::default(),
            capacities: MeshletCapacityConfig::default(),
            auto_benchmark_profile: None,
        }
    }
}

impl MeshletRendererConfig {
    pub fn validate(self) -> Result<(), MeshletConfigError> {
        if self.bindless.max_textures < MESHLET_FALLBACK_TEXTURE_SLOTS {
            return Err(MeshletConfigError::InsufficientFallbackSlots {
                resource: "textures",
                requested: self.bindless.max_textures,
                required: MESHLET_FALLBACK_TEXTURE_SLOTS,
            });
        }
        if self.bindless.max_samplers < MESHLET_FALLBACK_SAMPLER_SLOTS {
            return Err(MeshletConfigError::InsufficientFallbackSlots {
                resource: "samplers",
                requested: self.bindless.max_samplers,
                required: MESHLET_FALLBACK_SAMPLER_SLOTS,
            });
        }
        self.capacities.validate()
    }

    #[must_use]
    pub fn benchmark_approves_task_mesh(self) -> bool {
        self.auto_benchmark_profile
            .is_some_and(MeshletBenchmarkProfile::approves_task_mesh)
    }
}

/// Invalid renderer configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshletConfigError {
    ZeroCapacity {
        name: &'static str,
    },
    InsufficientFallbackSlots {
        resource: &'static str,
        requested: u32,
        required: u32,
    },
    VisibleCapacityExceedsCandidateCapacity {
        visible: u32,
        candidate: u32,
    },
    IndirectCapacityDoesNotMatchVisibleCapacity {
        indirect_per_bin: u32,
        visible: u32,
    },
    StatisticsCounterCapacityOverflow {
        visible: u32,
        maximum_per_meshlet: u32,
    },
    CapacityAddressSpaceOverflow {
        name: &'static str,
        per_bin: u32,
    },
}

impl fmt::Display for MeshletConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroCapacity { name } => {
                write!(formatter, "meshlet capacity {name} must be non-zero")
            }
            Self::InsufficientFallbackSlots {
                resource,
                requested,
                required,
            } => write!(
                formatter,
                "bindless {resource} capacity {requested} cannot hold {required} fallback slots"
            ),
            Self::VisibleCapacityExceedsCandidateCapacity { visible, candidate } => write!(
                formatter,
                "visible meshlet capacity {visible} exceeds candidate capacity {candidate}"
            ),
            Self::IndirectCapacityDoesNotMatchVisibleCapacity {
                indirect_per_bin,
                visible,
            } => write!(
                formatter,
                "visible meshlet capacity {visible} must equal two PSO bins at {indirect_per_bin} draws each"
            ),
            Self::StatisticsCounterCapacityOverflow {
                visible,
                maximum_per_meshlet,
            } => write!(
                formatter,
                "visible capacity {visible} times {maximum_per_meshlet} outputs per meshlet exceeds the u32 GPU statistics ABI"
            ),
            Self::CapacityAddressSpaceOverflow { name, per_bin } => write!(
                formatter,
                "two PSO bins at {per_bin} entries each overflow the u32 {name} address space"
            ),
        }
    }
}

impl Error for MeshletConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualifying_profile() -> MeshletBenchmarkProfile {
        MeshletBenchmarkProfile {
            geometry_bound: true,
            warmup_frames: 120,
            sample_frames: 600,
            indexed_gpu_p95_ns: 10_000,
            task_mesh_gpu_p95_ns: 9_000,
        }
    }

    #[test]
    fn backend_names_match_the_demo_cli() {
        for (name, expected) in [
            ("auto", MeshletBackend::Auto),
            ("indexed", MeshletBackend::IndexedIndirect),
            ("mesh", MeshletBackend::MeshOnly),
            ("task-mesh", MeshletBackend::TaskMesh),
        ] {
            assert_eq!(name.parse(), Ok(expected));
            assert_eq!(expected.as_str(), name);
        }
        assert!("legacy".parse::<MeshletBackend>().is_err());
    }

    #[test]
    fn auto_profile_enforces_the_full_promotion_gate() {
        assert!(qualifying_profile().approves_task_mesh());

        let mut profile = qualifying_profile();
        profile.task_mesh_gpu_p95_ns = 9_001;
        assert!(!profile.approves_task_mesh());

        let mut profile = qualifying_profile();
        profile.geometry_bound = false;
        assert!(!profile.approves_task_mesh());

        let mut profile = qualifying_profile();
        profile.warmup_frames = 119;
        assert!(!profile.approves_task_mesh());

        let mut profile = qualifying_profile();
        profile.sample_frames = 599;
        assert!(!profile.approves_task_mesh());
    }

    #[test]
    fn benchmark_math_does_not_overflow_u64() {
        let profile = MeshletBenchmarkProfile {
            indexed_gpu_p95_ns: u64::MAX,
            task_mesh_gpu_p95_ns: u64::MAX / 2,
            ..qualifying_profile()
        };
        assert!(profile.approves_task_mesh());
    }

    #[test]
    fn default_config_reserves_all_fallback_slots() {
        let config = MeshletRendererConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.bindless.max_textures >= MESHLET_FALLBACK_TEXTURE_SLOTS);
        assert!(config.bindless.max_samplers >= MESHLET_FALLBACK_SAMPLER_SLOTS);
        assert!(!config.benchmark_approves_task_mesh());
    }

    #[test]
    fn invalid_capacity_relationships_are_rejected() {
        let capacities = MeshletCapacityConfig {
            max_candidate_meshlets: 10,
            max_visible_meshlets: 11,
            ..Default::default()
        };
        assert!(matches!(
            capacities.validate(),
            Err(MeshletConfigError::VisibleCapacityExceedsCandidateCapacity { .. })
        ));

        let capacities = MeshletCapacityConfig {
            max_candidate_meshlets: 100,
            max_visible_meshlets: 80,
            max_indirect_draws_per_bin: 41,
            ..Default::default()
        };
        assert!(matches!(
            capacities.validate(),
            Err(MeshletConfigError::IndirectCapacityDoesNotMatchVisibleCapacity { .. })
        ));

        let capacities = MeshletCapacityConfig {
            max_candidate_meshlets: u32::MAX,
            max_visible_meshlets: u32::MAX,
            max_indirect_draws_per_bin: u32::MAX,
            ..Default::default()
        };
        assert!(matches!(
            capacities.validate(),
            Err(MeshletConfigError::CapacityAddressSpaceOverflow {
                name: "max_indirect_draws_per_bin",
                ..
            })
        ));

        let capacities = MeshletCapacityConfig {
            max_candidate_meshlets: 100,
            max_visible_meshlets: 2,
            max_indirect_draws_per_bin: 1,
            max_task_packets: u32::MAX,
            ..Default::default()
        };
        assert!(matches!(
            capacities.validate(),
            Err(MeshletConfigError::CapacityAddressSpaceOverflow {
                name: "max_task_packets",
                ..
            })
        ));

        let capacities = MeshletCapacityConfig {
            max_candidate_meshlets: 67_108_864,
            max_visible_meshlets: 67_108_864,
            max_indirect_draws_per_bin: 33_554_432,
            max_task_packets: 1,
            ..Default::default()
        };
        assert!(matches!(
            capacities.validate(),
            Err(MeshletConfigError::StatisticsCounterCapacityOverflow { .. })
        ));
    }
}

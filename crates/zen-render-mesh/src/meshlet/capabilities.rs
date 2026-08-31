use std::{error::Error, fmt};

use super::config::{
    MESHLET_FALLBACK_SAMPLER_SLOTS, MESHLET_FALLBACK_TEXTURE_SLOTS, MESHLET_MAX_TRIANGLES,
    MESHLET_MAX_VERTICES, MeshletBackend, MeshletConfigError, MeshletRendererConfig,
    TASK_PACKET_MESHLET_COUNT,
};

const MESH_WORKGROUP_INVOCATIONS: u32 = if MESHLET_MAX_VERTICES > MESHLET_MAX_TRIANGLES {
    MESHLET_MAX_VERTICES
} else {
    MESHLET_MAX_TRIANGLES
};
const TASK_PAYLOAD_WORK_BYTES: u32 = 16;
const TASK_PAYLOAD_BYTES: u32 = TASK_PACKET_MESHLET_COUNT * TASK_PAYLOAD_WORK_BYTES;

/// Stable Vulkan driver identity used by persisted benchmark profiles and explicit deny rules.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct MeshletDriverKey {
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub driver_info: String,
}

impl MeshletDriverKey {
    #[must_use]
    pub fn stable_id(&self) -> String {
        format!(
            "{:08x}:{:08x}:{}:{}:{}:{}",
            self.vendor,
            self.device,
            self.driver.len(),
            self.driver,
            self.driver_info.len(),
            self.driver_info
        )
    }
}

/// One exact, auditable driver/backend deny rule. The built-in list is intentionally empty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshletDriverBlacklistEntry {
    pub key: MeshletDriverKey,
    pub backend: MeshletBackend,
    pub reason: String,
}

/// Application-supplied Vulkan driver deny rules. No unknown driver is denied by default.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeshletDriverBlacklist {
    pub entries: Vec<MeshletDriverBlacklistEntry>,
}

impl MeshletDriverBlacklist {
    fn reason_for(&self, key: &MeshletDriverKey, backend: MeshletBackend) -> Option<&str> {
        self.entries
            .iter()
            .find(|entry| entry.backend == backend && entry.key == *key)
            .map(|entry| entry.reason.as_str())
    }
}

/// Effective fixed-table sizes after clamping the renderer request to adapter limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletBindlessCapacity {
    pub textures: u32,
    pub samplers: u32,
}

/// Adapter capabilities relevant to the Vulkan-only meshlet renderer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshletCapabilities {
    backend: wgpu::Backend,
    features: wgpu::Features,
    limits: wgpu::Limits,
    downlevel_flags: wgpu::DownlevelFlags,
    driver_key: MeshletDriverKey,
}

impl MeshletCapabilities {
    #[must_use]
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        Self {
            backend: info.backend,
            features: adapter.features(),
            limits: adapter.limits(),
            downlevel_flags: adapter.get_downlevel_capabilities().flags,
            driver_key: MeshletDriverKey {
                vendor: info.vendor,
                device: info.device,
                driver: info.driver,
                driver_info: info.driver_info,
            },
        }
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn from_parts(
        backend: wgpu::Backend,
        features: wgpu::Features,
        limits: wgpu::Limits,
    ) -> Self {
        Self {
            backend,
            features,
            limits,
            downlevel_flags: wgpu::DownlevelFlags::INDIRECT_EXECUTION,
            driver_key: MeshletDriverKey::default(),
        }
    }

    /// Pure-data constructor for tests and serialized adapter snapshots with explicit downlevel
    /// capabilities.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn from_parts_with_downlevel(
        backend: wgpu::Backend,
        features: wgpu::Features,
        limits: wgpu::Limits,
        downlevel_flags: wgpu::DownlevelFlags,
    ) -> Self {
        Self {
            backend,
            features,
            limits,
            downlevel_flags,
            driver_key: MeshletDriverKey::default(),
        }
    }

    #[must_use]
    pub const fn backend(&self) -> wgpu::Backend {
        self.backend
    }

    #[must_use]
    pub const fn features(&self) -> wgpu::Features {
        self.features
    }

    #[must_use]
    pub const fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    #[must_use]
    pub const fn downlevel_flags(&self) -> wgpu::DownlevelFlags {
        self.downlevel_flags
    }

    #[must_use]
    pub const fn driver_key(&self) -> &MeshletDriverKey {
        &self.driver_key
    }

    /// Validates one concrete backend. `Auto` must be resolved with [`Self::resolve_backend`].
    pub fn validate_backend(&self, backend: MeshletBackend) -> Result<(), MeshletCapabilityError> {
        self.validate_vulkan()?;
        if !backend.is_resolved() {
            return Err(MeshletCapabilityError::UnresolvedBackend);
        }

        let missing_downlevel =
            MeshletDeviceRequirements::required_downlevel_flags(backend)? - self.downlevel_flags;
        if !missing_downlevel.is_empty() {
            return Err(MeshletCapabilityError::MissingDownlevelFlags {
                backend,
                missing: missing_downlevel,
            });
        }

        let missing = MeshletDeviceRequirements::required_features(backend)? - self.features;
        if !missing.is_empty() {
            return Err(MeshletCapabilityError::MissingFeatures { backend, missing });
        }

        let failures = mesh_shader_limit_failures(backend, &self.limits);
        if !failures.is_empty() {
            return Err(MeshletCapabilityError::InsufficientLimits { backend, failures });
        }

        Ok(())
    }

    /// Returns whether a concrete backend is available. `Auto` means its conservative,
    /// unprofiled result (`IndexedIndirect`).
    #[must_use]
    pub fn supports_backend(&self, backend: MeshletBackend) -> bool {
        let backend = match backend {
            MeshletBackend::Auto => MeshletBackend::IndexedIndirect,
            concrete => concrete,
        };
        self.validate_backend(backend).is_ok()
    }

    /// Resolves `Auto` once at startup; it never creates a per-pass conditional backend tree.
    ///
    /// A qualifying benchmark profile promotes `Auto` to `TaskMesh` only when the adapter also
    /// supports that backend. Unknown, rejected, stale, or unsupported profiles fall back to
    /// `IndexedIndirect`.
    pub fn resolve_backend(
        &self,
        config: &MeshletRendererConfig,
    ) -> Result<MeshletBackend, MeshletCapabilityError> {
        self.resolve_backend_with_blacklist(config, &MeshletDriverBlacklist::default())
    }

    /// Resolves a backend while applying exact application-owned driver deny rules.
    ///
    /// A denied `TaskMesh` candidate in `Auto` mode falls back to `IndexedIndirect`. An explicitly
    /// selected denied backend returns a clear error.
    pub fn resolve_backend_with_blacklist(
        &self,
        config: &MeshletRendererConfig,
        blacklist: &MeshletDriverBlacklist,
    ) -> Result<MeshletBackend, MeshletCapabilityError> {
        config.validate()?;
        self.validate_vulkan()?;

        let mut resolved = match config.backend {
            MeshletBackend::Auto
                if config.benchmark_approves_task_mesh()
                    && self.validate_backend(MeshletBackend::TaskMesh).is_ok() =>
            {
                MeshletBackend::TaskMesh
            }
            MeshletBackend::Auto => MeshletBackend::IndexedIndirect,
            concrete => concrete,
        };

        if let Some(reason) = blacklist.reason_for(&self.driver_key, resolved) {
            if config.backend == MeshletBackend::Auto && resolved == MeshletBackend::TaskMesh {
                resolved = MeshletBackend::IndexedIndirect;
            } else {
                return Err(MeshletCapabilityError::BlacklistedDriver {
                    key: self.driver_key.clone(),
                    backend: resolved,
                    reason: reason.to_owned(),
                });
            }
        }

        if let Some(reason) = blacklist.reason_for(&self.driver_key, resolved) {
            return Err(MeshletCapabilityError::BlacklistedDriver {
                key: self.driver_key.clone(),
                backend: resolved,
                reason: reason.to_owned(),
            });
        }

        self.validate_backend(resolved)?;
        Ok(resolved)
    }

    /// Produces the exact device request for the resolved backend and adapter-clamped bindless
    /// table sizes.
    pub fn device_requirements(
        &self,
        config: &MeshletRendererConfig,
    ) -> Result<MeshletDeviceRequirements, MeshletCapabilityError> {
        self.device_requirements_with_blacklist(config, &MeshletDriverBlacklist::default())
    }

    /// Produces a device request after resolving `Auto` against explicit driver deny rules.
    pub fn device_requirements_with_blacklist(
        &self,
        config: &MeshletRendererConfig,
        blacklist: &MeshletDriverBlacklist,
    ) -> Result<MeshletDeviceRequirements, MeshletCapabilityError> {
        let backend = self.resolve_backend_with_blacklist(config, blacklist)?;
        let bindless = self.bindless_capacity(config)?;
        let required_features = MeshletDeviceRequirements::required_features(backend)?;

        let mut required_limits = wgpu::Limits {
            // wgpu-core currently validates the stage-wide BindingArrayElements total across
            // texture and sampler arrays, in addition to the sampler-specific limit.
            max_binding_array_elements_per_shader_stage: bindless
                .textures
                .checked_add(bindless.samplers)
                .expect("adapter-clamped bindless counts cannot overflow"),
            max_binding_array_sampler_elements_per_shader_stage: bindless.samplers,
            // Static asset sizes are not known until after device creation. Request the Vulkan
            // adapter's advertised buffer limits, then reject an oversized asset explicitly in
            // MeshletRenderer::new instead of letting resource creation fail validation.
            max_storage_buffer_binding_size: self.limits.max_storage_buffer_binding_size,
            max_buffer_size: self.limits.max_buffer_size,
            ..wgpu::Limits::default()
        };

        if backend.uses_mesh_shaders() {
            required_limits.max_mesh_workgroup_total_count =
                self.limits.max_mesh_workgroup_total_count;
            required_limits.max_mesh_workgroups_per_dimension =
                self.limits.max_mesh_workgroups_per_dimension;
            required_limits.max_mesh_invocations_per_workgroup =
                self.limits.max_mesh_invocations_per_workgroup;
            required_limits.max_mesh_invocations_per_dimension =
                self.limits.max_mesh_invocations_per_dimension;
            required_limits.max_mesh_output_vertices = self.limits.max_mesh_output_vertices;
            required_limits.max_mesh_output_primitives = self.limits.max_mesh_output_primitives;
        }

        if backend.uses_task_shaders() {
            required_limits.max_task_workgroup_total_count =
                self.limits.max_task_workgroup_total_count;
            required_limits.max_task_workgroups_per_dimension =
                self.limits.max_task_workgroups_per_dimension;
            required_limits.max_task_invocations_per_workgroup =
                self.limits.max_task_invocations_per_workgroup;
            required_limits.max_task_invocations_per_dimension =
                self.limits.max_task_invocations_per_dimension;
            required_limits.max_task_payload_size = self.limits.max_task_payload_size;
        }

        debug_assert!(required_limits.check_limits(&self.limits));

        Ok(MeshletDeviceRequirements {
            adapter_backend: self.backend,
            driver_key: self.driver_key.clone(),
            source_config: *config,
            backend,
            required_features,
            required_downlevel_flags: MeshletDeviceRequirements::required_downlevel_flags(backend)?,
            required_limits,
            bindless,
        })
    }

    #[must_use]
    pub fn missing_features(&self, backend: MeshletBackend) -> wgpu::Features {
        let backend = match backend {
            MeshletBackend::Auto => MeshletBackend::IndexedIndirect,
            concrete => concrete,
        };
        MeshletDeviceRequirements::required_features(backend)
            .unwrap_or_else(|_| wgpu::Features::empty())
            - self.features
    }

    #[must_use]
    pub fn missing_downlevel_flags(&self, backend: MeshletBackend) -> wgpu::DownlevelFlags {
        let backend = match backend {
            MeshletBackend::Auto => MeshletBackend::IndexedIndirect,
            concrete => concrete,
        };
        MeshletDeviceRequirements::required_downlevel_flags(backend)
            .unwrap_or_else(|_| wgpu::DownlevelFlags::empty())
            - self.downlevel_flags
    }

    fn validate_vulkan(&self) -> Result<(), MeshletCapabilityError> {
        if self.backend != wgpu::Backend::Vulkan {
            return Err(MeshletCapabilityError::UnsupportedWgpuBackend {
                actual: self.backend,
            });
        }
        Ok(())
    }

    fn bindless_capacity(
        &self,
        config: &MeshletRendererConfig,
    ) -> Result<MeshletBindlessCapacity, MeshletCapabilityError> {
        let general_limit = self.limits.max_binding_array_elements_per_shader_stage;
        let sampler_limit = self
            .limits
            .max_binding_array_sampler_elements_per_shader_stage;

        let sampler_budget = general_limit.saturating_sub(MESHLET_FALLBACK_TEXTURE_SLOTS);
        let samplers = config
            .bindless
            .max_samplers
            .min(sampler_limit)
            .min(sampler_budget);
        let textures = config
            .bindless
            .max_textures
            .min(general_limit.saturating_sub(samplers));

        if textures < MESHLET_FALLBACK_TEXTURE_SLOTS {
            return Err(MeshletCapabilityError::InsufficientBindlessCapacity {
                resource: "textures",
                supported: textures,
                required: MESHLET_FALLBACK_TEXTURE_SLOTS,
            });
        }
        if samplers < MESHLET_FALLBACK_SAMPLER_SLOTS {
            return Err(MeshletCapabilityError::InsufficientBindlessCapacity {
                resource: "samplers",
                supported: samplers,
                required: MESHLET_FALLBACK_SAMPLER_SLOTS,
            });
        }

        Ok(MeshletBindlessCapacity { textures, samplers })
    }
}

/// Device features, limits, and effective resource-table sizes for one resolved backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshletDeviceRequirements {
    adapter_backend: wgpu::Backend,
    driver_key: MeshletDriverKey,
    source_config: MeshletRendererConfig,
    backend: MeshletBackend,
    required_features: wgpu::Features,
    /// Checked on the adapter before requesting a device; this is not a device descriptor field.
    required_downlevel_flags: wgpu::DownlevelFlags,
    required_limits: wgpu::Limits,
    bindless: MeshletBindlessCapacity,
}

impl MeshletDeviceRequirements {
    #[must_use]
    pub const fn adapter_backend(&self) -> wgpu::Backend {
        self.adapter_backend
    }

    #[must_use]
    pub const fn driver_key(&self) -> &MeshletDriverKey {
        &self.driver_key
    }

    #[must_use]
    pub const fn source_config(&self) -> MeshletRendererConfig {
        self.source_config
    }

    #[must_use]
    pub const fn backend(&self) -> MeshletBackend {
        self.backend
    }

    #[must_use]
    pub const fn features(&self) -> wgpu::Features {
        self.required_features
    }

    #[must_use]
    pub const fn downlevel_flags(&self) -> wgpu::DownlevelFlags {
        self.required_downlevel_flags
    }

    #[must_use]
    pub const fn limits(&self) -> &wgpu::Limits {
        &self.required_limits
    }

    #[must_use]
    pub const fn bindless_capacity(&self) -> MeshletBindlessCapacity {
        self.bindless
    }

    /// Returns the downlevel capabilities shared by every concrete meshlet backend.
    pub fn required_downlevel_flags(
        backend: MeshletBackend,
    ) -> Result<wgpu::DownlevelFlags, MeshletCapabilityError> {
        if !backend.is_resolved() {
            return Err(MeshletCapabilityError::UnresolvedBackend);
        }
        Ok(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
    }

    /// Returns the feature set for one concrete meshlet backend.
    pub fn required_features(
        backend: MeshletBackend,
    ) -> Result<wgpu::Features, MeshletCapabilityError> {
        if !backend.is_resolved() {
            return Err(MeshletCapabilityError::UnresolvedBackend);
        }

        let indexed = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::INDIRECT_FIRST_INSTANCE;

        Ok(if backend.uses_mesh_shaders() {
            indexed | wgpu::Features::EXPERIMENTAL_MESH_SHADER
        } else {
            indexed
        })
    }

    #[must_use]
    pub const fn requires_experimental_features(&self) -> bool {
        self.backend.uses_mesh_shaders()
    }

    /// Returns the token to place in `wgpu::DeviceDescriptor::experimental_features`.
    ///
    /// Indexed requirements return a disabled token. Mesh requirements return an enabled token,
    /// so callers cannot opt into wgpu's experimental API without an explicit unsafe block.
    ///
    /// # Safety
    ///
    /// For a mesh backend, the caller must accept wgpu's experimental feature contract, including
    /// the possibility of implementation bugs causing undefined behavior, and should keep Vulkan
    /// validation enabled during development.
    #[must_use]
    pub unsafe fn experimental_features_token(&self) -> wgpu::ExperimentalFeatures {
        if self.requires_experimental_features() {
            // SAFETY: This function is unsafe specifically so the application retains the explicit
            // acknowledgement required by wgpu instead of having it hidden in renderer setup.
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        }
    }
}

/// One failed numeric adapter-limit requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletLimitFailure {
    pub name: &'static str,
    pub required: u32,
    pub supported: u32,
}

/// Capability or device-request failure for the Vulkan-only renderer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshletCapabilityError {
    InvalidConfig(MeshletConfigError),
    UnsupportedWgpuBackend {
        actual: wgpu::Backend,
    },
    BlacklistedDriver {
        key: MeshletDriverKey,
        backend: MeshletBackend,
        reason: String,
    },
    UnresolvedBackend,
    MissingFeatures {
        backend: MeshletBackend,
        missing: wgpu::Features,
    },
    MissingDownlevelFlags {
        backend: MeshletBackend,
        missing: wgpu::DownlevelFlags,
    },
    InsufficientLimits {
        backend: MeshletBackend,
        failures: Vec<MeshletLimitFailure>,
    },
    InsufficientBindlessCapacity {
        resource: &'static str,
        supported: u32,
        required: u32,
    },
}

impl From<MeshletConfigError> for MeshletCapabilityError {
    fn from(error: MeshletConfigError) -> Self {
        Self::InvalidConfig(error)
    }
}

impl fmt::Display for MeshletCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(formatter, "invalid meshlet renderer config: {error}"),
            Self::UnsupportedWgpuBackend { actual } => write!(
                formatter,
                "MeshletRenderer currently supports only Vulkan; selected wgpu backend is {actual}"
            ),
            Self::BlacklistedDriver {
                key,
                backend,
                reason,
            } => write!(
                formatter,
                "meshlet backend {backend} is disabled for Vulkan driver {}: {reason}",
                key.stable_id()
            ),
            Self::UnresolvedBackend => formatter.write_str(
                "MeshletBackend::Auto must be resolved against adapter capabilities and a benchmark profile",
            ),
            Self::MissingFeatures { backend, missing } => write!(
                formatter,
                "adapter is missing features required by meshlet backend {backend}: {missing:?}"
            ),
            Self::MissingDownlevelFlags { backend, missing } => write!(
                formatter,
                "adapter is missing downlevel capabilities required by meshlet backend {backend}: {missing:?}"
            ),
            Self::InsufficientLimits { backend, failures } => {
                write!(formatter, "adapter limits are insufficient for {backend}")?;
                for failure in failures {
                    write!(
                        formatter,
                        "; {} requires {}, supports {}",
                        failure.name, failure.required, failure.supported
                    )?;
                }
                Ok(())
            }
            Self::InsufficientBindlessCapacity {
                resource,
                supported,
                required,
            } => write!(
                formatter,
                "adapter bindless {resource} capacity {supported} is below the {required} reserved fallback slots"
            ),
        }
    }
}

impl Error for MeshletCapabilityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfig(error) => Some(error),
            _ => None,
        }
    }
}

fn mesh_shader_limit_failures(
    backend: MeshletBackend,
    limits: &wgpu::Limits,
) -> Vec<MeshletLimitFailure> {
    if !backend.uses_mesh_shaders() {
        return Vec::new();
    }

    let mut failures = Vec::new();
    check_limit(
        &mut failures,
        "max_mesh_workgroup_total_count",
        if backend.uses_task_shaders() {
            TASK_PACKET_MESHLET_COUNT
        } else {
            1
        },
        limits.max_mesh_workgroup_total_count,
    );
    check_limit(
        &mut failures,
        "max_mesh_workgroups_per_dimension",
        if backend.uses_task_shaders() {
            TASK_PACKET_MESHLET_COUNT
        } else {
            1
        },
        limits.max_mesh_workgroups_per_dimension,
    );
    check_limit(
        &mut failures,
        "max_mesh_invocations_per_workgroup",
        MESH_WORKGROUP_INVOCATIONS,
        limits.max_mesh_invocations_per_workgroup,
    );
    check_limit(
        &mut failures,
        "max_mesh_invocations_per_dimension",
        MESH_WORKGROUP_INVOCATIONS,
        limits.max_mesh_invocations_per_dimension,
    );
    check_limit(
        &mut failures,
        "max_mesh_output_vertices",
        MESHLET_MAX_VERTICES,
        limits.max_mesh_output_vertices,
    );
    check_limit(
        &mut failures,
        "max_mesh_output_primitives",
        MESHLET_MAX_TRIANGLES,
        limits.max_mesh_output_primitives,
    );

    if backend.uses_task_shaders() {
        check_limit(
            &mut failures,
            "max_task_workgroup_total_count",
            1,
            limits.max_task_workgroup_total_count,
        );
        check_limit(
            &mut failures,
            "max_task_workgroups_per_dimension",
            1,
            limits.max_task_workgroups_per_dimension,
        );
        check_limit(
            &mut failures,
            "max_task_invocations_per_workgroup",
            TASK_PACKET_MESHLET_COUNT,
            limits.max_task_invocations_per_workgroup,
        );
        check_limit(
            &mut failures,
            "max_task_invocations_per_dimension",
            TASK_PACKET_MESHLET_COUNT,
            limits.max_task_invocations_per_dimension,
        );
        check_limit(
            &mut failures,
            "max_task_payload_size",
            TASK_PAYLOAD_BYTES,
            limits.max_task_payload_size,
        );
    }

    failures
}

fn check_limit(
    failures: &mut Vec<MeshletLimitFailure>,
    name: &'static str,
    required: u32,
    supported: u32,
) {
    if supported < required {
        failures.push(MeshletLimitFailure {
            name,
            required,
            supported,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet::config::MeshletBenchmarkProfile;

    fn indexed_features() -> wgpu::Features {
        MeshletDeviceRequirements::required_features(MeshletBackend::IndexedIndirect).unwrap()
    }

    fn mesh_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_binding_array_elements_per_shader_stage: 8_192,
            max_binding_array_sampler_elements_per_shader_stage: 1_024,
            ..wgpu::Limits::default().using_recommended_minimum_mesh_shader_values()
        }
    }

    fn capabilities(features: wgpu::Features) -> MeshletCapabilities {
        MeshletCapabilities::from_parts(wgpu::Backend::Vulkan, features, mesh_limits())
    }

    fn qualifying_profile() -> MeshletBenchmarkProfile {
        MeshletBenchmarkProfile {
            geometry_bound: true,
            warmup_frames: 120,
            sample_frames: 600,
            indexed_gpu_p95_ns: 100,
            task_mesh_gpu_p95_ns: 90,
        }
    }

    #[test]
    fn indexed_feature_tier_does_not_enable_experimental_mesh_shaders() {
        let indexed = indexed_features();
        assert!(indexed.contains(wgpu::Features::TEXTURE_BINDING_ARRAY));
        assert!(indexed.contains(wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY));
        assert!(indexed.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(indexed.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE));
        assert!(!indexed.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER));
    }

    #[test]
    fn mesh_tiers_add_the_experimental_feature() {
        for backend in [MeshletBackend::MeshOnly, MeshletBackend::TaskMesh] {
            let required = MeshletDeviceRequirements::required_features(backend).unwrap();
            assert!(required.contains(indexed_features()));
            assert!(required.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER));
        }
    }

    #[test]
    fn non_vulkan_backends_are_rejected_even_with_all_features() {
        let capabilities = MeshletCapabilities::from_parts(
            wgpu::Backend::Dx12,
            wgpu::Features::all(),
            mesh_limits(),
        );
        assert_eq!(
            capabilities.validate_backend(MeshletBackend::IndexedIndirect),
            Err(MeshletCapabilityError::UnsupportedWgpuBackend {
                actual: wgpu::Backend::Dx12
            })
        );
    }

    #[test]
    fn every_concrete_backend_requires_indirect_execution() {
        for backend in [
            MeshletBackend::IndexedIndirect,
            MeshletBackend::MeshOnly,
            MeshletBackend::TaskMesh,
        ] {
            assert_eq!(
                MeshletDeviceRequirements::required_downlevel_flags(backend).unwrap(),
                wgpu::DownlevelFlags::INDIRECT_EXECUTION
            );
        }

        let capabilities = MeshletCapabilities::from_parts_with_downlevel(
            wgpu::Backend::Vulkan,
            indexed_features(),
            mesh_limits(),
            wgpu::DownlevelFlags::empty(),
        );
        assert_eq!(
            capabilities.validate_backend(MeshletBackend::IndexedIndirect),
            Err(MeshletCapabilityError::MissingDownlevelFlags {
                backend: MeshletBackend::IndexedIndirect,
                missing: wgpu::DownlevelFlags::INDIRECT_EXECUTION,
            })
        );
    }

    #[test]
    fn unknown_auto_profile_selects_indexed() {
        let capabilities = capabilities(indexed_features());
        let resolved = capabilities
            .resolve_backend(&MeshletRendererConfig::default())
            .unwrap();
        assert_eq!(resolved, MeshletBackend::IndexedIndirect);
    }

    #[test]
    fn qualifying_auto_profile_selects_task_mesh_when_supported() {
        let capabilities =
            capabilities(indexed_features() | wgpu::Features::EXPERIMENTAL_MESH_SHADER);
        let config = MeshletRendererConfig {
            auto_benchmark_profile: Some(qualifying_profile()),
            ..Default::default()
        };
        assert_eq!(
            capabilities.resolve_backend(&config).unwrap(),
            MeshletBackend::TaskMesh
        );
    }

    #[test]
    fn qualifying_profile_falls_back_when_mesh_shader_is_unavailable() {
        let capabilities = capabilities(indexed_features());
        let config = MeshletRendererConfig {
            auto_benchmark_profile: Some(qualifying_profile()),
            ..Default::default()
        };
        assert_eq!(
            capabilities.resolve_backend(&config).unwrap(),
            MeshletBackend::IndexedIndirect
        );
    }

    #[test]
    fn task_mesh_checks_task_limits_in_addition_to_mesh_limits() {
        let mut limits = mesh_limits();
        limits.max_task_payload_size = TASK_PAYLOAD_BYTES - 1;
        let capabilities = MeshletCapabilities::from_parts(
            wgpu::Backend::Vulkan,
            indexed_features() | wgpu::Features::EXPERIMENTAL_MESH_SHADER,
            limits,
        );
        let error = capabilities
            .validate_backend(MeshletBackend::TaskMesh)
            .unwrap_err();
        let MeshletCapabilityError::InsufficientLimits { failures, .. } = error else {
            panic!("expected an insufficient-limits error")
        };
        assert!(failures.iter().any(|failure| {
            failure.name == "max_task_payload_size"
                && failure.required == 32 * 16
                && failure.required == TASK_PAYLOAD_BYTES
        }));
    }

    #[test]
    fn device_request_clamps_bindless_tables_to_adapter_limits() {
        let mut limits = mesh_limits();
        limits.max_binding_array_elements_per_shader_stage = 64;
        limits.max_binding_array_sampler_elements_per_shader_stage = 16;
        let capabilities =
            MeshletCapabilities::from_parts(wgpu::Backend::Vulkan, indexed_features(), limits);
        let requirements = capabilities
            .device_requirements(&MeshletRendererConfig::default())
            .unwrap();
        assert_eq!(
            requirements.bindless,
            MeshletBindlessCapacity {
                textures: 48,
                samplers: 16
            }
        );
        assert_eq!(
            requirements
                .required_limits
                .max_binding_array_elements_per_shader_stage,
            64
        );
    }

    #[test]
    fn only_mesh_backends_require_the_experimental_token() {
        let indexed = capabilities(indexed_features())
            .device_requirements(&MeshletRendererConfig {
                backend: MeshletBackend::IndexedIndirect,
                ..Default::default()
            })
            .unwrap();
        assert!(!indexed.requires_experimental_features());
        // SAFETY: The indexed path returns only the disabled token.
        assert!(!unsafe { indexed.experimental_features_token() }.is_enabled());

        let mesh = capabilities(indexed_features() | wgpu::Features::EXPERIMENTAL_MESH_SHADER)
            .device_requirements(&MeshletRendererConfig {
                backend: MeshletBackend::MeshOnly,
                ..Default::default()
            })
            .unwrap();
        assert!(mesh.requires_experimental_features());
    }

    #[test]
    fn task_mesh_requires_room_for_every_packet_mesh_output() {
        let mut limits = mesh_limits();
        limits.max_mesh_workgroup_total_count = TASK_PACKET_MESHLET_COUNT - 1;
        let capabilities = MeshletCapabilities::from_parts(
            wgpu::Backend::Vulkan,
            indexed_features() | wgpu::Features::EXPERIMENTAL_MESH_SHADER,
            limits,
        );

        assert!(
            capabilities
                .validate_backend(MeshletBackend::MeshOnly)
                .is_ok()
        );
        let MeshletCapabilityError::InsufficientLimits { failures, .. } = capabilities
            .validate_backend(MeshletBackend::TaskMesh)
            .unwrap_err()
        else {
            panic!("expected task mesh limit failure")
        };
        assert!(failures.iter().any(|failure| {
            failure.name == "max_mesh_workgroup_total_count"
                && failure.required == TASK_PACKET_MESHLET_COUNT
        }));

        let mut limits = mesh_limits();
        limits.max_mesh_workgroups_per_dimension = TASK_PACKET_MESHLET_COUNT - 1;
        let capabilities = MeshletCapabilities::from_parts(
            wgpu::Backend::Vulkan,
            indexed_features() | wgpu::Features::EXPERIMENTAL_MESH_SHADER,
            limits,
        );
        assert!(
            capabilities
                .validate_backend(MeshletBackend::MeshOnly)
                .is_ok()
        );
        let MeshletCapabilityError::InsufficientLimits { failures, .. } = capabilities
            .validate_backend(MeshletBackend::TaskMesh)
            .unwrap_err()
        else {
            panic!("expected task mesh per-dimension limit failure")
        };
        assert!(failures.iter().any(|failure| {
            failure.name == "max_mesh_workgroups_per_dimension"
                && failure.required == TASK_PACKET_MESHLET_COUNT
        }));
    }

    #[test]
    fn mesh_device_request_preserves_the_adapters_full_dispatch_limits() {
        let mut limits = mesh_limits();
        limits.max_mesh_workgroup_total_count = 97;
        limits.max_mesh_workgroups_per_dimension = 89;
        limits.max_task_workgroup_total_count = 83;
        limits.max_task_workgroups_per_dimension = 79;
        let capabilities = MeshletCapabilities::from_parts(
            wgpu::Backend::Vulkan,
            indexed_features() | wgpu::Features::EXPERIMENTAL_MESH_SHADER,
            limits.clone(),
        );
        let requirements = capabilities
            .device_requirements(&MeshletRendererConfig {
                backend: MeshletBackend::TaskMesh,
                ..Default::default()
            })
            .unwrap();

        assert_eq!(
            requirements.limits().max_mesh_workgroup_total_count,
            limits.max_mesh_workgroup_total_count
        );
        assert_eq!(
            requirements.limits().max_mesh_workgroups_per_dimension,
            limits.max_mesh_workgroups_per_dimension
        );
        assert_eq!(
            requirements.limits().max_task_workgroup_total_count,
            limits.max_task_workgroup_total_count
        );
        assert_eq!(
            requirements.limits().max_task_workgroups_per_dimension,
            limits.max_task_workgroups_per_dimension
        );
    }

    #[test]
    fn exact_driver_blacklist_is_empty_by_default_and_auditable_when_supplied() {
        let mut capabilities =
            capabilities(indexed_features() | wgpu::Features::EXPERIMENTAL_MESH_SHADER);
        capabilities.driver_key = MeshletDriverKey {
            vendor: 0x10de,
            device: 0x1234,
            driver: "example-vulkan".into(),
            driver_info: "1.2.3".into(),
        };
        let auto = MeshletRendererConfig {
            auto_benchmark_profile: Some(qualifying_profile()),
            ..Default::default()
        };
        assert_eq!(
            capabilities.resolve_backend(&auto).unwrap(),
            MeshletBackend::TaskMesh
        );

        let blacklist = MeshletDriverBlacklist {
            entries: vec![MeshletDriverBlacklistEntry {
                key: capabilities.driver_key().clone(),
                backend: MeshletBackend::TaskMesh,
                reason: "known test-only regression".into(),
            }],
        };
        assert_eq!(
            capabilities
                .resolve_backend_with_blacklist(&auto, &blacklist)
                .unwrap(),
            MeshletBackend::IndexedIndirect
        );

        let explicit = MeshletRendererConfig {
            backend: MeshletBackend::TaskMesh,
            ..Default::default()
        };
        assert!(matches!(
            capabilities.resolve_backend_with_blacklist(&explicit, &blacklist),
            Err(MeshletCapabilityError::BlacklistedDriver {
                backend: MeshletBackend::TaskMesh,
                ..
            })
        ));
    }

    #[test]
    fn driver_stable_id_is_unambiguous_when_driver_strings_contain_separators() {
        let first = MeshletDriverKey {
            vendor: 1,
            device: 2,
            driver: "alpha:beta".into(),
            driver_info: "gamma".into(),
        };
        let second = MeshletDriverKey {
            vendor: 1,
            device: 2,
            driver: "alpha".into(),
            driver_info: "beta:gamma".into(),
        };
        assert_ne!(first, second);
        assert_ne!(first.stable_id(), second.stable_id());
    }
}

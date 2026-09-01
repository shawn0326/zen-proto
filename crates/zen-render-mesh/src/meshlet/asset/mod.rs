//! Offline/runtime-cache asset representation for the Vulkan meshlet renderer.
//!
//! `.zenmesh` is deliberately independent from wgpu and from the GPU upload layout. The file is
//! little-endian, sectioned, checksummed, deterministic for identical inputs, and contains both
//! meshlet-local topology and a global `u32` fallback index stream.

mod builder;
mod format;

use std::fmt;

pub use builder::RawStaticMesh;

/// Current on-disk `.zenmesh` format version.
pub const ZENMESH_VERSION: u32 = 1;

/// Revision of the deterministic fallback builder.
///
/// Increment this whenever an implementation change can alter generated LODs or meshlets. It is
/// included in [`MeshletBuildConfig::build_hash`], so stale caches are rejected automatically.
pub const MESHLET_BUILDER_REVISION: u32 = 2;

/// Stable 128-bit content identity used by the cache protocol.
///
/// This is a fast deterministic identity, not a cryptographic digest. Callers that accept assets
/// from an untrusted source must still rely on the structural validation and per-section checksums
/// performed by [`MeshletSceneAsset::decode_zenmesh`].
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
pub struct MeshletAssetHash([u8; 16]);

impl MeshletAssetHash {
    pub const ZERO: Self = Self([0; 16]);

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }

    pub fn digest(bytes: &[u8]) -> Self {
        let mut hasher = StableHasher::new();
        hasher.write(bytes);
        hasher.finish()
    }
}

impl fmt::Debug for MeshletAssetHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "MeshletAssetHash({self})")
    }
}

impl fmt::Display for MeshletAssetHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Key used to decide whether an existing `.zenmesh` cache entry is reusable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletCacheKey {
    pub source_hash: MeshletAssetHash,
    pub build_hash: MeshletAssetHash,
}

/// Parameters that affect deterministic meshlet asset construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshletBuildConfig {
    /// Maximum distinct vertex references in one meshlet. Limited to 256 by `u8` micro-indices.
    pub max_meshlet_vertices: u32,
    /// Maximum triangles in one meshlet.
    pub max_meshlet_triangles: u32,
    /// Mesh children emitted by one task workgroup. Must match the task/mesh shader contract.
    pub task_workgroup_meshlets: u32,
    /// Meshoptimizer clustering tradeoff between locality and normal-cone quality.
    pub meshlet_cone_weight: f32,
    /// Maximum number of LODs including LOD0. Set to one to disable simplification.
    pub max_lods: u32,
    /// Target triangle ratio relative to the preceding LOD.
    pub lod_target_ratio: f32,
    /// Stop after producing an LOD at or below this triangle count.
    pub min_lod_triangles: u32,
}

impl Default for MeshletBuildConfig {
    fn default() -> Self {
        Self {
            max_meshlet_vertices: 64,
            max_meshlet_triangles: 64,
            task_workgroup_meshlets: 32,
            meshlet_cone_weight: 0.5,
            max_lods: 8,
            lod_target_ratio: 0.5,
            min_lod_triangles: 128,
        }
    }
}

impl MeshletBuildConfig {
    pub fn validate(&self) -> Result<(), MeshletAssetError> {
        if !(3..=256).contains(&self.max_meshlet_vertices) {
            return Err(MeshletAssetError::InvalidConfig(
                "max_meshlet_vertices must be in 3..=256".into(),
            ));
        }
        if !(1..=256).contains(&self.max_meshlet_triangles) {
            return Err(MeshletAssetError::InvalidConfig(
                "max_meshlet_triangles must be in 1..=256".into(),
            ));
        }
        if !(1..=32).contains(&self.task_workgroup_meshlets) {
            return Err(MeshletAssetError::InvalidConfig(
                "task_workgroup_meshlets must be in 1..=32".into(),
            ));
        }
        if !self.meshlet_cone_weight.is_finite() || !(0.0..=1.0).contains(&self.meshlet_cone_weight)
        {
            return Err(MeshletAssetError::InvalidConfig(
                "meshlet_cone_weight must be finite and in 0..=1".into(),
            ));
        }
        if !(1..=32).contains(&self.max_lods) {
            return Err(MeshletAssetError::InvalidConfig(
                "max_lods must be in 1..=32".into(),
            ));
        }
        if !self.lod_target_ratio.is_finite()
            || self.lod_target_ratio <= 0.0
            || self.lod_target_ratio >= 1.0
        {
            return Err(MeshletAssetError::InvalidConfig(
                "lod_target_ratio must be finite and strictly between zero and one".into(),
            ));
        }
        if self.min_lod_triangles == 0 {
            return Err(MeshletAssetError::InvalidConfig(
                "min_lod_triangles must be non-zero".into(),
            ));
        }
        Ok(())
    }

    pub fn build_hash(&self) -> MeshletAssetHash {
        let mut hasher = StableHasher::new();
        hasher.write(b"zen-render-mesh.meshlet-builder");
        hasher.write_u32(MESHLET_BUILDER_REVISION);
        hasher.write_u32(self.max_meshlet_vertices);
        hasher.write_u32(self.max_meshlet_triangles);
        hasher.write_u32(self.task_workgroup_meshlets);
        hasher.write_u32(self.meshlet_cone_weight.to_bits());
        hasher.write_u32(self.max_lods);
        hasher.write_u32(self.lod_target_ratio.to_bits());
        hasher.write_u32(self.min_lod_triangles);
        hasher.finish()
    }
}

/// Opaque rasterization class. Meshlets never cross this boundary.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MeshletPsoClass {
    #[default]
    OpaqueBackface = 0,
    OpaqueTwoSided = 1,
}

impl TryFrom<u32> for MeshletPsoClass {
    type Error = MeshletAssetError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::OpaqueBackface),
            1 => Ok(Self::OpaqueTwoSided),
            _ => Err(MeshletAssetError::InvalidAsset(format!(
                "unknown PSO class {value}"
            ))),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoundsSphere {
    pub center: [f32; 3],
    pub radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalCone {
    pub axis: [f32; 3],
    /// Meshoptimizer's conservative cone cutoff used by
    /// `dot(to_center, axis) >= cutoff * length(to_center) + radius`. Values above `1.0` disable
    /// cone culling.
    pub cutoff: f32,
}

impl Default for NormalCone {
    fn default() -> Self {
        Self {
            axis: [0.0, 0.0, 1.0],
            cutoff: 2.0,
        }
    }
}

/// Attribute stream paired one-to-one with [`MeshletSceneAsset::positions`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PackedVertexAttributes {
    /// Octahedral `snorm16x2` normal.
    pub normal_oct: u32,
    /// Full precision UVs preserve clamp/repeat coordinates outside `[0, 1]`.
    pub uv: [f32; 2],
    /// Linear RGBA packed as UNORM8.
    pub color_rgba8: u32,
}

impl PackedVertexAttributes {
    pub fn from_components(normal: [f32; 3], uv: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            normal_oct: pack_normal_oct(normal),
            uv,
            color_rgba8: pack_rgba8(color),
        }
    }

    pub fn unpack_normal(self) -> [f32; 3] {
        unpack_normal_oct(self.normal_oct)
    }
}

/// Entry in the asset's mesh table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeshTableEntry {
    pub first_lod: u32,
    pub lod_count: u32,
    /// Range containing all vertex records owned by every LOD of this mesh.
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub material_slot: u32,
    pub pso_class: u32,
    pub bounds: BoundsSphere,
}

impl MeshTableEntry {
    pub fn pso_class(&self) -> Result<MeshletPsoClass, MeshletAssetError> {
        self.pso_class.try_into()
    }
}

/// Entry in the LOD table. Its fallback index range is directly drawable as `u32` indices.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LodTableEntry {
    pub first_meshlet: u32,
    pub meshlet_count: u32,
    pub first_index: u32,
    pub index_count: u32,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub geometric_error: f32,
    pub bounds: BoundsSphere,
}

/// Entry in the meshlet table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeshletTableEntry {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    /// Byte offset into the micro-index stream.
    pub triangle_offset: u32,
    pub triangle_count: u32,
    pub fallback_first_index: u32,
    pub fallback_index_count: u32,
    pub bounds: BoundsSphere,
    pub normal_cone: NormalCone,
}

/// CPU asset shared by all three Vulkan meshlet rendering paths.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshletSceneAsset {
    source_hash: MeshletAssetHash,
    build_hash: MeshletAssetHash,
    config: MeshletBuildConfig,
    meshes: Vec<MeshTableEntry>,
    lods: Vec<LodTableEntry>,
    meshlets: Vec<MeshletTableEntry>,
    positions: Vec<[f32; 3]>,
    attributes: Vec<PackedVertexAttributes>,
    meshlet_vertex_refs: Vec<u32>,
    micro_indices: Vec<u8>,
    fallback_indices: Vec<u32>,
}

impl MeshletSceneAsset {
    /// Build a deterministic asset from static indexed meshes.
    pub fn build(
        meshes: &[RawStaticMesh],
        config: MeshletBuildConfig,
    ) -> Result<Self, MeshletAssetError> {
        builder::build_scene(meshes, config)
    }

    /// Hash only source data, independent from build parameters.
    pub fn source_hash(meshes: &[RawStaticMesh]) -> MeshletAssetHash {
        builder::source_hash(meshes)
    }

    pub fn cache_key(&self) -> MeshletCacheKey {
        MeshletCacheKey {
            source_hash: self.source_hash,
            build_hash: self.build_hash,
        }
    }

    pub fn config(&self) -> MeshletBuildConfig {
        self.config
    }

    pub fn meshes(&self) -> &[MeshTableEntry] {
        &self.meshes
    }

    pub fn lods(&self) -> &[LodTableEntry] {
        &self.lods
    }

    pub fn meshlets(&self) -> &[MeshletTableEntry] {
        &self.meshlets
    }

    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    pub fn attributes(&self) -> &[PackedVertexAttributes] {
        &self.attributes
    }

    pub fn meshlet_vertex_refs(&self) -> &[u32] {
        &self.meshlet_vertex_refs
    }

    pub fn micro_indices(&self) -> &[u8] {
        &self.micro_indices
    }

    pub fn fallback_indices(&self) -> &[u32] {
        &self.fallback_indices
    }

    /// Deterministically encode this asset as `.zenmesh` v1.
    pub fn encode_zenmesh(&self) -> Result<Vec<u8>, MeshletAssetError> {
        format::encode(self)
    }

    /// Decode, checksum, and fully validate a `.zenmesh` v1 byte stream.
    pub fn decode_zenmesh(bytes: &[u8]) -> Result<Self, MeshletAssetError> {
        format::decode(bytes)
    }

    /// Decode a cache entry and reject it unless both source and build identities match.
    pub fn decode_cached(
        bytes: &[u8],
        expected: MeshletCacheKey,
    ) -> Result<Self, MeshletAssetError> {
        let asset = Self::decode_zenmesh(bytes)?;
        let actual = asset.cache_key();
        if actual != expected {
            return Err(MeshletAssetError::CacheKeyMismatch { expected, actual });
        }
        Ok(asset)
    }

    /// Verify table ranges, topology reconstruction, bounds, and all format invariants.
    pub fn validate(&self) -> Result<(), MeshletAssetError> {
        self.config.validate()?;
        if self.build_hash != self.config.build_hash() {
            return Err(MeshletAssetError::InvalidAsset(
                "build hash does not match the serialized build configuration".into(),
            ));
        }
        if self.positions.len() != self.attributes.len() {
            return Err(MeshletAssetError::InvalidAsset(format!(
                "position/attribute stream length mismatch: {} != {}",
                self.positions.len(),
                self.attributes.len()
            )));
        }
        for (index, position) in self.positions.iter().enumerate() {
            if !position.iter().all(|component| component.is_finite()) {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "position {index} is not finite"
                )));
            }
        }
        for (index, attributes) in self.attributes.iter().enumerate() {
            if !attributes.uv.iter().all(|component| component.is_finite()) {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "UV {index} is not finite"
                )));
            }
        }

        let mut expected_lod = 0usize;
        let mut expected_mesh_vertex = 0usize;
        for (mesh_index, mesh) in self.meshes.iter().enumerate() {
            if mesh.first_lod as usize != expected_lod || mesh.lod_count == 0 {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "mesh {mesh_index} has a non-contiguous or empty LOD range"
                )));
            }
            if mesh.vertex_count == 0 {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "mesh {mesh_index} has an empty vertex range"
                )));
            }
            let lod_end = checked_range_end(mesh.first_lod, mesh.lod_count, self.lods.len())
                .map_err(MeshletAssetError::InvalidAsset)?;
            expected_lod = lod_end;
            if mesh.first_vertex as usize != expected_mesh_vertex {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "mesh {mesh_index} has a non-contiguous vertex range"
                )));
            }
            expected_mesh_vertex =
                checked_range_end(mesh.first_vertex, mesh.vertex_count, self.positions.len())
                    .map_err(MeshletAssetError::InvalidAsset)?;
            mesh.pso_class()?;
            validate_sphere(mesh.bounds, &format!("mesh {mesh_index}"))?;

            let mesh_vertex_start = mesh.first_vertex as usize;
            let mesh_vertex_end = expected_mesh_vertex;
            let mut expected_lod_vertex = mesh_vertex_start;
            let mut previous_triangles = usize::MAX;
            let mut previous_error = -1.0f32;
            for lod_index in mesh.first_lod as usize..lod_end {
                let lod = self.lods[lod_index];
                if lod.first_vertex as usize != expected_lod_vertex {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "LOD {lod_index} has a non-contiguous vertex range"
                    )));
                }
                expected_lod_vertex =
                    checked_range_end(lod.first_vertex, lod.vertex_count, mesh_vertex_end)
                        .map_err(MeshletAssetError::InvalidAsset)?;
                if lod.vertex_count == 0 || lod.meshlet_count == 0 || lod.index_count == 0 {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "LOD {lod_index} has an empty geometry range"
                    )));
                }
                if !lod.index_count.is_multiple_of(3) {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "LOD {lod_index} index count is not divisible by three"
                    )));
                }
                let triangles = lod.index_count as usize / 3;
                if triangles >= previous_triangles {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "LOD {lod_index} does not reduce triangle count"
                    )));
                }
                previous_triangles = triangles;
                if !lod.geometric_error.is_finite()
                    || lod.geometric_error < 0.0
                    || lod.geometric_error < previous_error
                    || (lod_index == mesh.first_lod as usize && lod.geometric_error != 0.0)
                {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "LOD {lod_index} has invalid/non-monotonic geometric error"
                    )));
                }
                previous_error = lod.geometric_error;
                validate_sphere(lod.bounds, &format!("LOD {lod_index}"))?;
                for &position in &self.positions[lod.first_vertex as usize..expected_lod_vertex] {
                    validate_sphere_contains(lod.bounds, position, &format!("LOD {lod_index}"))?;
                    // Instance classification uses the mesh-level sphere before selecting a LOD,
                    // so it must conservatively contain every serialized LOD, not only LOD0.
                    validate_sphere_contains(mesh.bounds, position, &format!("mesh {mesh_index}"))?;
                }
            }
            if expected_lod_vertex != mesh_vertex_end {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "mesh {mesh_index} vertex range contains unowned records"
                )));
            }
        }
        if expected_lod != self.lods.len() {
            return Err(MeshletAssetError::InvalidAsset(
                "LOD table contains entries not owned by a mesh".into(),
            ));
        }
        if expected_mesh_vertex != self.positions.len() {
            return Err(MeshletAssetError::InvalidAsset(
                "vertex streams contain records not owned by a mesh".into(),
            ));
        }

        let mut expected_meshlet = 0usize;
        let mut expected_fallback = 0usize;
        for (lod_index, lod) in self.lods.iter().enumerate() {
            if lod.first_meshlet as usize != expected_meshlet {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "LOD {lod_index} has a non-contiguous meshlet range"
                )));
            }
            let meshlet_end =
                checked_range_end(lod.first_meshlet, lod.meshlet_count, self.meshlets.len())
                    .map_err(MeshletAssetError::InvalidAsset)?;
            expected_meshlet = meshlet_end;
            if lod.first_index as usize != expected_fallback {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "LOD {lod_index} has a non-contiguous fallback index range"
                )));
            }
            expected_fallback = checked_range_end(
                lod.first_index,
                lod.index_count,
                self.fallback_indices.len(),
            )
            .map_err(MeshletAssetError::InvalidAsset)?;

            let lod_vertex_start = lod.first_vertex;
            let lod_vertex_end =
                lod.first_vertex
                    .checked_add(lod.vertex_count)
                    .ok_or_else(|| {
                        MeshletAssetError::InvalidAsset("LOD vertex range overflow".into())
                    })?;
            let mut meshlet_fallback = lod.first_index as usize;
            for meshlet_index in lod.first_meshlet as usize..meshlet_end {
                let meshlet = self.meshlets[meshlet_index];
                if meshlet.vertex_count == 0
                    || meshlet.vertex_count > self.config.max_meshlet_vertices
                {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "meshlet {meshlet_index} has invalid vertex count"
                    )));
                }
                if meshlet.triangle_count == 0
                    || meshlet.triangle_count > self.config.max_meshlet_triangles
                {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "meshlet {meshlet_index} has invalid triangle count"
                    )));
                }
                let vertex_end = checked_range_end(
                    meshlet.vertex_offset,
                    meshlet.vertex_count,
                    self.meshlet_vertex_refs.len(),
                )
                .map_err(MeshletAssetError::InvalidAsset)?;
                let micro_count = meshlet.triangle_count.checked_mul(3).ok_or_else(|| {
                    MeshletAssetError::InvalidAsset("meshlet micro-index overflow".into())
                })?;
                let micro_end = checked_range_end(
                    meshlet.triangle_offset,
                    micro_count,
                    self.micro_indices.len(),
                )
                .map_err(MeshletAssetError::InvalidAsset)?;
                if meshlet.fallback_first_index as usize != meshlet_fallback
                    || meshlet.fallback_index_count != micro_count
                {
                    return Err(MeshletAssetError::InvalidAsset(format!(
                        "meshlet {meshlet_index} has an invalid fallback range"
                    )));
                }
                let fallback_end = checked_range_end(
                    meshlet.fallback_first_index,
                    meshlet.fallback_index_count,
                    self.fallback_indices.len(),
                )
                .map_err(MeshletAssetError::InvalidAsset)?;
                meshlet_fallback = fallback_end;

                let refs = &self.meshlet_vertex_refs[meshlet.vertex_offset as usize..vertex_end];
                for &vertex in refs {
                    if vertex < lod_vertex_start || vertex >= lod_vertex_end {
                        return Err(MeshletAssetError::InvalidAsset(format!(
                            "meshlet {meshlet_index} references a vertex outside its LOD"
                        )));
                    }
                }
                let micro = &self.micro_indices[meshlet.triangle_offset as usize..micro_end];
                let fallback =
                    &self.fallback_indices[meshlet.fallback_first_index as usize..fallback_end];
                for (corner, (&local, &global)) in micro.iter().zip(fallback).enumerate() {
                    let local = local as usize;
                    if local >= refs.len() || refs[local] != global {
                        return Err(MeshletAssetError::InvalidAsset(format!(
                            "meshlet {meshlet_index} fallback reconstruction differs at corner {corner}"
                        )));
                    }
                }
                validate_sphere(meshlet.bounds, &format!("meshlet {meshlet_index}"))?;
                validate_cone(meshlet.normal_cone, meshlet_index)?;
                for &vertex in refs {
                    validate_sphere_contains(
                        meshlet.bounds,
                        self.positions[vertex as usize],
                        &format!("meshlet {meshlet_index}"),
                    )?;
                }
                validate_cone_contains(
                    meshlet.normal_cone,
                    fallback,
                    &self.positions,
                    meshlet_index,
                )?;
            }
            if meshlet_fallback != expected_fallback {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "LOD {lod_index} fallback range does not match its meshlets"
                )));
            }
        }
        if expected_meshlet != self.meshlets.len() {
            return Err(MeshletAssetError::InvalidAsset(
                "meshlet table contains entries not owned by a LOD".into(),
            ));
        }
        if expected_fallback != self.fallback_indices.len() {
            return Err(MeshletAssetError::InvalidAsset(
                "fallback index stream contains unowned indices".into(),
            ));
        }

        // Meshlet vertex and micro-index streams must also be packed without gaps.
        let mut next_vertex_ref = 0usize;
        let mut next_micro = 0usize;
        for (index, meshlet) in self.meshlets.iter().enumerate() {
            if meshlet.vertex_offset as usize != next_vertex_ref
                || meshlet.triangle_offset as usize != next_micro
            {
                return Err(MeshletAssetError::InvalidAsset(format!(
                    "meshlet {index} has non-contiguous topology ranges"
                )));
            }
            next_vertex_ref += meshlet.vertex_count as usize;
            next_micro += meshlet.triangle_count as usize * 3;
        }
        if next_vertex_ref != self.meshlet_vertex_refs.len()
            || next_micro != self.micro_indices.len()
        {
            return Err(MeshletAssetError::InvalidAsset(
                "meshlet topology streams contain unowned records".into(),
            ));
        }
        Ok(())
    }
}

/// Build, cache, or format error. All failures are recoverable and contain enough context for a
/// CLI to report the bad mesh/section without panicking.
#[derive(Debug)]
pub enum MeshletAssetError {
    InvalidConfig(String),
    InvalidInput {
        mesh_index: usize,
        message: String,
    },
    InvalidAsset(String),
    InvalidFormat(String),
    UnsupportedVersion(u32),
    CacheKeyMismatch {
        expected: MeshletCacheKey,
        actual: MeshletCacheKey,
    },
    SizeOverflow(&'static str),
}

impl fmt::Display for MeshletAssetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid meshlet build config: {message}")
            }
            Self::InvalidInput {
                mesh_index,
                message,
            } => {
                write!(formatter, "invalid source mesh {mesh_index}: {message}")
            }
            Self::InvalidAsset(message) => write!(formatter, "invalid meshlet asset: {message}"),
            Self::InvalidFormat(message) => write!(formatter, "invalid .zenmesh file: {message}"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported .zenmesh version {version}")
            }
            Self::CacheKeyMismatch { expected, actual } => write!(
                formatter,
                "stale .zenmesh cache: expected source={} build={}, found source={} build={}",
                expected.source_hash, expected.build_hash, actual.source_hash, actual.build_hash
            ),
            Self::SizeOverflow(label) => {
                write!(formatter, "{label} exceeds the .zenmesh v1 limits")
            }
        }
    }
}

impl std::error::Error for MeshletAssetError {}

pub(super) struct StableHasher {
    first: u64,
    second: u64,
}

impl StableHasher {
    pub(super) const fn new() -> Self {
        Self {
            first: 0xcbf2_9ce4_8422_2325,
            second: 0x8422_2325_cbf2_9ce4,
        }
    }

    pub(super) fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.first ^= byte as u64;
            self.first = self.first.wrapping_mul(0x0000_0100_0000_01b3);
            self.second ^= (byte ^ 0xa5) as u64;
            self.second = self.second.wrapping_mul(0x0000_0100_0000_01b3);
            self.second ^= self.second >> 32;
        }
    }

    pub(super) fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    pub(super) fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    pub(super) fn finish(self) -> MeshletAssetHash {
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&self.first.to_le_bytes());
        bytes[8..].copy_from_slice(&self.second.to_le_bytes());
        MeshletAssetHash(bytes)
    }
}

fn checked_range_end(offset: u32, count: u32, limit: usize) -> Result<usize, String> {
    let offset = offset as usize;
    let count = count as usize;
    let end = offset
        .checked_add(count)
        .ok_or_else(|| "table range overflow".to_owned())?;
    if end > limit {
        return Err(format!(
            "table range {offset}..{end} exceeds stream length {limit}"
        ));
    }
    Ok(end)
}

fn validate_sphere(sphere: BoundsSphere, owner: &str) -> Result<(), MeshletAssetError> {
    if !sphere.center.iter().all(|value| value.is_finite())
        || !sphere.radius.is_finite()
        || sphere.radius < 0.0
    {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "{owner} has an invalid bounds sphere"
        )));
    }
    Ok(())
}

fn validate_cone(cone: NormalCone, index: usize) -> Result<(), MeshletAssetError> {
    if !cone.axis.iter().all(|value| value.is_finite()) || !cone.cutoff.is_finite() {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "meshlet {index} has an invalid normal cone"
        )));
    }
    if cone.cutoff <= 1.0 {
        if cone.cutoff < 0.0 {
            return Err(MeshletAssetError::InvalidAsset(format!(
                "meshlet {index} has a negative normal-cone cutoff"
            )));
        }
        let length_squared = cone.axis.iter().map(|value| value * value).sum::<f32>();
        if !(0.999..=1.001).contains(&length_squared) {
            return Err(MeshletAssetError::InvalidAsset(format!(
                "meshlet {index} normal-cone axis is not normalized"
            )));
        }
    }
    Ok(())
}

fn validate_sphere_contains(
    sphere: BoundsSphere,
    position: [f32; 3],
    owner: &str,
) -> Result<(), MeshletAssetError> {
    let distance = glam::Vec3::from_array(sphere.center).distance(glam::Vec3::from_array(position));
    let tolerance = sphere.radius.max(1.0) * 1.0e-4;
    if distance > sphere.radius + tolerance {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "{owner} bounds sphere does not contain all referenced vertices"
        )));
    }
    Ok(())
}

fn validate_cone_contains(
    cone: NormalCone,
    indices: &[u32],
    positions: &[[f32; 3]],
    meshlet_index: usize,
) -> Result<(), MeshletAssetError> {
    if cone.cutoff > 1.0 {
        return Ok(());
    }
    // Match the GPU culling code exactly. Serialized axes are allowed a small length tolerance;
    // using the raw vector here could inflate dot products above one and accept a non-conservative
    // cutoff which the shader's normalized axis would later reject with.
    let axis = glam::Vec3::from_array(cone.axis).normalize();
    let mut minimum_axis_dot = 1.0f32;
    let mut has_triangle = false;
    for triangle in indices.as_chunks::<3>().0 {
        let a = glam::Vec3::from_array(positions[triangle[0] as usize]);
        let b = glam::Vec3::from_array(positions[triangle[1] as usize]);
        let c = glam::Vec3::from_array(positions[triangle[2] as usize]);
        let normal = (b - a).cross(c - a).normalize_or_zero();
        if normal != glam::Vec3::ZERO {
            minimum_axis_dot = minimum_axis_dot.min(axis.dot(normal));
            has_triangle = true;
        }
    }
    if !has_triangle {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "meshlet {meshlet_index} enables cone culling without a non-degenerate triangle"
        )));
    }
    if minimum_axis_dot <= 0.0 {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "meshlet {meshlet_index} enables a cone spanning more than a hemisphere"
        )));
    }
    let required_cutoff = (1.0 - minimum_axis_dot * minimum_axis_dot).max(0.0).sqrt();
    if cone.cutoff + 1.0e-4 < required_cutoff {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "meshlet {meshlet_index} normal cone is not conservative"
        )));
    }
    Ok(())
}

fn pack_rgba8(color: [f32; 4]) -> u32 {
    fn quantize(value: f32) -> u32 {
        (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u32
    }
    quantize(color[0])
        | (quantize(color[1]) << 8)
        | (quantize(color[2]) << 16)
        | (quantize(color[3]) << 24)
}

fn pack_normal_oct(normal: [f32; 3]) -> u32 {
    let mut normal = glam::Vec3::from_array(normal).normalize_or_zero();
    if normal == glam::Vec3::ZERO {
        normal = glam::Vec3::Z;
    }
    normal /= normal.x.abs() + normal.y.abs() + normal.z.abs();
    let mut oct = glam::Vec2::new(normal.x, normal.y);
    if normal.z < 0.0 {
        oct = glam::Vec2::new(
            (1.0 - oct.y.abs()) * oct.x.signum(),
            (1.0 - oct.x.abs()) * oct.y.signum(),
        );
    }
    fn quantize(value: f32) -> u32 {
        ((value.clamp(-1.0, 1.0) * 32767.0).round() as i32).clamp(i16::MIN as i32, i16::MAX as i32)
            as i16 as u16 as u32
    }
    quantize(oct.x) | (quantize(oct.y) << 16)
}

fn unpack_normal_oct(packed: u32) -> [f32; 3] {
    let decode = |bits: u16| (bits as i16 as f32 / 32767.0).clamp(-1.0, 1.0);
    let oct = glam::Vec2::new(decode(packed as u16), decode((packed >> 16) as u16));
    let mut normal = glam::Vec3::new(oct.x, oct.y, 1.0 - oct.x.abs() - oct.y.abs());
    if normal.z < 0.0 {
        let old_x = normal.x;
        normal.x = (1.0 - normal.y.abs()) * old_x.signum();
        normal.y = (1.0 - old_x.abs()) * normal.y.signum();
    }
    normal.normalize_or_zero().to_array()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> RawStaticMesh {
        RawStaticMesh::new(
            vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn defaults_match_the_v1_contract() {
        let config = MeshletBuildConfig::default();
        config.validate().unwrap();
        assert_eq!(config.max_meshlet_vertices, 64);
        assert_eq!(config.max_meshlet_triangles, 64);
        assert_eq!(config.task_workgroup_meshlets, 32);
        assert_eq!(config.lod_target_ratio, 0.5);
        assert_eq!(config.min_lod_triangles, 128);
        assert_eq!(ZENMESH_VERSION, 1);
        assert_eq!(MESHLET_BUILDER_REVISION, 2);
    }

    #[test]
    fn triangle_builds_reconstructable_fallback_topology() {
        let asset = MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        asset.validate().unwrap();
        assert_eq!(asset.meshes().len(), 1);
        assert_eq!(asset.lods().len(), 1);
        assert_eq!(asset.meshlets().len(), 1);
        assert_eq!(asset.meshlet_vertex_refs(), &[0, 1, 2]);
        assert_eq!(asset.micro_indices(), &[0, 1, 2]);
        assert_eq!(asset.fallback_indices(), &[0, 1, 2]);
        let cone = asset.meshlets()[0].normal_cone;
        assert!(cone.axis[2] > 0.999, "{cone:?}");
        // A planar single-triangle cone has zero angular spread in meshoptimizer semantics.
        assert!(cone.cutoff < 0.01, "{cone:?}");
    }

    #[test]
    fn partition_honors_vertex_and_triangle_limits() {
        let mesh = RawStaticMesh::new(
            vec![
                [-1.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [-0.5, 1.0, 0.0],
                [1.0, -1.0, 0.0],
            ],
            vec![0, 1, 2, 1, 3, 2],
        );
        let config = MeshletBuildConfig {
            max_meshlet_vertices: 3,
            max_meshlet_triangles: 1,
            max_lods: 1,
            ..Default::default()
        };
        let asset = MeshletSceneAsset::build(&[mesh], config).unwrap();
        assert_eq!(asset.meshlets().len(), 2);
        assert!(
            asset
                .meshlets()
                .iter()
                .all(|meshlet| { meshlet.vertex_count <= 3 && meshlet.triangle_count <= 1 })
        );
        assert_eq!(asset.fallback_indices(), &[0, 1, 2, 1, 3, 2]);
        asset.validate().unwrap();
    }

    #[test]
    fn zenmesh_round_trip_is_byte_deterministic() {
        let mut mesh = triangle();
        mesh.material_slot = 17;
        mesh.pso_class = MeshletPsoClass::OpaqueTwoSided;
        let asset = MeshletSceneAsset::build(&[mesh], MeshletBuildConfig::default()).unwrap();
        let first = asset.encode_zenmesh().unwrap();
        let decoded = MeshletSceneAsset::decode_zenmesh(&first).unwrap();
        let second = decoded.encode_zenmesh().unwrap();
        assert_eq!(decoded, asset);
        assert_eq!(second, first);
        assert_eq!(first.len() % 16, 0);
    }

    #[test]
    fn cache_validation_checks_both_hashes() {
        let asset = MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        let bytes = asset.encode_zenmesh().unwrap();
        MeshletSceneAsset::decode_cached(&bytes, asset.cache_key()).unwrap();

        let stale = MeshletCacheKey {
            source_hash: MeshletAssetHash::digest(b"different source"),
            ..asset.cache_key()
        };
        assert!(matches!(
            MeshletSceneAsset::decode_cached(&bytes, stale),
            Err(MeshletAssetError::CacheKeyMismatch { .. })
        ));
    }

    #[test]
    fn malformed_or_corrupt_files_are_rejected() {
        let asset = MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        let bytes = asset.encode_zenmesh().unwrap();
        assert!(MeshletSceneAsset::decode_zenmesh(&bytes[..bytes.len() - 1]).is_err());

        let mut corrupt = bytes;
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0x80;
        assert!(MeshletSceneAsset::decode_zenmesh(&corrupt).is_err());

        let mut corrupt_header = asset.encode_zenmesh().unwrap();
        corrupt_header[24] ^= 0x80; // First byte of the serialized source hash.
        assert!(MeshletSceneAsset::decode_zenmesh(&corrupt_header).is_err());
    }

    #[test]
    fn invalid_source_indices_are_rejected() {
        let mut mesh = triangle();
        mesh.indices[2] = 99;
        assert!(matches!(
            MeshletSceneAsset::build(&[mesh], MeshletBuildConfig::default()),
            Err(MeshletAssetError::InvalidInput { .. })
        ));
    }

    #[test]
    fn generated_lods_decrease_triangles_and_keep_valid_ranges() {
        let side = 18u32;
        let mut positions = Vec::new();
        for y in 0..=side {
            for x in 0..=side {
                positions.push([x as f32, y as f32, ((x ^ y) & 1) as f32 * 0.01]);
            }
        }
        let stride = side + 1;
        let mut indices = Vec::new();
        for y in 0..side {
            for x in 0..side {
                let a = y * stride + x;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        let config = MeshletBuildConfig {
            max_lods: 4,
            min_lod_triangles: 64,
            ..Default::default()
        };
        let asset =
            MeshletSceneAsset::build(&[RawStaticMesh::new(positions, indices)], config).unwrap();
        asset.validate().unwrap();
        assert!(asset.lods().len() >= 2);
        for pair in asset.lods().windows(2) {
            assert!(pair[1].index_count < pair[0].index_count);
            assert!(pair[1].geometric_error >= pair[0].geometric_error);
        }
    }

    #[test]
    fn lod_vertices_preserve_uv_normal_and_color_associations() {
        let side = 18u32;
        let mut mesh = RawStaticMesh::new(Vec::new(), Vec::new());
        for y in 0..=side {
            for x in 0..=side {
                mesh.positions.push([x as f32, y as f32, 0.0]);
                mesh.normals.push(if x <= side / 2 {
                    [0.0, 0.0, 1.0]
                } else {
                    [0.0, 1.0, 0.0]
                });
                mesh.tex_coords.push([
                    x as f32 / side as f32 + f32::from(x > side / 2),
                    y as f32 / side as f32,
                ]);
                mesh.colors.push(if y <= side / 2 {
                    [1.0, 0.0, 0.0, 1.0]
                } else {
                    [0.0, 1.0, 0.0, 1.0]
                });
            }
        }
        let stride = side + 1;
        for y in 0..side {
            for x in 0..side {
                let a = y * stride + x;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                mesh.indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        let source_pairs = mesh
            .positions
            .iter()
            .copied()
            .zip(
                mesh.normals
                    .iter()
                    .copied()
                    .zip(mesh.tex_coords.iter().copied())
                    .zip(mesh.colors.iter().copied())
                    .map(|((normal, uv), color)| {
                        PackedVertexAttributes::from_components(normal, uv, color)
                    }),
            )
            .collect::<Vec<_>>();
        let asset = MeshletSceneAsset::build(
            &[mesh],
            MeshletBuildConfig {
                max_lods: 4,
                min_lod_triangles: 32,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(asset.lods().len() >= 2);
        for lod in asset.lods() {
            let range = lod.first_vertex as usize..(lod.first_vertex + lod.vertex_count) as usize;
            for index in range {
                assert!(
                    source_pairs.iter().any(|pair| {
                        pair.0 == asset.positions[index] && pair.1 == asset.attributes[index]
                    }),
                    "LOD introduced an averaged or mismatched vertex attribute"
                );
            }
        }
    }

    #[test]
    fn lod_generation_stops_when_topology_cannot_be_safely_reduced() {
        let asset = MeshletSceneAsset::build(
            &[triangle()],
            MeshletBuildConfig {
                max_lods: 8,
                min_lod_triangles: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(asset.lods().len(), 1);
    }

    #[test]
    fn validation_detects_topology_corruption() {
        let mut asset =
            MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        asset.fallback_indices[0] = 2;
        assert!(matches!(
            asset.validate(),
            Err(MeshletAssetError::InvalidAsset(_))
        ));
    }

    #[test]
    fn validation_rejects_non_conservative_culling_bounds() {
        let mut asset =
            MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        asset.meshlets[0].bounds.radius = 0.0;
        assert!(matches!(
            asset.validate(),
            Err(MeshletAssetError::InvalidAsset(_))
        ));

        let mut asset =
            MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        asset.meshlets[0].normal_cone.axis = [1.0, 0.0, 0.0];
        assert!(matches!(
            asset.validate(),
            Err(MeshletAssetError::InvalidAsset(_))
        ));

        let mut asset =
            MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        let angle = 1.0_f32.to_radians();
        let tolerated_length = 1.001_f32.sqrt();
        asset.meshlets[0].normal_cone.axis = [
            angle.sin() * tolerated_length,
            0.0,
            angle.cos() * tolerated_length,
        ];
        asset.meshlets[0].normal_cone.cutoff = 0.0;
        assert!(matches!(
            asset.validate(),
            Err(MeshletAssetError::InvalidAsset(_))
        ));
    }

    #[test]
    fn every_truncation_of_a_cache_file_is_rejected_without_panicking() {
        let asset = MeshletSceneAsset::build(&[triangle()], MeshletBuildConfig::default()).unwrap();
        let bytes = asset.encode_zenmesh().unwrap();
        for length in 0..bytes.len() {
            assert!(
                MeshletSceneAsset::decode_zenmesh(&bytes[..length]).is_err(),
                "truncation at byte {length} was accepted"
            );
        }
    }
}

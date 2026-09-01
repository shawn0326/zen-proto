use std::collections::BTreeMap;

use glam::Vec3;

use super::{
    BoundsSphere, LodTableEntry, MeshTableEntry, MeshletAssetError, MeshletBuildConfig,
    MeshletPsoClass, MeshletSceneAsset, MeshletTableEntry, NormalCone, PackedVertexAttributes,
    StableHasher,
};

/// One static indexed primitive. A source record owns one material/PSO class, so generated
/// meshlets cannot accidentally straddle a material boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct RawStaticMesh {
    pub positions: Vec<[f32; 3]>,
    /// Optional; when empty, smooth area-weighted normals are generated.
    pub normals: Vec<[f32; 3]>,
    /// Optional; when empty, all UVs are zero.
    pub tex_coords: Vec<[f32; 2]>,
    /// Optional; when empty, all colors are opaque white.
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub material_slot: u32,
    pub pso_class: MeshletPsoClass,
}

impl RawStaticMesh {
    pub fn new(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            normals: Vec::new(),
            tex_coords: Vec::new(),
            colors: Vec::new(),
            indices,
            material_slot: 0,
            pso_class: MeshletPsoClass::OpaqueBackface,
        }
    }
}

#[derive(Clone)]
struct BuildVertex {
    position: Vec3,
    normal: Vec3,
    uv: [f32; 2],
    color: [f32; 4],
}

#[derive(Clone)]
struct LodGeometry {
    vertices: Vec<BuildVertex>,
    indices: Vec<u32>,
    geometric_error: f32,
}

pub(super) fn source_hash(meshes: &[RawStaticMesh]) -> super::MeshletAssetHash {
    let mut hasher = StableHasher::new();
    hasher.write(b"zen-render-mesh.raw-static-mesh.v1");
    hasher.write_u64(meshes.len() as u64);
    for mesh in meshes {
        hasher.write_u32(mesh.material_slot);
        hasher.write_u32(mesh.pso_class as u32);
        hash_f32x3(&mut hasher, &mesh.positions);
        hash_f32x3(&mut hasher, &mesh.normals);
        hasher.write_u64(mesh.tex_coords.len() as u64);
        for value in &mesh.tex_coords {
            hasher.write_u32(value[0].to_bits());
            hasher.write_u32(value[1].to_bits());
        }
        hasher.write_u64(mesh.colors.len() as u64);
        for value in &mesh.colors {
            for component in value {
                hasher.write_u32(component.to_bits());
            }
        }
        hasher.write_u64(mesh.indices.len() as u64);
        for &index in &mesh.indices {
            hasher.write_u32(index);
        }
    }
    hasher.finish()
}

fn hash_f32x3(hasher: &mut StableHasher, values: &[[f32; 3]]) {
    hasher.write_u64(values.len() as u64);
    for value in values {
        for component in value {
            hasher.write_u32(component.to_bits());
        }
    }
}

pub(super) fn build_scene(
    source_meshes: &[RawStaticMesh],
    config: MeshletBuildConfig,
) -> Result<MeshletSceneAsset, MeshletAssetError> {
    config.validate()?;
    let source_hash = source_hash(source_meshes);
    let build_hash = config.build_hash();
    let mut asset = MeshletSceneAsset {
        source_hash,
        build_hash,
        config,
        meshes: Vec::new(),
        lods: Vec::new(),
        meshlets: Vec::new(),
        positions: Vec::new(),
        attributes: Vec::new(),
        meshlet_vertex_refs: Vec::new(),
        micro_indices: Vec::new(),
        fallback_indices: Vec::new(),
    };

    for (mesh_index, source) in source_meshes.iter().enumerate() {
        let lod0 = validate_and_expand_source(mesh_index, source)?;
        let lods = generate_lods(lod0, config);
        append_mesh(&mut asset, source, &lods)?;
    }
    asset.validate()?;
    Ok(asset)
}

fn validate_and_expand_source(
    mesh_index: usize,
    source: &RawStaticMesh,
) -> Result<LodGeometry, MeshletAssetError> {
    let fail = |message: String| MeshletAssetError::InvalidInput {
        mesh_index,
        message,
    };
    if source.positions.is_empty() {
        return Err(fail("position stream is empty".into()));
    }
    if source.indices.is_empty() || !source.indices.len().is_multiple_of(3) {
        return Err(fail(
            "index stream must contain a non-zero multiple of three indices".into(),
        ));
    }
    if source.positions.len() > u32::MAX as usize {
        return Err(MeshletAssetError::SizeOverflow("source position stream"));
    }
    for (vertex, position) in source.positions.iter().enumerate() {
        if !position.iter().all(|value| value.is_finite()) {
            return Err(fail(format!("position {vertex} is not finite")));
        }
    }
    for (corner, &index) in source.indices.iter().enumerate() {
        if index as usize >= source.positions.len() {
            return Err(fail(format!(
                "index {corner} references vertex {index}, but only {} vertices exist",
                source.positions.len()
            )));
        }
    }
    validate_optional_len(
        mesh_index,
        "normal",
        source.normals.len(),
        source.positions.len(),
    )?;
    validate_optional_len(
        mesh_index,
        "texture coordinate",
        source.tex_coords.len(),
        source.positions.len(),
    )?;
    validate_optional_len(
        mesh_index,
        "color",
        source.colors.len(),
        source.positions.len(),
    )?;
    for (vertex, normal) in source.normals.iter().enumerate() {
        if !normal.iter().all(|value| value.is_finite()) {
            return Err(fail(format!("normal {vertex} is not finite")));
        }
    }
    for (vertex, uv) in source.tex_coords.iter().enumerate() {
        if !uv.iter().all(|value| value.is_finite()) {
            return Err(fail(format!("texture coordinate {vertex} is not finite")));
        }
    }
    for (vertex, color) in source.colors.iter().enumerate() {
        if !color.iter().all(|value| value.is_finite()) {
            return Err(fail(format!("color {vertex} is not finite")));
        }
    }

    let generated_normals = source
        .normals
        .is_empty()
        .then(|| generate_normals(&source.positions, &source.indices));
    let mut vertices = Vec::with_capacity(source.positions.len());
    for index in 0..source.positions.len() {
        let mut normal = if let Some(normals) = &generated_normals {
            normals[index]
        } else {
            Vec3::from_array(source.normals[index]).normalize_or_zero()
        };
        if normal == Vec3::ZERO {
            normal = Vec3::Z;
        }
        vertices.push(BuildVertex {
            position: Vec3::from_array(source.positions[index]),
            normal,
            uv: source.tex_coords.get(index).copied().unwrap_or([0.0; 2]),
            color: source.colors.get(index).copied().unwrap_or([1.0; 4]),
        });
    }
    Ok(LodGeometry {
        vertices,
        indices: source.indices.clone(),
        geometric_error: 0.0,
    })
}

fn validate_optional_len(
    mesh_index: usize,
    label: &str,
    actual: usize,
    expected: usize,
) -> Result<(), MeshletAssetError> {
    if actual != 0 && actual != expected {
        return Err(MeshletAssetError::InvalidInput {
            mesh_index,
            message: format!(
                "optional {label} stream has {actual} records; expected zero or {expected}"
            ),
        });
    }
    Ok(())
}

fn generate_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let a = Vec3::from_array(positions[triangle[0] as usize]);
        let b = Vec3::from_array(positions[triangle[1] as usize]);
        let c = Vec3::from_array(positions[triangle[2] as usize]);
        let weighted = (b - a).cross(c - a);
        if weighted.is_finite() {
            normals[triangle[0] as usize] += weighted;
            normals[triangle[1] as usize] += weighted;
            normals[triangle[2] as usize] += weighted;
        }
    }
    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
        if *normal == Vec3::ZERO {
            *normal = Vec3::Z;
        }
    }
    normals
}

fn generate_lods(lod0: LodGeometry, config: MeshletBuildConfig) -> Vec<LodGeometry> {
    let mut lods = vec![lod0];
    let lod0_triangles = lods[0].indices.len() / 3;
    for level in 1..config.max_lods as usize {
        let previous_triangles = lods.last().expect("LOD0 always exists").indices.len() / 3;
        if previous_triangles <= config.min_lod_triangles as usize {
            break;
        }
        let target = ((lod0_triangles as f32 * config.lod_target_ratio.powi(level as i32)).round()
            as usize)
            .max(config.min_lod_triangles as usize)
            .max(1)
            .min(lod0_triangles.saturating_sub(1));
        let Some(mut simplified) = simplify_with_meshoptimizer(&lods[0], target) else {
            break;
        };
        if simplified.indices.len() / 3 >= previous_triangles || simplified.indices.is_empty() {
            break;
        }
        simplified.geometric_error = simplified
            .geometric_error
            .max(lods.last().expect("LOD0 always exists").geometric_error);
        lods.push(simplified);
    }
    lods
}

fn simplify_with_meshoptimizer(
    source: &LodGeometry,
    target_triangles: usize,
) -> Option<LodGeometry> {
    let positions: Vec<[f32; 3]> = source
        .vertices
        .iter()
        .map(|vertex| vertex.position.to_array())
        .collect();
    let adapter = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(&positions),
        std::mem::size_of::<[f32; 3]>(),
        0,
    )
    .ok()?;
    let attributes = source
        .vertices
        .iter()
        .map(|vertex| {
            [
                vertex.normal.x,
                vertex.normal.y,
                vertex.normal.z,
                vertex.uv[0],
                vertex.uv[1],
                vertex.color[0],
                vertex.color[1],
                vertex.color[2],
                vertex.color[3],
            ]
        })
        .collect::<Vec<[f32; 9]>>();
    const ATTRIBUTE_WEIGHTS: [f32; 9] = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let vertex_locks = vec![false; source.vertices.len()];
    let scale = meshopt::simplify_scale(&adapter);
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let mut relative_error = 0.0f32;
    let simplified = meshopt::simplify_with_attributes_and_locks(
        &source.indices,
        &adapter,
        bytemuck::cast_slice(&attributes),
        &ATTRIBUTE_WEIGHTS,
        std::mem::size_of::<[f32; 9]>(),
        &vertex_locks,
        target_triangles.checked_mul(3)?,
        0.01,
        meshopt::SimplifyOptions::None,
        Some(&mut relative_error),
    );
    let geometric_error = relative_error * scale;
    if simplified.is_empty()
        || !simplified.len().is_multiple_of(3)
        || simplified.len() >= source.indices.len()
        || !geometric_error.is_finite()
        || geometric_error < 0.0
    {
        return None;
    }

    // meshopt intentionally keeps indices in the input vertex domain. Compact the referenced
    // subset so each LOD remains a self-contained, contiguous range in the asset streams.
    let mut old_to_new = BTreeMap::<u32, u32>::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::with_capacity(simplified.len());
    for old_index in simplified {
        let new_index = if let Some(&existing) = old_to_new.get(&old_index) {
            existing
        } else {
            let index = u32::try_from(vertices.len()).ok()?;
            vertices.push(source.vertices.get(old_index as usize)?.clone());
            old_to_new.insert(old_index, index);
            index
        };
        indices.push(new_index);
    }
    Some(LodGeometry {
        vertices,
        indices,
        geometric_error,
    })
}

fn append_mesh(
    asset: &mut MeshletSceneAsset,
    source: &RawStaticMesh,
    lods: &[LodGeometry],
) -> Result<(), MeshletAssetError> {
    let first_lod = as_u32(asset.lods.len(), "LOD table")?;
    let first_mesh_vertex = as_u32(asset.positions.len(), "position stream")?;
    let mesh_bounds = bounds_sphere(lods[0].vertices.iter().map(|vertex| vertex.position));

    for lod in lods {
        let first_vertex = as_u32(asset.positions.len(), "position stream")?;
        for vertex in &lod.vertices {
            asset.positions.push(vertex.position.to_array());
            asset
                .attributes
                .push(PackedVertexAttributes::from_components(
                    vertex.normal.to_array(),
                    vertex.uv,
                    vertex.color,
                ));
        }
        let first_meshlet = as_u32(asset.meshlets.len(), "meshlet table")?;
        let first_index = as_u32(asset.fallback_indices.len(), "fallback index stream")?;
        partition_meshlets(asset, lod, first_vertex)?;
        let meshlet_count = as_u32(asset.meshlets.len(), "meshlet table")? - first_meshlet;
        let index_count =
            as_u32(asset.fallback_indices.len(), "fallback index stream")? - first_index;
        asset.lods.push(LodTableEntry {
            first_meshlet,
            meshlet_count,
            first_index,
            index_count,
            first_vertex,
            vertex_count: as_u32(lod.vertices.len(), "LOD vertex range")?,
            geometric_error: lod.geometric_error,
            bounds: bounds_sphere(lod.vertices.iter().map(|vertex| vertex.position)),
        });
    }

    asset.meshes.push(MeshTableEntry {
        first_lod,
        lod_count: as_u32(lods.len(), "mesh LOD range")?,
        first_vertex: first_mesh_vertex,
        vertex_count: as_u32(asset.positions.len(), "position stream")? - first_mesh_vertex,
        material_slot: source.material_slot,
        pso_class: source.pso_class as u32,
        bounds: mesh_bounds,
    });
    Ok(())
}

fn partition_meshlets(
    asset: &mut MeshletSceneAsset,
    lod: &LodGeometry,
    first_vertex: u32,
) -> Result<(), MeshletAssetError> {
    let first_fallback = asset.fallback_indices.len();
    if partition_meshlets_with_meshoptimizer(asset, lod, first_vertex)? {
        return verify_partition_topology(
            &lod.indices,
            &asset.fallback_indices[first_fallback..],
            first_vertex,
        );
    }
    partition_meshlets_fallback(asset, lod, first_vertex)?;
    verify_partition_topology(
        &lod.indices,
        &asset.fallback_indices[first_fallback..],
        first_vertex,
    )
}

fn verify_partition_topology(
    source: &[u32],
    global_fallback: &[u32],
    first_vertex: u32,
) -> Result<(), MeshletAssetError> {
    if source.len() != global_fallback.len() {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "meshlet partition changed index count from {} to {}",
            source.len(),
            global_fallback.len()
        )));
    }
    let mut expected: Vec<[u32; 3]> = source
        .as_chunks::<3>()
        .0
        .iter()
        .copied()
        .map(canonical_oriented_triangle)
        .collect();
    let mut actual = Vec::with_capacity(expected.len());
    for triangle in global_fallback.as_chunks::<3>().0 {
        let local = [
            triangle[0].checked_sub(first_vertex),
            triangle[1].checked_sub(first_vertex),
            triangle[2].checked_sub(first_vertex),
        ];
        let [Some(a), Some(b), Some(c)] = local else {
            return Err(MeshletAssetError::InvalidAsset(
                "meshlet partition produced a vertex below its LOD base".into(),
            ));
        };
        actual.push(canonical_oriented_triangle([a, b, c]));
    }
    expected.sort_unstable();
    actual.sort_unstable();
    if actual != expected {
        return Err(MeshletAssetError::InvalidAsset(
            "meshlet partition changed triangle coverage or winding".into(),
        ));
    }
    Ok(())
}

fn canonical_oriented_triangle(triangle: [u32; 3]) -> [u32; 3] {
    let rotations = [
        triangle,
        [triangle[1], triangle[2], triangle[0]],
        [triangle[2], triangle[0], triangle[1]],
    ];
    *rotations.iter().min().expect("three rotations exist")
}

fn partition_meshlets_with_meshoptimizer(
    asset: &mut MeshletSceneAsset,
    lod: &LodGeometry,
    first_vertex: u32,
) -> Result<bool, MeshletAssetError> {
    // meshopt requires the triangle limit to be divisible by four. Custom test/build settings that
    // do not meet that constraint remain supported by the deterministic greedy fallback below.
    if !asset.config.max_meshlet_triangles.is_multiple_of(4) {
        return Ok(false);
    }
    let positions: Vec<[f32; 3]> = lod
        .vertices
        .iter()
        .map(|vertex| vertex.position.to_array())
        .collect();
    let adapter = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(&positions),
        std::mem::size_of::<[f32; 3]>(),
        0,
    )
    .map_err(|error| {
        MeshletAssetError::InvalidAsset(format!(
            "meshoptimizer rejected the position stream: {error}"
        ))
    })?;
    let meshlets = meshopt::build_meshlets(
        &lod.indices,
        &adapter,
        asset.config.max_meshlet_vertices as usize,
        asset.config.max_meshlet_triangles as usize,
        asset.config.meshlet_cone_weight,
    );
    if meshlets.is_empty() {
        return Ok(false);
    }
    for meshlet in meshlets.iter() {
        flush_meshlet(
            asset,
            lod,
            first_vertex,
            meshlet.vertices,
            meshlet.triangles,
        )?;
    }
    Ok(true)
}

fn partition_meshlets_fallback(
    asset: &mut MeshletSceneAsset,
    lod: &LodGeometry,
    first_vertex: u32,
) -> Result<(), MeshletAssetError> {
    let mut local_map = BTreeMap::<u32, u8>::new();
    let mut vertex_refs = Vec::<u32>::new();
    let mut micro_indices = Vec::<u8>::new();

    for triangle in lod.indices.as_chunks::<3>().0 {
        let new_vertices = triangle
            .iter()
            .filter(|&&vertex| !local_map.contains_key(&vertex))
            .count();
        let full = micro_indices.len() / 3 >= asset.config.max_meshlet_triangles as usize
            || vertex_refs.len() + new_vertices > asset.config.max_meshlet_vertices as usize;
        if full && !micro_indices.is_empty() {
            flush_meshlet(asset, lod, first_vertex, &vertex_refs, &micro_indices)?;
            local_map.clear();
            vertex_refs.clear();
            micro_indices.clear();
        }
        for &vertex in triangle {
            let local = if let Some(&local) = local_map.get(&vertex) {
                local
            } else {
                let local = u8::try_from(vertex_refs.len()).map_err(|_| {
                    MeshletAssetError::InvalidAsset(
                        "meshlet partition exceeded the u8 micro-index range".into(),
                    )
                })?;
                local_map.insert(vertex, local);
                vertex_refs.push(vertex);
                local
            };
            micro_indices.push(local);
        }
    }
    if !micro_indices.is_empty() {
        flush_meshlet(asset, lod, first_vertex, &vertex_refs, &micro_indices)?;
    }
    Ok(())
}

fn flush_meshlet(
    asset: &mut MeshletSceneAsset,
    lod: &LodGeometry,
    first_vertex: u32,
    local_vertex_refs: &[u32],
    micro: &[u8],
) -> Result<(), MeshletAssetError> {
    let vertex_offset = as_u32(asset.meshlet_vertex_refs.len(), "meshlet vertex-ref stream")?;
    let triangle_offset = as_u32(asset.micro_indices.len(), "meshlet micro-index stream")?;
    let fallback_first_index = as_u32(asset.fallback_indices.len(), "fallback index stream")?;
    let global_refs: Vec<u32> = local_vertex_refs
        .iter()
        .map(|&vertex| {
            first_vertex
                .checked_add(vertex)
                .ok_or(MeshletAssetError::SizeOverflow("global vertex reference"))
        })
        .collect::<Result<_, _>>()?;

    asset.meshlet_vertex_refs.extend_from_slice(&global_refs);
    asset.micro_indices.extend_from_slice(micro);
    for &local in micro {
        asset.fallback_indices.push(global_refs[local as usize]);
    }
    let (bounds, normal_cone) = cluster_bounds(lod, local_vertex_refs, micro);
    asset.meshlets.push(MeshletTableEntry {
        vertex_offset,
        vertex_count: as_u32(global_refs.len(), "meshlet vertex range")?,
        triangle_offset,
        triangle_count: as_u32(micro.len() / 3, "meshlet triangle range")?,
        fallback_first_index,
        fallback_index_count: as_u32(micro.len(), "meshlet fallback range")?,
        bounds,
        normal_cone,
    });
    Ok(())
}

fn bounds_sphere(points: impl IntoIterator<Item = Vec3>) -> BoundsSphere {
    let points: Vec<Vec3> = points.into_iter().collect();
    if points.is_empty() {
        return BoundsSphere::default();
    }

    // Ritter sphere initialized from the widest axis-extreme pair, then expanded to contain all
    // points. It is deterministic and conservative, which is more important than minimal radius.
    let mut min = [points[0]; 3];
    let mut max = [points[0]; 3];
    for &point in &points[1..] {
        if point.x < min[0].x {
            min[0] = point;
        }
        if point.x > max[0].x {
            max[0] = point;
        }
        if point.y < min[1].y {
            min[1] = point;
        }
        if point.y > max[1].y {
            max[1] = point;
        }
        if point.z < min[2].z {
            min[2] = point;
        }
        if point.z > max[2].z {
            max[2] = point;
        }
    }
    let (start, end) = (0..3)
        .map(|axis| (min[axis], max[axis]))
        .max_by(|(a0, a1), (b0, b1)| {
            a0.distance_squared(*a1)
                .total_cmp(&b0.distance_squared(*b1))
        })
        .expect("three axes exist");
    let mut center = (start + end) * 0.5;
    let mut radius = start.distance(end) * 0.5;
    for &point in &points {
        let distance = center.distance(point);
        if distance > radius {
            let new_radius = (radius + distance) * 0.5;
            if distance > 0.0 {
                center += (point - center) * ((new_radius - radius) / distance);
            }
            radius = new_radius;
        }
    }
    // One final max-distance pass absorbs rounding error from the center updates.
    for point in &points {
        radius = radius.max(center.distance(*point));
    }
    BoundsSphere {
        center: center.to_array(),
        radius,
    }
}

fn cluster_bounds(lod: &LodGeometry, refs: &[u32], micro: &[u8]) -> (BoundsSphere, NormalCone) {
    let positions: Vec<[f32; 3]> = lod
        .vertices
        .iter()
        .map(|vertex| vertex.position.to_array())
        .collect();
    let mut indices = Vec::with_capacity(micro.len());
    for &local in micro {
        indices.push(refs[local as usize]);
    }
    if let Ok(adapter) = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(&positions),
        std::mem::size_of::<[f32; 3]>(),
        0,
    ) {
        let computed = meshopt::compute_cluster_bounds(&indices, &adapter);
        let sphere = BoundsSphere {
            center: computed.center,
            radius: computed.radius,
        };
        let axis = Vec3::from_array(computed.cone_axis);
        let cone = if computed.cone_cutoff.is_finite()
            && (0.0..=1.0).contains(&computed.cone_cutoff)
            && axis.is_finite()
            && axis.length_squared() > 0.999
            && axis.length_squared() < 1.001
        {
            NormalCone {
                axis: computed.cone_axis,
                cutoff: computed.cone_cutoff,
            }
        } else {
            NormalCone::default()
        };
        if sphere.center.iter().all(|value| value.is_finite())
            && sphere.radius.is_finite()
            && sphere.radius >= 0.0
        {
            return (sphere, cone);
        }
    }

    let sphere = bounds_sphere(
        refs.iter()
            .map(|&index| lod.vertices[index as usize].position),
    );
    (sphere, normal_cone(lod, refs, micro))
}

fn normal_cone(lod: &LodGeometry, refs: &[u32], micro: &[u8]) -> NormalCone {
    let mut normals = Vec::with_capacity(micro.len() / 3);
    for triangle in micro.as_chunks::<3>().0 {
        let a = lod.vertices[refs[triangle[0] as usize] as usize].position;
        let b = lod.vertices[refs[triangle[1] as usize] as usize].position;
        let c = lod.vertices[refs[triangle[2] as usize] as usize].position;
        let normal = (b - a).cross(c - a).normalize_or_zero();
        if normal != Vec3::ZERO && normal.is_finite() {
            normals.push(normal);
        }
    }
    if normals.is_empty() {
        return NormalCone::default();
    }
    let axis = normals
        .iter()
        .copied()
        .fold(Vec3::ZERO, |sum, normal| sum + normal)
        .normalize_or_zero();
    if axis == Vec3::ZERO {
        return NormalCone::default();
    }
    let minimum_axis_dot = normals
        .iter()
        .map(|normal| axis.dot(*normal))
        .fold(1.0f32, f32::min);
    if minimum_axis_dot <= 0.0 {
        return NormalCone::default();
    }
    // meshoptimizer's cutoff is sin(cone half-angle), while minimum_axis_dot is cos(angle).
    let cutoff = (1.0 - minimum_axis_dot * minimum_axis_dot).max(0.0).sqrt();
    NormalCone {
        axis: axis.to_array(),
        cutoff,
    }
}

fn as_u32(value: usize, label: &'static str) -> Result<u32, MeshletAssetError> {
    u32::try_from(value).map_err(|_| MeshletAssetError::SizeOverflow(label))
}

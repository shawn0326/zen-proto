struct LodRecord {
    first_meshlet: u32,
    meshlet_count: u32,
    geometric_error: f32,
    _pad: u32,
    sphere: vec4<f32>,
};

struct InstanceData {
    model: mat4x4<f32>,
    mesh_id: u32,
    material_id: u32,
    _pad: vec2<u32>,
};

struct Work {
    meshlet_id: u32,
    instance_id: u32,
    material_id: u32,
    pso_bin: u32,
};

struct FrameUniform {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
    parameters: vec4<f32>,
    counts: vec4<u32>,
    hiz_mip_count: u32,
    max_dispatch_dimension: u32,
    perspective_projection: u32,
    _pad: u32,
};

struct Counters {
    candidate_count: atomic<u32>,
    visible_count_backface: atomic<u32>,
    visible_count_two_sided: atomic<u32>,
    instances_visible: atomic<u32>,
    culled_frustum: atomic<u32>,
    culled_cone: atomic<u32>,
    culled_hiz: atomic<u32>,
    output_vertices: atomic<u32>,
    output_primitives: atomic<u32>,
    overflow: atomic<u32>,
    lod_histogram: array<atomic<u32>, 8>,
    lod_overflow_instances: atomic<u32>,
    conservatively_visible_meshlets: atomic<u32>,
};

struct InstanceClassification {
    meshlet_count: u32,
    meshlet_offset: u32,
    selected_lod: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> lods: array<LodRecord>;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(2) var<storage, read> classifications: array<InstanceClassification>;
@group(0) @binding(3) var<storage, read_write> candidates: array<Work>;
@group(0) @binding(4) var<storage, read_write> counters: Counters;
@group(0) @binding(5) var<uniform> frame: FrameUniform;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let instance_id = (group_id.y * max(frame.max_dispatch_dimension, 1u) + group_id.x) * 64u + lane;
    if (instance_id >= frame.counts.x) {
        return;
    }
    let classification = classifications[instance_id];
    let count = classification.meshlet_count;
    if (count == 0u) {
        return;
    }
    let selected = classification.selected_lod;
    if (selected == 0xffffffffu) {
        return;
    }
    let instance = instances[instance_id];
    // Raster state belongs to the geometry instance, not to a texture/material bitfield. The
    // uploader stores MeshletPsoClass in InstanceData._pad.x. Clamp corrupt input to one of the
    // two supported opaque bins so it can never address outside the per-bin work arenas.
    let bin = min(instance._pad.x, 1u);
    let offset = classification.meshlet_offset;
    let lod = lods[selected];
    for (var local = 0u; local < count; local += 1u) {
        // Test the subtraction form before adding so a capacity near u32::MAX cannot make
        // offset + local wrap around and overwrite the beginning of the candidate arena.
        if (offset < frame.counts.z && local < frame.counts.z - offset) {
            let destination = offset + local;
            candidates[destination] = Work(
                lod.first_meshlet + local,
                instance_id,
                instance.material_id,
                bin,
            );
        } else {
            atomicOr(&counters.overflow, 1u);
        }
    }
}

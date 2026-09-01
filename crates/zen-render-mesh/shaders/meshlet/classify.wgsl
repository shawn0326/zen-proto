struct MeshRecord {
    first_lod: u32,
    lod_count: u32,
    _pad: vec2<u32>,
    sphere: vec4<f32>,
};

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

@group(0) @binding(0) var<storage, read> meshes: array<MeshRecord>;
@group(0) @binding(1) var<storage, read> lods: array<LodRecord>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(3) var<storage, read_write> classifications: array<InstanceClassification>;
@group(0) @binding(4) var<storage, read_write> lod_history: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: Counters;
@group(0) @binding(6) var<uniform> frame: FrameUniform;

fn conservative_model_scale(model: mat4x4<f32>) -> f32 {
    let column0 = model[0].xyz;
    let column1 = model[1].xyz;
    let column2 = model[2].xyz;
    let g00 = dot(column0, column0);
    let g11 = dot(column1, column1);
    let g22 = dot(column2, column2);
    let g01 = abs(dot(column0, column1));
    let g02 = abs(dot(column0, column2));
    let g12 = abs(dot(column1, column2));
    // Gershgorin bounds the largest eigenvalue of A^T A by its largest absolute row sum.
    // Therefore its square root conservatively bounds every affine stretch, while remaining exact
    // for orthogonal TRS columns.
    let maximum_eigenvalue_bound = max(g00 + g01 + g02, max(g11 + g01 + g12, g22 + g02 + g12));
    return sqrt(maximum_eigenvalue_bound);
}

fn finite3(value: vec3<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec3<f32>(3.4e38));
}

fn finite1(value: f32) -> bool {
    return value == value && abs(value) <= 3.4e38;
}

fn sphere_is_outside(center: vec3<f32>, radius: f32) -> bool {
    if (!finite3(center) || !finite1(radius) || radius < 0.0) {
        return false;
    }
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = frame.frustum_planes[plane_index];
        if (dot(plane.xyz, center) + plane.w < -radius) {
            return true;
        }
    }
    return false;
}

fn vertical_focal_pixels() -> f32 {
    // For an ordinary perspective matrix, the spatial part of clip-space row Y is projection[1][1]
    // times view-space row Y. Taking their length ratio recovers abs(projection[1][1]) without
    // depending on camera rotation or adding another uniform ABI field.
    let view_y = vec3<f32>(frame.view[0].y, frame.view[1].y, frame.view[2].y);
    let clip_y = vec3<f32>(
        frame.view_projection[0].y,
        frame.view_projection[1].y,
        frame.view_projection[2].y,
    );
    let focal_ndc = length(clip_y) / max(length(view_y), 1e-20);
    return 0.5 * frame.viewport.y * focal_ndc;
}

fn projected_error(lod: LodRecord, model: mat4x4<f32>, focal_pixels: f32) -> f32 {
    let numerator = lod.geometric_error * conservative_model_scale(model) * focal_pixels;
    if (frame.perspective_projection == 0u) {
        // Orthographic projected error is independent of camera distance. Unknown/custom
        // projections are deliberately treated as orthographic so LOD selection stays
        // conservative instead of applying a perspective divide that may be invalid.
        return numerator;
    }
    let world_center = model * vec4<f32>(lod.sphere.xyz, 1.0);
    let view_center = frame.view * world_center;
    let distance = max(-view_center.z, max(frame.parameters.z, 1e-4));
    return numerator / distance;
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let instance_id = (group_id.y * max(frame.max_dispatch_dimension, 1u) + group_id.x) * 64u + lane;
    if (instance_id >= frame.counts.x) {
        return;
    }

    let instance = instances[instance_id];
    if (instance.mesh_id >= frame.counts.y) {
        classifications[instance_id].meshlet_count = 0u;
        classifications[instance_id].selected_lod = 0xffffffffu;
        return;
    }
    let mesh = meshes[instance.mesh_id];
    if (mesh.lod_count == 0u) {
        classifications[instance_id].meshlet_count = 0u;
        classifications[instance_id].selected_lod = 0xffffffffu;
        return;
    }

    let world_center4 = instance.model * vec4<f32>(mesh.sphere.xyz, 1.0);
    let world_radius = mesh.sphere.w * conservative_model_scale(instance.model);
    if (sphere_is_outside(world_center4.xyz, world_radius)) {
        classifications[instance_id].meshlet_count = 0u;
        classifications[instance_id].selected_lod = 0xffffffffu;
        return;
    }

    var selected_relative = 0u;
    let threshold = max(frame.parameters.x, 0.01);
    let focal_pixels = vertical_focal_pixels();
    for (var relative = 1u; relative < mesh.lod_count; relative += 1u) {
        let lod = lods[mesh.first_lod + relative];
        if (projected_error(lod, instance.model, focal_pixels) <= threshold) {
            selected_relative = relative;
        }
    }

    let previous = min(lod_history[instance_id], mesh.lod_count - 1u);
    let hysteresis = clamp(frame.parameters.y, 0.0, 0.49);
    if (selected_relative > previous) {
        let next = min(previous + 1u, mesh.lod_count - 1u);
        if (projected_error(lods[mesh.first_lod + next], instance.model, focal_pixels) > threshold * (1.0 - hysteresis)) {
            selected_relative = previous;
        } else {
            selected_relative = next;
        }
    } else if (selected_relative < previous) {
        let next = previous - 1u;
        if (projected_error(lods[mesh.first_lod + previous], instance.model, focal_pixels) < threshold * (1.0 + hysteresis)) {
            selected_relative = previous;
        } else {
            selected_relative = max(selected_relative, next);
        }
    }

    lod_history[instance_id] = selected_relative;
    let selected = mesh.first_lod + selected_relative;
    classifications[instance_id].selected_lod = selected;
    classifications[instance_id].meshlet_count = lods[selected].meshlet_count;
    atomicAdd(&counters.instances_visible, 1u);
    if (selected_relative < 8u) {
        atomicAdd(&counters.lod_histogram[selected_relative], 1u);
    } else {
        atomicAdd(&counters.lod_overflow_instances, 1u);
    }
}

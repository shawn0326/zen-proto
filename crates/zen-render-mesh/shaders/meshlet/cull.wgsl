struct MeshletRecord {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
    fallback_first_index: u32,
    fallback_index_count: u32,
    _pad: vec2<u32>,
    sphere: vec4<f32>,
    cone: vec4<f32>,
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

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
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

struct CullResult {
    culled: bool,
    conservatively_visible: bool,
};

fn reserve_visible(bin: u32, amount: u32, capacity: u32) -> vec2<u32> {
    loop {
        var current = 0u;
        if (bin == 0u) {
            current = atomicLoad(&counters.visible_count_backface);
        } else {
            current = atomicLoad(&counters.visible_count_two_sided);
        }
        if (current >= capacity || amount == 0u) {
            return vec2<u32>(capacity, 0u);
        }
        let granted = min(amount, capacity - current);
        if (bin == 0u) {
            let exchange = atomicCompareExchangeWeak(
                &counters.visible_count_backface,
                current,
                current + granted,
            );
            if (exchange.exchanged) {
                return vec2<u32>(current, granted);
            }
        } else {
            let exchange = atomicCompareExchangeWeak(
                &counters.visible_count_two_sided,
                current,
                current + granted,
            );
            if (exchange.exchanged) {
                return vec2<u32>(current, granted);
            }
        }
    }
    return vec2<u32>(capacity, 0u);
}

@group(0) @binding(0) var<storage, read> meshlets: array<MeshletRecord>;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(2) var<storage, read> candidates: array<Work>;
@group(0) @binding(3) var<storage, read_write> visible: array<Work>;
@group(0) @binding(4) var<storage, read_write> draw_args: array<DrawIndexedIndirectArgs>;
@group(0) @binding(5) var<storage, read_write> counters: Counters;
@group(0) @binding(6) var<uniform> frame: FrameUniform;
@group(0) @binding(7) var hiz: texture_2d<f32>;
@group(0) @binding(8) var hiz_sampler: sampler;

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

fn finite4(value: vec4<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec4<f32>(3.4e38));
}

fn finite1(value: f32) -> bool {
    return value == value && abs(value) <= 3.4e38;
}

fn sphere_frustum_result(center: vec3<f32>, radius: f32) -> CullResult {
    if (!finite3(center) || !finite1(radius) || radius < 0.0) {
        return CullResult(false, true);
    }
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = frame.frustum_planes[plane_index];
        if (!finite4(plane)) {
            return CullResult(false, true);
        }
        if (dot(plane.xyz, center) + plane.w < -radius) {
            return CullResult(true, false);
        }
    }
    return CullResult(false, false);
}

fn cone_cull_result(meshlet: MeshletRecord, instance: InstanceData, center: vec3<f32>, radius: f32) -> CullResult {
    if (frame.perspective_projection == 0u) {
        // Orthographic rays have a fixed direction. Until that direction is part of the shared
        // ABI, disabling the optional cone test is conservative and cannot create false culls.
        return CullResult(false, false);
    }
    // A cutoff above one explicitly means that the asset has no usable cone. That is a normal
    // asset state, not a numerical uncertainty, so it must not inflate the conservative counter.
    if (finite4(meshlet.cone) && meshlet.cone.w > 1.0) {
        return CullResult(false, false);
    }
    if (!finite4(meshlet.cone)) {
        return CullResult(false, true);
    }
    let sx = length(instance.model[0].xyz);
    let sy = length(instance.model[1].xyz);
    let sz = length(instance.model[2].xyz);
    let largest = max(sx, max(sy, sz));
    let smallest = min(sx, min(sy, sz));
    let determinant = dot(instance.model[0].xyz, cross(instance.model[1].xyz, instance.model[2].xyz));
    // Normal cones are only exact for a non-reflected uniform transform. All other cases remain
    // visible, which prevents false culling for mirrored or non-uniform instances.
    let column0 = instance.model[0].xyz;
    let column1 = instance.model[1].xyz;
    let column2 = instance.model[2].xyz;
    // The asset cutoff carries a 2e-4 conservative margin. Keep transform tolerances two orders
    // tighter so the normal-axis error cannot consume that margin at grazing angles.
    let orthogonal = abs(dot(column0, column1)) <= sx * sy * 1e-6
        && abs(dot(column0, column2)) <= sx * sz * 1e-6
        && abs(dot(column1, column2)) <= sy * sz * 1e-6;
    if (smallest <= 1e-6 || largest - smallest > largest * 1e-6 || determinant <= 0.0 || !orthogonal) {
        return CullResult(false, true);
    }
    let transformed_axis = mat3x3<f32>(instance.model[0].xyz, instance.model[1].xyz, instance.model[2].xyz) * meshlet.cone.xyz;
    let to_center = center - frame.camera_position.xyz;
    if (!finite3(transformed_axis) || dot(transformed_axis, transformed_axis) <= 1e-12
        || !finite3(to_center) || !finite1(radius)) {
        return CullResult(false, true);
    }
    let axis = normalize(transformed_axis);
    return CullResult(dot(to_center, axis) >= meshlet.cone.w * length(to_center) + radius, false);
}

fn cube_corner_offset(index: u32, radius: f32) -> vec3<f32> {
    return vec3<f32>(
        select(-radius, radius, (index & 1u) != 0u),
        select(-radius, radius, (index & 2u) != 0u),
        select(-radius, radius, (index & 4u) != 0u),
    );
}

fn sphere_hiz_result(center: vec3<f32>, radius: f32) -> CullResult {
    if (frame.parameters.w < 0.5 || frame.hiz_mip_count == 0u) {
        return CullResult(false, false);
    }
    if (!finite3(center) || !finite1(radius) || radius < 0.0) {
        return CullResult(false, true);
    }
    var ndc_min = vec3<f32>(1e20);
    var ndc_max = vec3<f32>(-1e20);
    // A sphere is contained by this world-space cube. With positive clip W, each perspective-
    // divided component is a linear-fractional function whose extrema over the cube occur at a
    // corner. Projecting all eight corners therefore gives a conservative screen/depth bound even
    // when the camera is rotated. If the cube reaches the near plane, retain the meshlet.
    for (var index = 0u; index < 8u; index += 1u) {
        let clip = frame.view_projection * vec4<f32>(center + cube_corner_offset(index, radius), 1.0);
        if (!finite4(clip) || clip.w <= 1e-5) {
            return CullResult(false, true);
        }
        let ndc = clip.xyz / clip.w;
        if (!finite3(ndc)) {
            return CullResult(false, true);
        }
        ndc_min = min(ndc_min, ndc);
        ndc_max = max(ndc_max, ndc);
    }
    if (ndc_min.z <= 0.0 || ndc_max.z >= 1.0) {
        return CullResult(false, true);
    }
    let uv_min = clamp(vec2<f32>(ndc_min.x * 0.5 + 0.5, 0.5 - ndc_max.y * 0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_max = clamp(vec2<f32>(ndc_max.x * 0.5 + 0.5, 0.5 - ndc_min.y * 0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let pixel_extent = max((uv_max.x - uv_min.x) * frame.viewport.x, (uv_max.y - uv_min.y) * frame.viewport.y);
    var mip = min(u32(max(0.0, ceil(log2(max(pixel_extent, 1.0))))), frame.hiz_mip_count - 1u);
    var texel_min = vec2<u32>(0u);
    var texel_max = vec2<u32>(0u);
    loop {
        let dimensions = textureDimensions(hiz, i32(mip));
        let last_texel = dimensions - vec2<u32>(1u);
        texel_min = min(vec2<u32>(floor(uv_min * vec2<f32>(dimensions))), last_texel);
        texel_max = min(vec2<u32>(floor(uv_max * vec2<f32>(dimensions))), last_texel);
        if ((texel_max.x - texel_min.x <= 1u && texel_max.y - texel_min.y <= 1u)
            || mip + 1u >= frame.hiz_mip_count) {
            break;
        }
        mip += 1u;
    }
    let level = i32(mip);
    let d0 = textureLoad(hiz, vec2<i32>(i32(texel_min.x), i32(texel_min.y)), level).x;
    let d1 = textureLoad(hiz, vec2<i32>(i32(texel_max.x), i32(texel_min.y)), level).x;
    let d2 = textureLoad(hiz, vec2<i32>(i32(texel_min.x), i32(texel_max.y)), level).x;
    let d3 = textureLoad(hiz, vec2<i32>(i32(texel_max.x), i32(texel_max.y)), level).x;
    if (!finite1(d0) || !finite1(d1) || !finite1(d2) || !finite1(d3)
        || min(min(d0, d1), min(d2, d3)) < 0.0
        || max(max(d0, d1), max(d2, d3)) > 1.0) {
        return CullResult(false, true);
    }
    let farthest_occluder = max(max(d0, d1), max(d2, d3));
    return CullResult(ndc_min.z > farthest_occluder + 1e-4, false);
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let candidate_id = (group_id.y * max(frame.max_dispatch_dimension, 1u) + group_id.x) * 64u + lane;
    let candidate_count = atomicLoad(&counters.candidate_count);
    if (candidate_id >= candidate_count) {
        return;
    }
    let work = candidates[candidate_id];
    let meshlet = meshlets[work.meshlet_id];
    let instance = instances[work.instance_id];
    let world_center4 = instance.model * vec4<f32>(meshlet.sphere.xyz, 1.0);
    let world_radius = meshlet.sphere.w * conservative_model_scale(instance.model);
    var conservatively_visible = false;
    let frustum_result = sphere_frustum_result(world_center4.xyz, world_radius);
    if (frustum_result.culled) {
        atomicAdd(&counters.culled_frustum, 1u);
        return;
    }
    conservatively_visible = conservatively_visible || frustum_result.conservatively_visible;
    if (work.pso_bin == 0u) {
        let cone_result = cone_cull_result(meshlet, instance, world_center4.xyz, world_radius);
        if (cone_result.culled) {
            atomicAdd(&counters.culled_cone, 1u);
            return;
        }
        conservatively_visible = conservatively_visible || cone_result.conservatively_visible;
    }
    let hiz_result = sphere_hiz_result(world_center4.xyz, world_radius);
    if (hiz_result.culled) {
        atomicAdd(&counters.culled_hiz, 1u);
        return;
    }
    conservatively_visible = conservatively_visible || hiz_result.conservatively_visible;
    if (conservatively_visible) {
        atomicAdd(&counters.conservatively_visible_meshlets, 1u);
    }

    let capacity = frame.counts.w;
    let reservation = reserve_visible(work.pso_bin, 1u, capacity);
    let local_slot = reservation.x;
    if (reservation.y == 0u) {
        atomicOr(&counters.overflow, select(2u, 4u, work.pso_bin != 0u));
        return;
    }
    let slot = work.pso_bin * capacity + local_slot;
    visible[slot] = work;
    draw_args[slot] = DrawIndexedIndirectArgs(
        meshlet.fallback_index_count,
        1u,
        meshlet.fallback_first_index,
        0i,
        slot,
    );
    atomicAdd(&counters.output_vertices, meshlet.vertex_count);
    atomicAdd(&counters.output_primitives, meshlet.triangle_count);
}

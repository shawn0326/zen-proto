enable wgpu_binding_array;
enable wgpu_mesh_shader;

struct VertexPacked {
    px: f32,
    py: f32,
    pz: f32,
    normal_oct: u32,
    uv: vec2<f32>,
    color: u32,
    _pad: u32,
};

struct MaterialData {
    albedo_factor: vec4<f32>,
    emissive_ao: vec4<f32>,
    tex_ids: vec4<u32>,
};

struct InstanceData {
    model: mat4x4<f32>,
    mesh_id: u32,
    material_id: u32,
    _pad: vec2<u32>,
};

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

struct Work {
    meshlet_id: u32,
    instance_id: u32,
    material_id: u32,
    pso_bin: u32,
};

struct TaskPacket {
    first_meshlet: u32,
    meshlet_count: u32,
    instance_id: u32,
    material_and_bin: u32,
};

struct RasterUniform {
    view_projection: mat4x4<f32>,
    visible_base: u32,
    task_packet_base: u32,
    _reserved: u32,
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
    limits: vec4<u32>,
};

struct Counters {
    candidate_count: atomic<u32>,
    packet_count_backface: atomic<u32>,
    packet_count_two_sided: atomic<u32>,
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
    raster_claim_backface: atomic<u32>,
    raster_claim_two_sided: atomic<u32>,
    _pad: array<u32, 8>,
};

struct CullResult {
    culled: bool,
    conservatively_visible: bool,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) material_id: u32,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 64>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 64>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
};

struct TaskPayload {
    works: array<Work, 32>,
};

@group(0) @binding(0) var<storage, read> vertices: array<VertexPacked>;
@group(0) @binding(1) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(3) var<storage, read> meshlets: array<MeshletRecord>;
@group(0) @binding(4) var<storage, read> meshlet_vertices: array<u32>;
@group(0) @binding(5) var<storage, read> micro_indices: array<u32>;
@group(0) @binding(6) var<storage, read> visible: array<Work>;
@group(0) @binding(7) var<storage, read_write> counters: Counters;
@group(0) @binding(8) var<uniform> raster: RasterUniform;
@group(0) @binding(9) var<storage, read> task_packets: array<TaskPacket>;
@group(0) @binding(10) var<uniform> frame: FrameUniform;
@group(0) @binding(11) var hiz: texture_2d<f32>;
@group(0) @binding(12) var hiz_sampler: sampler;

struct BackendWorkCounts {
    mesh: vec2<u32>,
    task: vec2<u32>,
};

@group(0) @binding(13) var<storage, read> backend_work_counts: BackendWorkCounts;
@group(1) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(1) var samplers: binding_array<sampler>;

var<workgroup> mesh_output: MeshOutput;
var<task_payload> task_payload_data: TaskPayload;
var<workgroup> claimed_visible_local: u32;

fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    return select(fallback, value * inverseSqrt(length_squared), length_squared > 1e-12);
}

fn decode_normal(value: u32) -> vec3<f32> {
    let encoded = unpack2x16snorm(value);
    var normal = vec3<f32>(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (normal.z < 0.0) {
        let folded = (vec2<f32>(1.0) - abs(normal.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), normal.xy >= vec2<f32>(0.0));
        normal.x = folded.x;
        normal.y = folded.y;
    }
    return normalize_or(normal, vec3<f32>(0.0, 0.0, 1.0));
}

fn transform_normal(model: mat4x4<f32>, local: vec3<f32>) -> vec3<f32> {
    let cofactor_x = cross(model[1].xyz, model[2].xyz);
    let cofactor_y = cross(model[2].xyz, model[0].xyz);
    let cofactor_z = cross(model[0].xyz, model[1].xyz);
    let orientation = select(-1.0, 1.0, dot(model[0].xyz, cofactor_x) >= 0.0);
    return normalize_or(mat3x3<f32>(cofactor_x, cofactor_y, cofactor_z) * local * orientation, local);
}

fn emit_meshlet(work: Work, lane: u32) {
    let meshlet = meshlets[work.meshlet_id];
    let instance = instances[work.instance_id];
    if (lane == 0u) {
        mesh_output.vertex_count = meshlet.vertex_count;
        mesh_output.primitive_count = meshlet.triangle_count;
    }
    if (lane < meshlet.vertex_count) {
        let vertex = vertices[meshlet_vertices[meshlet.vertex_offset + lane]];
        mesh_output.vertices[lane].position = raster.view_projection * instance.model
            * vec4<f32>(vertex.px, vertex.py, vertex.pz, 1.0);
        mesh_output.vertices[lane].color = unpack4x8unorm(vertex.color);
        mesh_output.vertices[lane].normal = transform_normal(instance.model, decode_normal(vertex.normal_oct));
        mesh_output.vertices[lane].uv = vertex.uv;
        mesh_output.vertices[lane].material_id = work.material_id;
    }
    if (lane < meshlet.triangle_count) {
        let first = meshlet.triangle_offset + lane * 3u;
        mesh_output.primitives[lane].indices = vec3<u32>(
            micro_indices[first], micro_indices[first + 1u], micro_indices[first + 2u],
        );
    }
}

fn claim_visible_work(count: u32, lane: u32) -> Work {
    if (lane == 0u) {
        if (raster.pso_bin == 0u) {
            claimed_visible_local = atomicAdd(&counters.raster_claim_backface, 1u);
        } else {
            claimed_visible_local = atomicAdd(&counters.raster_claim_two_sided, 1u);
        }
    }
    workgroupBarrier();
    // A zero count produces a zero indirect dispatch. Rectangular padding and TaskMesh's fixed
    // 32-child fanout claim beyond the logical end and deliberately repeat the final legal work.
    let safe_local = min(claimed_visible_local, max(count, 1u) - 1u);
    return visible[raster.visible_base + safe_local];
}

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

@mesh(mesh_output)
@workgroup_size(64)
fn ms_main(
    @builtin(local_invocation_index) lane: u32,
) {
    let count = backend_work_counts.mesh[raster.pso_bin];
    // Naga 30/NVIDIA reports zero WorkgroupId/GlobalInvocationId for every mesh workgroup. An
    // atomic per-bin claim supplies the unique logical work index without depending on builtins.
    emit_meshlet(claim_visible_work(count, lane), lane);
    workgroupBarrier();
}

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
    if (frame.limits.w == 0u) {
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
    if (frame.parameters.w < 0.5 || frame.limits.y == 0u) {
        return CullResult(false, false);
    }
    if (!finite3(center) || !finite1(radius) || radius < 0.0) {
        return CullResult(false, true);
    }
    var ndc_min = vec3<f32>(1e20);
    var ndc_max = vec3<f32>(-1e20);
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
    let extent = max((uv_max.x - uv_min.x) * frame.viewport.x, (uv_max.y - uv_min.y) * frame.viewport.y);
    var mip = min(u32(max(0.0, ceil(log2(max(extent, 1.0))))), frame.limits.y - 1u);
    var texel_min = vec2<u32>(0u);
    var texel_max = vec2<u32>(0u);
    loop {
        let dimensions = textureDimensions(hiz, i32(mip));
        let last_texel = dimensions - vec2<u32>(1u);
        texel_min = min(vec2<u32>(floor(uv_min * vec2<f32>(dimensions))), last_texel);
        texel_max = min(vec2<u32>(floor(uv_max * vec2<f32>(dimensions))), last_texel);
        if ((texel_max.x - texel_min.x <= 1u && texel_max.y - texel_min.y <= 1u)
            || mip + 1u >= frame.limits.y) {
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
    return CullResult(ndc_min.z > max(max(d0, d1), max(d2, d3)) + 1e-4, false);
}

@task
@payload(task_payload_data)
@workgroup_size(32)
fn ts_main(
    @builtin(local_invocation_index) lane: u32,
) -> @builtin(mesh_task_size) vec3<u32> {
    // Child mesh workgroups claim their visible work atomically, so the payload only needs to be a
    // valid initialized value. A zero count produces a zero indirect task dispatch.
    task_payload_data.works[lane] = visible[raster.visible_base];
    workgroupBarrier();
    // Naga 30's dynamic task emission count produces no child geometry on the tested NVIDIA Vulkan
    // driver. Compute culling owns logical visibility/stats; fixed 32-child fanout and duplicate-last
    // tail work are deliberately included in TaskMesh benchmarks.
    return vec3<u32>(32u, 1u, 1u);
}

@mesh(mesh_output)
@payload(task_payload_data)
@workgroup_size(64)
fn ms_task_main(
    @builtin(local_invocation_index) lane: u32,
) {
    let count = backend_work_counts.task[raster.pso_bin];
    emit_meshlet(claim_visible_work(count, lane), lane);
    workgroupBarrier();
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let material = materials[input.material_id];
    let base = textureSample(textures[material.tex_ids.x], samplers[material.tex_ids.w], input.uv)
        * material.albedo_factor * input.color;
    let emission = textureSample(textures[material.tex_ids.y], samplers[material.tex_ids.w], input.uv).rgb
        * material.emissive_ao.rgb;
    let ao = clamp(textureSample(textures[material.tex_ids.z], samplers[material.tex_ids.w], input.uv).r
        * material.emissive_ao.w, 0.0, 1.0);
    let normal = normalize_or(input.normal, vec3<f32>(0.0, 0.0, 1.0));
    let lighting = max(dot(normal, normalize(vec3<f32>(0.5, 1.0, 0.8))), 0.0);
    return vec4<f32>(base.rgb * lighting + base.rgb * ao * 0.01 + emission, base.a);
}

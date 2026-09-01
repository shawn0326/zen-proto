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
    albedo: vec2<u32>,
    emissive: vec2<u32>,
    occlusion: vec2<u32>,
    _padding: vec2<u32>,
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

struct RasterUniform {
    view_projection: mat4x4<f32>,
    visible_base: u32,
    render_mode: u32,
    pso_bin: u32,
    _pad: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) material_id: u32,
    @location(4) @interpolate(flat) meshlet_id: u32,
    @location(5) @interpolate(flat) render_mode: u32,
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

struct BackendWorkCounts {
    mesh: vec2<u32>,
    task: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> vertices: array<VertexPacked>;
@group(0) @binding(1) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(3) var<storage, read> meshlets: array<MeshletRecord>;
@group(0) @binding(4) var<storage, read> meshlet_vertices: array<u32>;
@group(0) @binding(5) var<storage, read> micro_indices: array<u32>;
@group(0) @binding(6) var<storage, read> visible: array<Work>;
@group(0) @binding(7) var<uniform> raster: RasterUniform;
@group(0) @binding(8) var<storage, read> backend_work_counts: BackendWorkCounts;
@group(1) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(1) var samplers: binding_array<sampler>;

var<workgroup> mesh_output: MeshOutput;
var<task_payload> task_payload_data: TaskPayload;

const TASK_MESHLETS_PER_WORKGROUP: u32 = 32u;

fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    return select(fallback, value * inverseSqrt(length_squared), length_squared > 1e-12);
}

fn decode_normal(value: u32) -> vec3<f32> {
    let encoded = unpack2x16snorm(value);
    var normal = vec3<f32>(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (normal.z < 0.0) {
        let folded = (vec2<f32>(1.0) - abs(normal.yx))
            * select(vec2<f32>(-1.0), vec2<f32>(1.0), normal.xy >= vec2<f32>(0.0));
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

fn hash_u32(value: u32) -> u32 {
    var hash = value;
    hash = hash ^ (hash >> 16u);
    hash = hash * 0x7feb352du;
    hash = hash ^ (hash >> 15u);
    hash = hash * 0x846ca68bu;
    return hash ^ (hash >> 16u);
}

fn meshlet_debug_color(meshlet_id: u32) -> vec3<f32> {
    let hue = f32(hash_u32(meshlet_id) & 0xffffu) / 65536.0;
    let rgb = clamp(
        abs(fract(hue + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3<f32>(3.0))
            - vec3<f32>(1.0),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return vec3<f32>(0.2) + rgb * 0.8;
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
        mesh_output.vertices[lane].normal =
            transform_normal(instance.model, decode_normal(vertex.normal_oct));
        mesh_output.vertices[lane].uv = vertex.uv;
        mesh_output.vertices[lane].material_id = work.material_id;
        mesh_output.vertices[lane].meshlet_id = work.meshlet_id;
        mesh_output.vertices[lane].render_mode = raster.render_mode;
    }
    if (lane < meshlet.triangle_count) {
        let first = meshlet.triangle_offset + lane * 3u;
        mesh_output.primitives[lane].indices = vec3<u32>(
            micro_indices[first],
            micro_indices[first + 1u],
            micro_indices[first + 2u],
        );
    }
}

fn emit_empty_meshlet(lane: u32) {
    if (lane == 0u) {
        mesh_output.vertex_count = 0u;
        mesh_output.primitive_count = 0u;
    }
}

fn flattened_id(group_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return (group_id.z * num_workgroups.y + group_id.y) * num_workgroups.x + group_id.x;
}

@mesh(mesh_output)
@workgroup_size(64)
fn ms_main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let local = flattened_id(group_id, num_workgroups);
    let count = backend_work_counts.mesh[raster.pso_bin];
    if (local >= count) {
        emit_empty_meshlet(lane);
        return;
    }
    emit_meshlet(visible[raster.visible_base + local], lane);
}

@task
@payload(task_payload_data)
@workgroup_size(32)
fn ts_main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) -> @builtin(mesh_task_size) vec3<u32> {
    let task_local = flattened_id(group_id, num_workgroups);
    let first = task_local * TASK_MESHLETS_PER_WORKGROUP;
    let count = backend_work_counts.task[raster.pso_bin];
    if (first >= count) {
        return vec3<u32>(0u);
    }

    let child_count = min(TASK_MESHLETS_PER_WORKGROUP, count - first);
    if (lane < child_count) {
        task_payload_data.works[lane] = visible[raster.visible_base + first + lane];
    }
    workgroupBarrier();
    return vec3<u32>(child_count, 1u, 1u);
}

@mesh(mesh_output)
@payload(task_payload_data)
@workgroup_size(64)
fn ms_task_main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    emit_meshlet(task_payload_data.works[group_id.x], lane);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (input.render_mode == 1u) {
        return vec4<f32>(meshlet_debug_color(input.meshlet_id), 1.0);
    }
    let material = materials[input.material_id];
    let base = textureSample(textures[material.albedo.x], samplers[material.albedo.y], input.uv)
        * material.albedo_factor * input.color;
    let emission =
        textureSample(textures[material.emissive.x], samplers[material.emissive.y], input.uv).rgb
        * material.emissive_ao.rgb;
    let ao = clamp(
        textureSample(
            textures[material.occlusion.x],
            samplers[material.occlusion.y],
            input.uv,
        ).r * material.emissive_ao.w,
        0.0,
        1.0,
    );
    let normal = normalize_or(input.normal, vec3<f32>(0.0, 0.0, 1.0));
    let lighting = max(dot(normal, normalize(vec3<f32>(0.5, 1.0, 0.8))), 0.0);
    return vec4<f32>(base.rgb * lighting + base.rgb * ao * 0.01 + emission, base.a);
}

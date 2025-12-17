struct Instance {
    model: mat4x4<f32>,
    mesh_id: u32,
    material_id: u32,
    _pad1: u32,
    _pad2: u32,
};

struct MeshTableEntry {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    _pad: u32,
    sphere: vec4<f32>,
};

struct Frustum {
    planes: array<vec4<f32>, 6>,
};

struct Params {
    instance_count: u32,
    mesh_count: u32,
    _pad0: u32,
    _pad1: u32,
};

struct Counters {
    visible_count: atomic<u32>,
};

@group(0) @binding(0)
var<storage, read> instances: array<Instance>;

@group(0) @binding(1)
var<storage, read> mesh_table: array<MeshTableEntry>;

@group(0) @binding(2)
var<uniform> frustum: Frustum;

@group(0) @binding(3)
var<uniform> params: Params;

// 共用 Counter Buffer
@group(0) @binding(4)
var<storage, read_write> counters: Counters;

// 输出：紧凑的可见实例索引列表
@group(0) @binding(5)
var<storage, read_write> visible_instances: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.instance_count) {
        return;
    }

    let inst = instances[index];

    if (inst.mesh_id >= params.mesh_count) {
        return;
    }

    let mesh = mesh_table[inst.mesh_id];

    let local_center = vec4<f32>(mesh.sphere.xyz, 1.0);
    let world_center = (inst.model * local_center).xyz;
    let radius = mesh.sphere.w;

    var visible = true;

    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let plane = frustum.planes[i];
        let dist = dot(plane.xyz, world_center) + plane.w;
        if (dist < -radius) {
            visible = false;
            break;
        }
    }

    if (visible) {
        let dst = atomicAdd(&counters.visible_count, 1u);
        visible_instances[dst] = index;
    }
}
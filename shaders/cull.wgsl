struct Instance {
    model: mat4x4<f32>,
    mesh_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct MeshTableEntry {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    _pad: u32,
    sphere: vec4<f32>, // xyz center, w radius (mesh local)
};

// 6 个 world space 平面：normal.xyz, d
struct Frustum {
    planes: array<vec4<f32>, 6>,
};

/// 对齐 wgpu 的 DrawIndexedIndirectArgs：
/// index_count, instance_count, first_index, base_vertex, first_instance
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct Counter {
    value: atomic<u32>,
};

struct Params {
    instance_count: u32,
    mesh_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> instances: array<Instance>;

@group(0) @binding(1)
var<storage, read_write> indirect_args: array<DrawIndexedIndirectArgs>;

@group(0) @binding(2)
var<uniform> frustum: Frustum;

@group(0) @binding(3)
var<uniform> params: Params;

/// 输出：可见 draw 的数量（用于 indirect_count）
@group(0) @binding(4)
var<storage, read_write> indirect_count: Counter;

@group(0) @binding(5)
var<storage, read> mesh_table: array<MeshTableEntry>;

/// 球体与平面组的简单剔除：
/// 平面方程：dot(n, x) + d = distance
/// 如果对任一平面 distance < -radius => 在平面外面 => 剔除
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

    // local sphere center -> world space
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
        let dst = atomicAdd(&indirect_count.value, 1u);
        indirect_args[dst] = DrawIndexedIndirectArgs(
            mesh.index_count,
            1u,
            mesh.first_index,
            mesh.base_vertex,
            index // first_instance：VS 里 instance_index 会带上这个 offset
        );
    }
}
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

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct Counters {
    visible_count: atomic<u32>,
};

@group(0) @binding(0)
var<storage, read> visible_instances: array<u32>;

@group(0) @binding(1)
var<storage, read> instances: array<Instance>;

@group(0) @binding(2)
var<storage, read> mesh_table: array<MeshTableEntry>;

@group(0) @binding(3)
var<storage, read> counters: Counters;

@group(0) @binding(4)
var<storage, read_write> indirect_args: array<DrawIndexedIndirectArgs>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;

    // 早退，避免空线程越界读取
    let vc = atomicLoad(&counters.visible_count);
    if (index >= vc) {
        return;
    }
    
    let instance_index = visible_instances[index];
    let inst = instances[instance_index];
    let mesh = mesh_table[inst.mesh_id];

    indirect_args[index] = DrawIndexedIndirectArgs(
        mesh.index_count,
        1u,
        mesh.first_index,
        mesh.base_vertex,
        instance_index
    );
}
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

struct MainCullUniform {
    planes: array<vec4<f32>, 6>,
    instance_count: u32,
    mesh_count: u32,
    enable_occlusion: u32,
    _pad1: u32,
};

struct Counters {
    visible_count: atomic<u32>,
};

struct HistoryVisibility {
    visible: u32,
}

@group(0) @binding(0)
var<storage, read> instances: array<Instance>;

@group(0) @binding(1)
var<storage, read> mesh_table: array<MeshTableEntry>;

@group(0) @binding(2)
var<uniform> main_cull_uniform: MainCullUniform;

@group(0) @binding(3)
var<storage, read_write> visibility_history: array<HistoryVisibility>;

@group(0) @binding(4)
var<storage, read_write> counters_a: Counters;

@group(0) @binding(5)
var<storage, read_write> visible_instances_a: array<u32>;

@group(0) @binding(6)
var<storage, read_write> counters_b: Counters;

@group(0) @binding(7)
var<storage, read_write> visible_instances_b: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= main_cull_uniform.instance_count) {
        return;
    }

    let inst = instances[index];

    if (inst.mesh_id >= main_cull_uniform.mesh_count) {
        return;
    }

    let mesh = mesh_table[inst.mesh_id];

    let local_center = vec4<f32>(mesh.sphere.xyz, 1.0);
    let world_center = (inst.model * local_center).xyz;
    let c0 = inst.model[0].xyz;
    let c1 = inst.model[1].xyz;
    let c2 = inst.model[2].xyz;
    let max_scale = max(length(c0), max(length(c1), length(c2)));
    let radius = mesh.sphere.w * max_scale;

    var visible = true;

    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let plane = main_cull_uniform.planes[i];
        let dist = dot(plane.xyz, world_center) + plane.w;
        if (dist < -radius) {
            visible = false;
            break;
        }
    }

    if (visible) {
        if (main_cull_uniform.enable_occlusion == 1u && visibility_history[index].visible == 0u) {
            let dst = atomicAdd(&counters_b.visible_count, 1u);
            visible_instances_b[dst] = index;
        } else {
            let dst = atomicAdd(&counters_a.visible_count, 1u);
            visible_instances_a[dst] = index;
        }

        // When occlusion is enabled, occlusion passes own the history updates.
        // When disabled, force everything that passes frustum cull to be visible.
        if (main_cull_uniform.enable_occlusion == 0u) {
            visibility_history[index].visible = 1u;
        }
    } else {
        visibility_history[index].visible = 0u;
    }
}

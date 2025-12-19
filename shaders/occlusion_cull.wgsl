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

struct Counters {
    visible_count: atomic<u32>,
};

struct Params {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    // x=width, y=height, z=bias, w=slack
    screen_bias: vec4<f32>,
};

struct HistoryVisibility {
    visible: u32,
}

@group(0) @binding(0)
var<storage, read> visible_instances: array<u32>;

@group(0) @binding(1)
var<storage, read> instances: array<Instance>;

@group(0) @binding(2)
var<storage, read> mesh_table: array<MeshTableEntry>;

@group(0) @binding(3)
var<storage, read> counters: Counters;

@group(0) @binding(4)
var<storage, read_write> history_visibility: array<HistoryVisibility>;

// Hi-Z mip0 (R32Float)
@group(0) @binding(5)
var hiz: texture_2d<f32>;

@group(0) @binding(6)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = atomicLoad(&counters.visible_count);
    if (idx >= count) {
        return;
    }

    let instance_index = visible_instances[idx];
    let inst = instances[instance_index];
    let mesh = mesh_table[inst.mesh_id];

    let world_center = (inst.model * vec4<f32>(mesh.sphere.xyz, 1.0)).xyz;
    
    // Extract world-space radius (scaled by model transform)
    let c0 = inst.model[0].xyz;
    let c1 = inst.model[1].xyz;
    let c2 = inst.model[2].xyz;
    let max_scale = max(length(c0), max(length(c1), length(c2)));
    let world_radius = mesh.sphere.w * max_scale;
    
    let view_center_temp = (params.view * vec4<f32>(world_center, 1.0)).xyz;
    
    // Move sphere center toward camera by its radius in view space
    // (view_center_temp.z is negative in right-handed view space, so subtract radius)
    let view_center = view_center_temp + vec3<f32>(0.0, 0.0, world_radius);

    let clip = params.proj * vec4<f32>(view_center, 1.0);
    if (clip.w <= 0.0) {
        history_visibility[instance_index].visible = 0u;
        return;
    }

    let ndc = clip.xyz / clip.w;

    // Conservative: if projection is outside the viewport or depth is out of range, keep visible.
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        history_visibility[instance_index].visible = 0u;
        return;
    }

    let w = params.screen_bias.x;
    let h = params.screen_bias.y;

    // NDC -> pixel. Note: flip Y for texture space.
    let fx = (ndc.x * 0.5 + 0.5) * w;
    let fy = (-ndc.y * 0.5 + 0.5) * h;

    let px = clamp(i32(floor(fx)), 0, i32(w) - 1);
    let py = clamp(i32(floor(fy)), 0, i32(h) - 1);

    // TODO
    let hiz_depth = textureLoad(hiz, vec2<i32>(px, py), 0).x;

    // Standard Z (near=0, far=1): larger depth means farther.
    // Conservative cull: only mark invisible if we are safely behind Hi-Z.
    // Note: "slack" is an *extra bias* added on top of bias.
    let bias = params.screen_bias.z;
    let slack = params.screen_bias.w;
    let object_depth = ndc.z;

    if (object_depth > hiz_depth + bias + slack) {
        history_visibility[instance_index].visible = 0u;
    } else {
        history_visibility[instance_index].visible = 1u;
    }
}

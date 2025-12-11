struct Instance {
    model: mat4x4<f32>,
    sphere: vec4<f32>, // center.xyz, radius
};

// 6 个 world space 平面：normal.xyz, d
struct Frustum {
    planes: array<vec4<f32>, 6>,
};

struct Params {
    instance_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> instances: array<Instance>;

@group(0) @binding(1)
var<storage, read_write> visibility: array<u32>;

@group(0) @binding(2)
var<uniform> frustum: Frustum;

@group(0) @binding(3)
var<uniform> params: Params;

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

    // local sphere center -> world space
    let local_center = vec4<f32>(inst.sphere.xyz, 1.0);
    let world_center = (inst.model * local_center).xyz;
    let radius = inst.sphere.w;

    var visible = true;

    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let plane = frustum.planes[i];
        let dist = dot(plane.xyz, world_center) + plane.w;
        if (dist < -radius) {
            visible = false;
            break;
        }
    }

    visibility[index] = select(0u, 1u, visible);
}
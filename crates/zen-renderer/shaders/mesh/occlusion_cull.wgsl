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

fn calc_mip_level_count(width: u32, height: u32) -> u32 {
    let max_dim = max(1u, max(width, height));
    return 32u - countLeadingZeros(max_dim);
}

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

    let world_center = inst.model * vec4<f32>(mesh.sphere.xyz, 1.0);

    let view_center4 = params.view * world_center;

    // Extract world-space radius (scaled by model transform)
    let c0 = inst.model[0].xyz;
    let c1 = inst.model[1].xyz;
    let c2 = inst.model[2].xyz;
    let max_scale = max(length(c0), max(length(c1), length(c2)));
    let radius = mesh.sphere.w * max_scale;

    let clip_center = params.proj * view_center4;
    if (clip_center.w <= 0.0) {
        history_visibility[instance_index].visible = 1u;
        return;
    }

    let ndc_center = clip_center.xyz / clip_center.w;

    let w = params.screen_bias.x;
    let h = params.screen_bias.y;

    // NDC -> pixel. Note: flip Y for texture space.
    let fx = (ndc_center.x * 0.5 + 0.5) * w;
    let fy = (-ndc_center.y * 0.5 + 0.5) * h;

    // --- 1) 估算包围球在屏幕的投影半径（像素） ---
    // 在 view space 中沿 x/y 偏移 radius，然后投影到 ndc，比中心点的 ndc 偏移量。
    // （假设 view 矩阵不含缩放；一般相机 view 只有旋转/平移）
    let clip_x = params.proj * vec4<f32>(view_center4.xyz + vec3<f32>(radius, 0.0, 0.0), 1.0);
    let clip_y = params.proj * vec4<f32>(view_center4.xyz + vec3<f32>(0.0, radius, 0.0), 1.0);

    // 如果偏移点无法投影，就无法稳定估半径；跳过 occlusion（保守）
    if (clip_x.w <= 0.0 || clip_y.w <= 0.0) {
        history_visibility[instance_index].visible = 1u;
        return;
    }

    let ndc_x = clip_x.xyz / clip_x.w;
    let ndc_y = clip_y.xyz / clip_y.w;

    let r_px_x = abs(ndc_x.x - ndc_center.x) * 0.5 * w;
    let r_px_y = abs(ndc_y.y - ndc_center.y) * 0.5 * h;
    let r_px = max(r_px_x, r_px_y);

    // NaN / 非法半径保护（NaN 与任何比较都为 false，所以用 !(r_px >= 0) 捕获 NaN）
    if (!(r_px >= 0.0)) {
        history_visibility[instance_index].visible = 1u;
        return;
    }

    // --- 2) 用投影半径形成屏幕矩形范围 ---
    let min_px = clamp(i32(floor(fx - r_px)), 0, i32(w) - 1);
    let max_px = clamp(i32(floor(fx + r_px)), 0, i32(w) - 1);
    let min_py = clamp(i32(floor(fy - r_px)), 0, i32(h) - 1);
    let max_py = clamp(i32(floor(fy + r_px)), 0, i32(h) - 1);

    // --- 3) 根据覆盖直径选择 mip（让 footprint 在该 mip 上接近 1 texel） ---
    let dims0 = textureDimensions(hiz); // level 0
    let max_mip = calc_mip_level_count(dims0.x, dims0.y) - 1u;

    let diameter_px = max(1.0, 2.0 * r_px);
    let mip_f = clamp(floor(log2(diameter_px)), 0.0, f32(max_mip));
    let mip = u32(mip_f);

    let mip_w = max(1u, dims0.x >> mip);
    let mip_h = max(1u, dims0.y >> mip);

    let min_mx = clamp(u32(min_px) >> mip, 0u, mip_w - 1u);
    let max_mx = clamp(u32(max_px) >> mip, 0u, mip_w - 1u);
    let min_my = clamp(u32(min_py) >> mip, 0u, mip_h - 1u);
    let max_my = clamp(u32(max_py) >> mip, 0u, mip_h - 1u);

    // --- 4) 在该 mip 上取范围深度（采四角），并做 conservative 比较 ---
    // 这里采用“取最大深度”更保守（标准Z: near=0 far=1，越大越远）。
    // 前提：你的 Hi-Z 金字塔在降采样时也是用 max 聚合。
    let a = textureLoad(hiz, vec2<i32>(i32(min_mx), i32(min_my)), i32(mip)).x;
    let b = textureLoad(hiz, vec2<i32>(i32(max_mx), i32(min_my)), i32(mip)).x;
    let c = textureLoad(hiz, vec2<i32>(i32(min_mx), i32(max_my)), i32(mip)).x;
    let d = textureLoad(hiz, vec2<i32>(i32(max_mx), i32(max_my)), i32(mip)).x;
    let hiz_depth = max(max(a, b), max(c, d));

    // 用球体“最靠近相机”的深度做测试（更保守：只要有一部分可能露出来，就不剔除）
    // RH view 且相机朝 -Z：向相机方向是 +Z
    let clip_near = params.proj * vec4<f32>(view_center4.xyz + vec3<f32>(0.0, 0.0, radius), 1.0);

    // 深度不可用时跳过 occlusion（保守）
    if (clip_near.w <= 0.0) {
        history_visibility[instance_index].visible = 1u;
        return;
    }

    let ndc_near = select(ndc_center, clip_near.xyz / clip_near.w, clip_near.w > 0.0);
    let object_depth = ndc_near.z;

    // 如果深度不在 0..1（WebGPU NDC 深度范围），就别做 occlusion（保守）
    if (object_depth < 0.0 || object_depth > 1.0) {
        history_visibility[instance_index].visible = 1u;
        return;
    }

    let bias = params.screen_bias.z;
    let slack = params.screen_bias.w;

    if (object_depth > hiz_depth + bias + slack) {
        history_visibility[instance_index].visible = 0u;
    } else {
        history_visibility[instance_index].visible = 1u;
    }
}

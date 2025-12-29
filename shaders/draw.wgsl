struct VertexPacked {
    px: f32,
    py: f32,
    pz: f32,
    n_oct: u32,
    uv01: u32,
    c: u32,
    _pad0: u32,
    _pad1: u32,
};

struct Material {
    color: vec4f,
    texture_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct InstanceData {
    model: mat4x4f,
    mesh_id: u32,
    material_id: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Camera {
    view_proj: mat4x4f,
};

@group(0) @binding(0) var<storage, read> vertices: array<VertexPacked>;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(3) var<uniform> camera: Camera;

@group(1) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(1) var tex_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) tex_id: u32,
};

fn decode_position(v: VertexPacked) -> vec3f {
    return vec3f(v.px, v.py, v.pz);
}

fn oct_decode(e: vec2f) -> vec3f {
    var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let ox = (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0);
        let oy = (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0);
        n.x = ox;
        n.y = oy;
    }
    return normalize(n);
}

fn decode_normal(v: VertexPacked) -> vec3f {
    let e = unpack2x16snorm(v.n_oct);
    return oct_decode(e);
}

fn decode_uv(v: VertexPacked) -> vec2f {
    return unpack2x16unorm(v.uv01);
}

fn decode_color(v: VertexPacked) -> vec4f {
    return unpack4x8unorm(v.c);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vtx: u32,
    @builtin(instance_index) inst: u32,
) -> VsOut {
    let v: VertexPacked = vertices[vtx];
    let model = instances[inst].model;
    let material = materials[instances[inst].material_id];

    let pos = decode_position(v);
    let local_normal = decode_normal(v);
    let local_color = decode_color(v);
    let uv = decode_uv(v);

    let normal_mat = mat3x3f(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz
    );
    let world_normal = normalize(normal_mat * local_normal);

    var out: VsOut;
    out.pos = camera.view_proj * model * vec4f(pos, 1.0);
    out.color = local_color * material.color;
    out.normal = world_normal;
    out.uv = uv;
    out.tex_id = material.texture_id;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let light_dir = normalize(vec3f(0.5, 1.0, 0.8));
    let n_dot_l = max(dot(in.normal, light_dir), 0.0);
    let diffuse = 0.0 + 0.9 * n_dot_l; // 环境光+漫反射

    let tex_color = textureSample(textures[in.tex_id], tex_sampler, in.uv);

    return in.color * diffuse * tex_color;
}
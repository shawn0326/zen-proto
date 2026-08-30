enable wgpu_binding_array;

struct VertexPacked {
    px: f32,
    py: f32,
    pz: f32,
    n_oct: u32,
    uv: vec2f,
    c: u32,
    _pad0: u32,
};

struct Material {
    albedo_factor: vec4f,
    // xyz: emissive factor, w: occlusion strength
    emissive_ao: vec4f,
    // x: albedo tex id, y: emissive tex id, z: occlusion tex id, w: reserved
    tex_ids: vec4u,
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
    @location(3) @interpolate(flat) material_id: u32,
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
    return v.uv;
}

fn decode_color(v: VertexPacked) -> vec4f {
    return unpack4x8unorm(v.c);
}

fn normalize_or(v: vec3f, fallback: vec3f) -> vec3f {
    let length_squared = dot(v, v);
    if (length_squared > 0.0) {
        return v * inverseSqrt(length_squared);
    }
    return fallback;
}

fn transform_normal(model: mat4x4f, local_normal: vec3f) -> vec3f {
    let model_x = model[0].xyz;
    let model_y = model[1].xyz;
    let model_z = model[2].xyz;

    // These are the columns of determinant(model) * inverse-transpose(model).
    // Retaining the determinant sign makes reflected transforms match a true
    // inverse-transpose while normalization lets us avoid an unstable division.
    let cofactor_x = cross(model_y, model_z);
    let cofactor_y = cross(model_z, model_x);
    let cofactor_z = cross(model_x, model_y);
    let determinant = dot(model_x, cofactor_x);
    let orientation = select(-1.0, 1.0, determinant >= 0.0);
    let cofactor_normal = mat3x3f(cofactor_x, cofactor_y, cofactor_z) * local_normal;

    return normalize_or(cofactor_normal * orientation, local_normal);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vtx: u32,
    @builtin(instance_index) inst: u32,
) -> VsOut {
    let v: VertexPacked = vertices[vtx];
    let model = instances[inst].model;
    let material_id = instances[inst].material_id;

    let pos = decode_position(v);
    let local_normal = decode_normal(v);
    let local_color = decode_color(v);
    let uv = decode_uv(v);

    let world_normal = transform_normal(model, local_normal);

    var out: VsOut;
    out.pos = camera.view_proj * model * vec4f(pos, 1.0);
    out.color = local_color;
    out.normal = world_normal;
    out.uv = uv;
    out.material_id = material_id;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let light_dir = normalize(vec3f(0.5, 1.0, 0.8));
    let world_normal = normalize_or(in.normal, vec3f(0.0, 0.0, 1.0));

    let material = materials[in.material_id];
    let albedo_tex_id = material.tex_ids.x;
    let emissive_tex_id = material.tex_ids.y;
    let occlusion_tex_id = material.tex_ids.z;

    let vertex_albedo = in.color * material.albedo_factor;
    let albedo = textureSample(textures[albedo_tex_id], tex_sampler, in.uv) * vertex_albedo;

    let emission = textureSample(textures[emissive_tex_id], tex_sampler, in.uv).rgb
        * material.emissive_ao.rgb;

    let ao_tex = textureSample(textures[occlusion_tex_id], tex_sampler, in.uv).r;
    let ao = clamp(ao_tex * material.emissive_ao.w, 0.0, 1.0);

    let n_dot_l = max(dot(world_normal, light_dir), 0.0);
    let lighted = n_dot_l * albedo.rgb;

    let ambient_color = vec3f(0.01, 0.01, 0.01);
    let ambient = ambient_color * albedo.rgb * ao;

    return vec4f(lighted + emission + ambient, albedo.a);
}

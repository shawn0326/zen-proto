enable wgpu_binding_array;

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

@group(0) @binding(0) var<storage, read> vertices: array<VertexPacked>;
@group(0) @binding(1) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(3) var<storage, read> visible: array<Work>;
@group(0) @binding(4) var<uniform> raster: RasterUniform;
@group(1) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(1) var samplers: binding_array<sampler>;

fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    return select(fallback, value * inverseSqrt(length_squared), length_squared > 1e-12);
}

fn decode_normal(value: u32) -> vec3<f32> {
    let encoded = unpack2x16snorm(value);
    var normal = vec3<f32>(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (normal.z < 0.0) {
        let folded = (vec2<f32>(1.0) - abs(normal.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), normal.xy >= vec2<f32>(0.0));
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

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) visible_id: u32) -> VertexOutput {
    let work = visible[visible_id];
    let instance = instances[work.instance_id];
    let vertex = vertices[vertex_id];
    var output: VertexOutput;
    output.position = raster.view_projection * instance.model * vec4<f32>(vertex.px, vertex.py, vertex.pz, 1.0);
    output.color = unpack4x8unorm(vertex.color);
    output.normal = transform_normal(instance.model, decode_normal(vertex.normal_oct));
    output.uv = vertex.uv;
    output.material_id = work.material_id;
    output.meshlet_id = work.meshlet_id;
    output.render_mode = raster.render_mode;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (input.render_mode == 1u) {
        return vec4<f32>(meshlet_debug_color(input.meshlet_id), 1.0);
    }
    let material = materials[input.material_id];
    let base = textureSample(textures[material.albedo.x], samplers[material.albedo.y], input.uv)
        * material.albedo_factor * input.color;
    let emission = textureSample(textures[material.emissive.x], samplers[material.emissive.y], input.uv).rgb
        * material.emissive_ao.rgb;
    let ao = clamp(textureSample(textures[material.occlusion.x], samplers[material.occlusion.y], input.uv).r
        * material.emissive_ao.w, 0.0, 1.0);
    let normal = normalize_or(input.normal, vec3<f32>(0.0, 0.0, 1.0));
    let lighting = max(dot(normal, normalize(vec3<f32>(0.5, 1.0, 0.8))), 0.0);
    return vec4<f32>(base.rgb * lighting + base.rgb * ao * 0.01 + emission, base.a);
}

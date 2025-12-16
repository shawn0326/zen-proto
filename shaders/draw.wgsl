struct Vertex {
    position: vec4f,
    color: vec4f,
};

struct Material {
    color: vec4f,
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

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(3) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vtx: u32,
    @builtin(instance_index) inst: u32,
) -> VsOut {
    let v: Vertex = vertices[vtx];
    let model = instances[inst].model;
    let material = materials[instances[inst].material_id];

    var out: VsOut;
    out.pos = camera.view_proj * model * v.position;
    out.color = v.color * material.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return in.color;
}
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

// Fullscreen triangle.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );

    var uvs = array<vec2f, 3>(
        vec2f(0.0, 0.0),
        vec2f(2.0, 0.0),
        vec2f(0.0, 2.0),
    );

    var o: VsOut;
    o.pos = vec4f(positions[vid], 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return textureSample(src_tex, src_samp, in.uv);
}

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

    let p = positions[vid];

    var o: VsOut;
    o.pos = vec4f(p, 0.0, 1.0);

    // NDC(-1..1) -> UV(0..1), 注意翻转Y
    var uv = p * 0.5 + vec2f(0.5, 0.5);
    uv.y = 1.0 - uv.y;
    o.uv = uv;

    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return textureSample(src_tex, src_samp, in.uv);
}

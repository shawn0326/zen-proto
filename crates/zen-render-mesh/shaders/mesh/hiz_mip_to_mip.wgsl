// Hi-Z mip N-1 -> Hi-Z mip N
//
// Standard Z (near=0, far=1, depth_compare=Less): conservative reduction is MAX.

const WORKGROUP_X: u32 = 8u;
const WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var hiz_src: texture_2d<f32>;
@group(0) @binding(1) var hiz_dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(hiz_dst);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) {
        return;
    }

    let src_dims = textureDimensions(hiz_src);
    // Native mip dimensions round odd sizes down. A fixed 2x2 footprint would therefore drop the
    // final source row/column (for example 13 -> 6), making a max pyramid non-conservative. Map
    // each destination texel to a conservative proportional source interval. The end uses ceil,
    // deliberately overlapping adjacent intervals for NPOT chains; this makes every native mip
    // texel a superset of the normalized footprint used later by floor(uv * mip_dimensions).
    let source_begin = vec2<u32>(
        gid.x * src_dims.x / dst_dims.x,
        gid.y * src_dims.y / dst_dims.y,
    );
    let source_end = vec2<u32>(
        ((gid.x + 1u) * src_dims.x + dst_dims.x - 1u) / dst_dims.x,
        ((gid.y + 1u) * src_dims.y + dst_dims.y - 1u) / dst_dims.y,
    );

    var reduced = 0.0;
    for (var y = source_begin.y; y < source_end.y; y += 1u) {
        for (var x = source_begin.x; x < source_end.x; x += 1u) {
            reduced = max(reduced, textureLoad(hiz_src, vec2<i32>(i32(x), i32(y)), 0).x);
        }
    }
    let d = reduced;
    textureStore(hiz_dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(d, 0.0, 0.0, 0.0));
}

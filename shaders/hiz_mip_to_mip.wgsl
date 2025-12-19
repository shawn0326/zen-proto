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
    let base_x = gid.x * 2u;
    let base_y = gid.y * 2u;

    let x0 = min(base_x + 0u, src_dims.x - 1u);
    let x1 = min(base_x + 1u, src_dims.x - 1u);
    let y0 = min(base_y + 0u, src_dims.y - 1u);
    let y1 = min(base_y + 1u, src_dims.y - 1u);

    let d00 = textureLoad(hiz_src, vec2<i32>(i32(x0), i32(y0)), 0).x;
    let d10 = textureLoad(hiz_src, vec2<i32>(i32(x1), i32(y0)), 0).x;
    let d01 = textureLoad(hiz_src, vec2<i32>(i32(x0), i32(y1)), 0).x;
    let d11 = textureLoad(hiz_src, vec2<i32>(i32(x1), i32(y1)), 0).x;

    let d = max(max(d00, d10), max(d01, d11));
    textureStore(hiz_dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(d, 0.0, 0.0, 0.0));
}

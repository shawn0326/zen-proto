// Depth32Float -> Hi-Z mip0 (R32Float)

const WORKGROUP_X: u32 = 8u;
const WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var hiz_dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let d = textureLoad(depth_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    textureStore(hiz_dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(d, 0.0, 0.0, 0.0));
}

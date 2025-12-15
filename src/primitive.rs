#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Primitive {
    pub transform: glam::Mat4,
    pub sphere: glam::Vec4, // xyz: center, w: radius
}

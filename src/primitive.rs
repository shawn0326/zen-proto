#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Primitive {
    pub transform: glam::Mat4,
    pub mesh_id: u32,
    pub _pad: [u32; 3],
}

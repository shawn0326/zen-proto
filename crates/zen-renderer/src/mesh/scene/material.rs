#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub albedo_factor: glam::Vec4,
    /// xyz: emissive factor (linear), w: occlusion strength
    pub emissive_ao: glam::Vec4,
    /// x: albedo tex id, y: emissive tex id, z: occlusion tex id, w: reserved
    pub tex_ids: [u32; 4],
}

pub(crate) struct MaterialStorage {
    material_buffer: wgpu::Buffer,
}

impl MaterialStorage {
    pub fn from_materials(device: &wgpu::Device, materials: &[Material]) -> Self {
        let material_buffer = super::create_non_empty_buffer_init(
            device,
            "materials.material_buffer",
            materials,
            wgpu::BufferUsages::STORAGE,
        );

        Self { material_buffer }
    }

    pub(crate) fn material_buffer(&self) -> &wgpu::Buffer {
        &self.material_buffer
    }
}

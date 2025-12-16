#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub color: glam::Vec4,
}

pub struct MaterialsContext {
    pub material_buffer: wgpu::Buffer,
    pub material_count: u32,
}

impl MaterialsContext {
    pub fn from_materials(device: &wgpu::Device, materials: &[Material]) -> Self {
        use wgpu::util::DeviceExt;

        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials.material_buffer"),
            contents: bytemuck::cast_slice(materials),
            usage: wgpu::BufferUsages::STORAGE,
        });

        MaterialsContext {
            material_buffer,
            material_count: materials.len() as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub color: glam::Vec4,
    pub texture_id: u32,
    pub _pad: [u32; 3],
}

pub struct MaterialStorage {
    material_buffer: wgpu::Buffer,
    material_count: u32,
}

impl MaterialStorage {
    pub fn from_materials(device: &wgpu::Device, materials: &[Material]) -> Self {
        use wgpu::util::DeviceExt;

        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials.material_buffer"),
            contents: bytemuck::cast_slice(materials),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            material_buffer,
            material_count: materials.len() as u32,
        }
    }

    pub(crate) fn material_buffer(&self) -> &wgpu::Buffer {
        &self.material_buffer
    }

    pub fn material_count(&self) -> u32 {
        self.material_count
    }
}

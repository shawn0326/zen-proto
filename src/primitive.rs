#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Primitive {
    pub transform: glam::Mat4,
    pub mesh_id: u32,
    pub material_id: u32,
    pub _pad: [u32; 2],
}

pub struct PrimitiveStorage {
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}

impl PrimitiveStorage {
    pub fn from_primitives(device: &wgpu::Device, primitives: &[Primitive]) -> Self {
        use wgpu::util::DeviceExt;

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("primitives.instance_buffer"),
            contents: bytemuck::cast_slice(primitives),
            usage: wgpu::BufferUsages::STORAGE,
        });

        PrimitiveStorage {
            instance_buffer,
            instance_count: primitives.len() as u32,
        }
    }
}

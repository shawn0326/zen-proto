#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub transform: glam::Mat4,
    pub mesh_id: u32,
    pub material_id: u32,
    pub _pad: [u32; 2],
}

pub(crate) struct InstanceStorage {
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
}

impl InstanceStorage {
    pub fn from_instances(device: &wgpu::Device, instances: &[Instance]) -> Self {
        use wgpu::util::DeviceExt;

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances.instance_buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            instance_buffer,
            instance_count: instances.len() as u32,
        }
    }

    pub(crate) fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }
}

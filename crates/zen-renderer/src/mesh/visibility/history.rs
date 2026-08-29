pub struct VisibilityHistory {
    buffer: wgpu::Buffer,
}

impl VisibilityHistory {
    pub fn new(device: &wgpu::Device, max_instance_count: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_history"),
            size: (max_instance_count.max(1) as u64) * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        VisibilityHistory { buffer }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

pub struct VisibilityList {
    id: u64,
    label: String,
    max_instance_count: u32,
    visible_instances: wgpu::Buffer,
    visible_count: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    draw_count: wgpu::Buffer,
}

impl VisibilityList {
    pub fn new(device: &wgpu::Device, label: &str, max_instance_count: u32) -> Self {
        let visible_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}.visible_instances", label)),
            size: (max_instance_count as u64) * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let visible_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}.visible_count", label)),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dispatch_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}.dispatch_args", label)),
            size: std::mem::size_of::<DispatchIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let draw_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}.draw_args", label)),
            size: (max_instance_count as u64)
                * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let draw_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}.draw_count", label)),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        VisibilityList {
            id: {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                label.hash(&mut hasher);
                hasher.finish()
            },
            label: label.to_string(),
            max_instance_count,
            visible_instances,
            visible_count,
            dispatch_args,
            draw_args,
            draw_count,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn max_instance_count(&self) -> u32 {
        self.max_instance_count
    }

    pub fn visible_instances_buffer(&self) -> &wgpu::Buffer {
        &self.visible_instances
    }

    pub fn visible_count_buffer(&self) -> &wgpu::Buffer {
        &self.visible_count
    }

    pub fn dispatch_args_buffer(&self) -> &wgpu::Buffer {
        &self.dispatch_args
    }

    pub fn draw_args_buffer(&self) -> &wgpu::Buffer {
        &self.draw_args
    }

    pub fn draw_count_buffer(&self) -> &wgpu::Buffer {
        &self.draw_count
    }

    pub fn reset(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.visible_count, 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(&self.draw_count, 0, bytemuck::bytes_of(&0u32));
    }
}

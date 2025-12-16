use crate::{
    camera::Camera,
    render::{MeshesContext, PrimitivesContext},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrustumUniform {
    // 6 planes in world space: normal.xyz, d (Ax + By + Cz + D = 0)
    pub planes: [glam::Vec4; 6],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullParams {
    pub instance_count: u32,
    pub mesh_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

pub struct CullResources {
    pub cull_pipeline: wgpu::ComputePipeline,
    pub cull_bind_group: wgpu::BindGroup,
    pub indirect_args_buffer: wgpu::Buffer,
    pub indirect_count_buffer: wgpu::Buffer,
    pub frustum_buffer: wgpu::Buffer,
}

impl CullResources {
    pub fn update_frustum(&self, queue: &wgpu::Queue, camera: &Camera) {
        queue.write_buffer(
            &self.frustum_buffer,
            0,
            bytemuck::bytes_of(&camera.frustum()),
        );
    }

    /// 每帧调用，重置裁剪输出 buffer
    pub fn reset_indirect_buffers(&self, queue: &wgpu::Queue) {
        // 重置 indirect_count_buffer 为 0
        let zero: u32 = 0;
        queue.write_buffer(&self.indirect_count_buffer, 0, bytemuck::bytes_of(&zero));

        // 可选：重置 indirect_args_buffer（通常只需重置 count，如果 args buffer内容不影响shader可省略）
        // 如果需要清空所有 args，可以这样：
        // let zeros = vec![0u8; self.instance_count as usize * std::mem::size_of::<DrawIndexedIndirectArgs>()];
        // queue.write_buffer(&self.indirect_args_buffer, 0, &zeros);
    }
}

pub fn create_cull_resources(
    device: &wgpu::Device,
    meshes: &MeshesContext,
    primitives: &PrimitivesContext,
) -> CullResources {
    use wgpu::util::DeviceExt;

    // 新增：Indirect Args Buffer（最多 instance_count 条）
    let indirect_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Indirect Args Buffer"),
        size: primitives.instance_count as u64
            * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 新增：Indirect Count Buffer（u32 计数；shader 用 atomic 写入）
    let indirect_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Indirect Count Buffer"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 创建 Frustum Uniform Buffer
    let frustum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frustum Uniform Buffer"),
        size: std::mem::size_of::<FrustumUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 创建 Params Buffer
    let cull_params = CullParams {
        instance_count: primitives.instance_count,
        mesh_count: 2,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cull Params Buffer"),
        contents: bytemuck::bytes_of(&cull_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cull Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cull.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Cull BGL"),
        entries: &[
            // instances
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // indirect_args
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // frustum
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // indirect_count (atomic u32)
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // mesh table
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Cull Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Frustum Culling Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Cull Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: primitives.instance_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: indirect_args_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: frustum_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: indirect_count_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: meshes.mesh_table_buffer.as_entire_binding(),
            },
        ],
    });

    CullResources {
        cull_pipeline,
        cull_bind_group,
        indirect_args_buffer,
        indirect_count_buffer,
        frustum_buffer,
    }
}

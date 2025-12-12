#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model: glam::Mat4,
    pub sphere: glam::Vec4, // local space sphere: center.xyz, radius
}

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
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
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

fn extract_frustum_from_matrix(view_proj: glam::Mat4) -> FrustumUniform {
    let m = view_proj.to_cols_array();
    // m: [f32; 16], 按列主序存储
    // 但我们要按行主序访问
    // 行0: m[0] m[4] m[8]  m[12]
    // 行1: m[1] m[5] m[9]  m[13]
    // 行2: m[2] m[6] m[10] m[14]
    // 行3: m[3] m[7] m[11] m[15]

    let row = |i| glam::Vec4::new(m[i], m[i + 4], m[i + 8], m[i + 12]);

    let m0 = row(0);
    let m1 = row(1);
    let m2 = row(2);
    let m3 = row(3);

    let mut planes = [glam::Vec4::ZERO; 6];
    planes[0] = (m3 + m0).normalize(); // left
    planes[1] = (m3 - m0).normalize(); // right
    planes[2] = (m3 + m1).normalize(); // bottom
    planes[3] = (m3 - m1).normalize(); // top
    planes[4] = (m3 + m2).normalize(); // near
    planes[5] = (m3 - m2).normalize(); // far

    FrustumUniform { planes }
}

pub struct CullResources {
    pub cull_pipeline: wgpu::ComputePipeline,
    pub cull_bind_group: wgpu::BindGroup,
    pub instance_count: u32,
    pub indirect_args_buffer: wgpu::Buffer,
    pub indirect_count_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
}

pub fn create_cull_resources(device: &wgpu::Device) -> CullResources {
    use wgpu::util::DeviceExt;

    let instance_count = 1000000u32;

    // 创建 Instance Buffer
    let mut instances = Vec::new();
    let grid = (instance_count as f32).cbrt().ceil() as u32; // 100
    let spacing = 3.0;
    for i in 0..instance_count {
        let x = (i % grid) as f32 - (grid as f32 - 1.0) * 0.5;
        let y = ((i / grid) % grid) as f32 - (grid as f32 - 1.0) * 0.5;
        let z = (i / (grid * grid)) as f32 - (grid as f32 - 1.0) * 0.5;
        let translation = glam::vec3(x * spacing, y * spacing, z * spacing);
        let model = glam::Mat4::from_translation(translation);
        let sphere = glam::Vec4::new(0.0, 0.0, 0.0, 1.0); // 半径为 1 的单位球体
        let instance = InstanceData { model, sphere };
        instances.push(instance);
    }
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // 新增：Indirect Args Buffer（最多 instance_count 条）
    let indirect_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Indirect Args Buffer"),
        size: instance_count as u64 * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
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
    let view_mat = glam::Mat4::look_at_rh(
        glam::vec3(0.0, 0.0, 10.0),
        glam::vec3(0.0, 0.0, 0.0),
        glam::vec3(0.0, 1.0, 0.0),
    );
    let proj_mat = glam::Mat4::perspective_rh_gl(45.0f32.to_radians(), 800.0 / 600.0, 0.1, 1000.0);
    let frustum_uniform = extract_frustum_from_matrix(proj_mat * view_mat);
    let frustum_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Frustum Uniform Buffer"),
        contents: bytemuck::bytes_of(&frustum_uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // 创建 Params Buffer
    let cull_params = CullParams {
        instance_count,
        index_count: 3,
        first_index: 0,
        base_vertex: 0,
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
                resource: instance_buffer.as_entire_binding(),
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
        ],
    });

    CullResources {
        cull_pipeline,
        cull_bind_group,
        instance_count,
        indirect_args_buffer,
        indirect_count_buffer,
        instance_buffer,
    }
}

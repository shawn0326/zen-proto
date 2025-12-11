#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model: glam::Mat4,
    // local space sphere: center.xyz, radius
    pub sphere: glam::Vec4,
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
    pub _pad: [u32; 3], // 保证 16 字节对齐
}

fn extract_frustum_from_matrix(view_proj: glam::Mat4) -> FrustumUniform {
    // 标准从 view-proj 矩阵提取 6 个平面的写法
    // 矩阵按列存储：m.col(i)
    let m = view_proj.to_cols_array_2d();

    // 行向量
    let (m0, m1, m2, m3) = (
        glam::Vec4::from(m[0]),
        glam::Vec4::from(m[1]),
        glam::Vec4::from(m[2]),
        glam::Vec4::from(m[3]),
    );

    let mut planes = [glam::Vec4::ZERO; 6];

    // left:  m3 + m0
    let p_left = (m3 + m0).normalize();
    planes[0] = p_left;

    // right: m3 - m0
    let p_right = (m3 - m0).normalize();
    planes[1] = p_right;

    // bottom: m3 + m1
    let p_bottom = (m3 + m1).normalize();
    planes[2] = p_bottom;

    // top: m3 - m1
    let p_top = (m3 - m1).normalize();
    planes[3] = p_top;

    // near: m3 + m2
    let p_near = (m3 + m2).normalize();
    planes[4] = p_near;

    // far:  m3 - m2
    let p_far = (m3 - m2).normalize();
    planes[5] = p_far;

    FrustumUniform { planes }
}

pub struct CullResources {
    pub cull_pipeline: wgpu::ComputePipeline,
    pub cull_bind_group: wgpu::BindGroup,
    pub instance_count: u32,
}

pub fn create_cull_resources(device: &wgpu::Device) -> CullResources {
    use wgpu::util::DeviceExt;

    let instance_count = 1000;

    // 创建 Instance Buffer
    let mut instances = Vec::new();
    for i in 0..instance_count {
        let translation = glam::vec3(
            (i % 10) as f32 * 3.0,
            ((i / 10) % 10) as f32 * 3.0,
            (i / 100) as f32 * 3.0,
        );
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

    // 创建 Visibility Buffer
    let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Visibility Buffer"),
        size: (instance_count as u64 * std::mem::size_of::<u32>() as u64),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::INDIRECT, // 之后做 indirect 用得上
        mapped_at_creation: false,
    });

    // 创建 Frustum Uniform Buffer
    let view_mat = glam::Mat4::look_at_rh(
        glam::vec3(15.0, 15.0, 15.0),
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
        _pad: [0; 3],
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
            // visibility
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
                resource: visibility_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: frustum_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    CullResources {
        cull_pipeline,
        cull_bind_group,
        instance_count,
    }
}

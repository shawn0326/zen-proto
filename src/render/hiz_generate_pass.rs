use crate::render::HiZTexture;

pub struct HiZGeneratePass {
    depth_to_mip0_pipeline: wgpu::ComputePipeline,
    depth_to_mip0_bgl: wgpu::BindGroupLayout,
    mip_to_mip_pipeline: wgpu::ComputePipeline,
    mip_to_mip_bgl: wgpu::BindGroupLayout,
}

impl HiZGeneratePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let depth_to_mip0_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_depth_to_mip0.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/hiz_depth_to_mip0.wgsl").into(),
            ),
        });

        let mip_to_mip_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_mip_to_mip.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/hiz_mip_to_mip.wgsl").into(),
            ),
        });

        // depth_to_mip0 bindings:
        // 0 depth_tex (sampled depth)
        // 1 hiz_dst (storage r32float, write)
        let depth_to_mip0_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz.depth_to_mip0_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let depth_to_mip0_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hiz.depth_to_mip0_pipeline_layout"),
                bind_group_layouts: &[&depth_to_mip0_bgl],
                push_constant_ranges: &[],
            });

        let depth_to_mip0_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hiz.depth_to_mip0_pipeline"),
                layout: Some(&depth_to_mip0_pipeline_layout),
                module: &depth_to_mip0_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // mip_to_mip bindings:
        // 0 hiz_src (sampled float, non-filterable)
        // 1 hiz_dst (storage r32float, write)
        let mip_to_mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz.mip_to_mip_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let mip_to_mip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hiz.mip_to_mip_pipeline_layout"),
                bind_group_layouts: &[&mip_to_mip_bgl],
                push_constant_ranges: &[],
            });

        let mip_to_mip_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hiz.mip_to_mip_pipeline"),
                layout: Some(&mip_to_mip_pipeline_layout),
                module: &mip_to_mip_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            depth_to_mip0_pipeline,
            depth_to_mip0_bgl,
            mip_to_mip_pipeline,
            mip_to_mip_bgl,
        }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        hiz: &HiZTexture,
    ) {
        fn ceil_div(a: u32, b: u32) -> u32 {
            (a + b - 1) / b
        }

        fn mip_dim(mut base: u32, level: u32) -> u32 {
            base = base >> level;
            base.max(1)
        }

        // Pass 1: depth -> HiZ mip0
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz.depth_to_mip0_bg"),
            layout: &self.depth_to_mip0_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(hiz.storage_view(0)),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ: depth_to_mip0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.depth_to_mip0_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.dispatch_workgroups(ceil_div(hiz.width(), 8), ceil_div(hiz.height(), 8), 1);
        }

        // Pass 2: mip pyramid (HiZ mip N-1 -> mip N)
        let mip_levels = hiz.mip_level_count();
        for mip in 1..mip_levels {
            let src_view = hiz.sampled_view(mip - 1);
            let dst_view = hiz.storage_view(mip);

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hiz.mip_to_mip_bg"),
                layout: &self.mip_to_mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(dst_view),
                    },
                ],
            });

            let dst_w = mip_dim(hiz.width(), mip);
            let dst_h = mip_dim(hiz.height(), mip);

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ: mip_to_mip"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mip_to_mip_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(ceil_div(dst_w, 8), ceil_div(dst_h, 8), 1);
        }
    }
}

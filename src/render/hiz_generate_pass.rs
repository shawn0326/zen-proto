use crate::render::HiZTexture;

pub struct HiZGeneratePass {
    depth_to_mip0_pipeline: wgpu::ComputePipeline,
    depth_to_mip0_bgl: wgpu::BindGroupLayout,
    mip_to_mip_pipeline: wgpu::ComputePipeline,
    mip_to_mip_bgl: wgpu::BindGroupLayout,

    // Cached bind groups (rebuild on resize / texture recreation)
    depth_to_mip0_bg: Option<wgpu::BindGroup>,
    mip_to_mip_bgs: Vec<wgpu::BindGroup>,
    cached_mip_levels: u32,

    cached_width: u32,
    cached_height: u32,
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

            // Cached bind groups (rebuild on resize / texture recreation)
            depth_to_mip0_bg: None,
            mip_to_mip_bgs: Vec::new(),
            cached_mip_levels: 0,

            cached_width: 0,
            cached_height: 0,
        }
    }

    fn ceil_div(a: u32, b: u32) -> u32 {
        (a + b - 1) / b
    }

    fn mip_dim(mut base: u32, level: u32) -> u32 {
        base >>= level;
        base.max(1)
    }

    /// Call this after:
    /// - depth texture view changed (resize / recreated)
    /// - HiZ texture recreated (resize)
    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        hiz: &HiZTexture,
    ) {
        // depth -> mip0
        self.depth_to_mip0_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }));

        // mip(n-1) -> mip(n)
        self.mip_to_mip_bgs.clear();
        let mip_levels = hiz.mip_level_count();
        self.mip_to_mip_bgs
            .reserve((mip_levels.saturating_sub(1)) as usize);

        for mip in 1..mip_levels {
            let src_view = hiz.sampled_view(mip - 1);
            let dst_view = hiz.storage_view(mip);

            self.mip_to_mip_bgs
                .push(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                }));
        }

        self.cached_mip_levels = mip_levels;
        self.cached_width = hiz.width();
        self.cached_height = hiz.height();
    }

    pub fn needs_rebuild(&self, hiz: &HiZTexture) -> bool {
        self.depth_to_mip0_bg.is_none()
            || self.cached_mip_levels != hiz.mip_level_count()
            || self.cached_width != hiz.width()
            || self.cached_height != hiz.height()
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>,
        hiz: &HiZTexture,
    ) {
        // 保险：如果没 rebuild 或缓存不匹配，直接放弃
        if self.depth_to_mip0_bg.is_none()
            || self.cached_mip_levels != hiz.mip_level_count()
            || self.cached_width != hiz.width()
            || self.cached_height != hiz.height()
        {
            return;
        }

        // Pass 1: depth -> HiZ mip0
        {
            let bg0 = self.depth_to_mip0_bg.as_ref().unwrap();
            let mut pass = encoder.scoped_compute_pass("HiZ: depth_to_mip0");
            pass.set_pipeline(&self.depth_to_mip0_pipeline);
            pass.set_bind_group(0, bg0, &[]);
            pass.dispatch_workgroups(
                Self::ceil_div(hiz.width(), 8),
                Self::ceil_div(hiz.height(), 8),
                1,
            );
        }

        // Pass 2: mip pyramid
        // NOTE: 这里仍按 mip 分 pass，避免潜在的跨-dispatch 读写 hazard。
        let mip_levels = hiz.mip_level_count();
        for mip in 1..mip_levels {
            let dst_w = Self::mip_dim(hiz.width(), mip);
            let dst_h = Self::mip_dim(hiz.height(), mip);

            let bg = &self.mip_to_mip_bgs[(mip - 1) as usize];

            let mut pass = encoder.scoped_compute_pass("HiZ: mip_to_mip");
            pass.set_pipeline(&self.mip_to_mip_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(Self::ceil_div(dst_w, 8), Self::ceil_div(dst_h, 8), 1);
        }
    }
}

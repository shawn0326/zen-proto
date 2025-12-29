pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub pixels: Vec<u8>,
}

impl Texture {
    pub fn white_1x1() -> Self {
        Self {
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            pixels: vec![255, 255, 255, 255],
        }
    }

    pub fn black_1x1() -> Self {
        Self {
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            pixels: vec![0, 0, 0, 255],
        }
    }

    pub fn validate(&self) {
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|v| v.checked_mul(4))
            .expect("Texture too large");
        assert!(
            self.format == wgpu::TextureFormat::Rgba8UnormSrgb,
            "Texture: only Rgba8UnormSrgb is supported for now"
        );
        assert!(
            self.pixels.len() == expected as usize,
            "Texture pixels size mismatch: got {}, expected {}",
            self.pixels.len(),
            expected
        );
    }
}

struct MipmapGenerator {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MipmapGenerator {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mipmap.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mipmap.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mipmap.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mipmap.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mipmap.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    fn encode_generate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        sampler: &wgpu::Sampler,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        mip_level_count: u32,
    ) {
        if mip_level_count <= 1 {
            return;
        }

        for level in 1..mip_level_count {
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mipmap.src_view"),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            });

            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mipmap.dst_view"),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mipmap.bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mipmap.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}

pub struct TextureStorage {
    max_texture_count: u32,

    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    sampler: wgpu::Sampler,
}

impl TextureStorage {
    pub const DEFAULT_MAX_TEXTURE_COUNT: u32 = 1024;

    pub fn from_textures(device: &wgpu::Device, queue: &wgpu::Queue, textures: &[Texture]) -> Self {
        let max_texture_count = Self::DEFAULT_MAX_TEXTURE_COUNT
            .min(device.limits().max_binding_array_elements_per_shader_stage);

        assert!(
            textures.len() as u32 <= max_texture_count,
            "Too many textures: {} > max_texture_count {}",
            textures.len(),
            max_texture_count
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("textures.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let mut gpu_textures: Vec<wgpu::Texture> = Vec::with_capacity(textures.len());
        let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(textures.len());

        // Cache: one pipeline for the one format we currently support.
        let mipmap_generator = MipmapGenerator::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

        let mut mipmap_jobs: Vec<(usize, u32)> = Vec::new();

        for texture in textures {
            texture.validate();
            let (gpu_texture, view, mip_level_count) =
                Self::create_gpu_texture_and_view(device, queue, texture);
            gpu_textures.push(gpu_texture);
            views.push(view);

            if mip_level_count > 1 {
                mipmap_jobs.push((gpu_textures.len() - 1, mip_level_count));
            }
        }

        if !mipmap_jobs.is_empty() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("textures.mipmaps.encoder"),
            });

            for (tex_index, mip_level_count) in &mipmap_jobs {
                let tex = &gpu_textures[*tex_index];
                mipmap_generator.encode_generate(
                    device,
                    &mut encoder,
                    &sampler,
                    tex,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    *mip_level_count,
                );
            }

            queue.submit(Some(encoder.finish()));
        }

        Self {
            max_texture_count,
            textures: gpu_textures,
            views,
            sampler,
        }
    }

    pub fn max_texture_count(&self) -> u32 {
        self.max_texture_count
    }

    pub fn texture_count(&self) -> u32 {
        self.views.len() as u32
    }

    pub fn textures(&self) -> &[wgpu::Texture] {
        &self.textures
    }

    pub fn texture_views(&self) -> &[wgpu::TextureView] {
        &self.views
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    fn create_gpu_texture_and_view(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &Texture,
    ) -> (wgpu::Texture, wgpu::TextureView, u32) {
        let bytes_per_pixel = 4u32;
        let unpadded_row_bytes = texture.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row_bytes = ((unpadded_row_bytes + align - 1) / align) * align;

        let mut padded = vec![0u8; (padded_row_bytes * texture.height) as usize];
        for y in 0..texture.height as usize {
            let src_offset = y * (unpadded_row_bytes as usize);
            let dst_offset = y * (padded_row_bytes as usize);
            padded[dst_offset..dst_offset + (unpadded_row_bytes as usize)].copy_from_slice(
                &texture.pixels[src_offset..src_offset + (unpadded_row_bytes as usize)],
            );
        }

        let mip_level_count = Self::mip_level_count_for_size(texture.width, texture.height);

        let gpu_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("textures.texture_2d"),
            size: wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = gpu_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("textures.texture_2d_view"),
            format: Some(texture.format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(texture.height),
            },
            wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
        );

        (gpu_texture, view, mip_level_count)
    }

    fn mip_level_count_for_size(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height).max(1);
        32 - max_dim.leading_zeros()
    }
}

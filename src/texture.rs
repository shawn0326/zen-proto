pub struct TextureStorage {
    width: u32,
    height: u32,
    max_texture_count: u32,
    format: wgpu::TextureFormat,

    _textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    sampler: wgpu::Sampler,

    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl TextureStorage {
    pub const DEFAULT_WIDTH: u32 = 256;
    pub const DEFAULT_HEIGHT: u32 = 256;
    pub const DEFAULT_MAX_TEXTURE_COUNT: u32 = 1024;

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let width = Self::DEFAULT_WIDTH;
        let height = Self::DEFAULT_HEIGHT;
        let max_texture_count = Self::DEFAULT_MAX_TEXTURE_COUNT;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let textures = Vec::with_capacity(2);
        let views = Vec::with_capacity(2);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("textures.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let mut textures: Vec<wgpu::Texture> = textures;
        let mut views: Vec<wgpu::TextureView> = views;

        // Default textures.
        let _white = Self::push_solid_color_texture_impl(
            device,
            queue,
            width,
            height,
            format,
            &mut textures,
            &mut views,
            max_texture_count,
            [255, 255, 255, 255],
        );
        let _black = Self::push_solid_color_texture_impl(
            device,
            queue,
            width,
            height,
            format,
            &mut textures,
            &mut views,
            max_texture_count,
            [0, 0, 0, 255],
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("textures.bindless_bgl"),
            entries: &[
                // bindless textures: texture_2d<f32> textures[]
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: std::num::NonZeroU32::new(max_texture_count),
                },
                // shared sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // With PARTIALLY_BOUND_BINDING_ARRAY, we only need to bind the subset we actually use.
        let view_refs: Vec<&wgpu::TextureView> = views.iter().collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("textures.bindless_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&view_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            width,
            height,
            max_texture_count,
            format,
            _textures: textures,
            views,
            sampler,
            bind_group_layout,
            bind_group,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn max_texture_count(&self) -> u32 {
        self.max_texture_count
    }

    pub fn texture_count(&self) -> u32 {
        self.views.len() as u32
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn texture_views(&self) -> &[wgpu::TextureView] {
        &self.views
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    fn push_solid_color_texture_impl(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        textures: &mut Vec<wgpu::Texture>,
        views: &mut Vec<wgpu::TextureView>,
        max_texture_count: u32,
        rgba: [u8; 4],
    ) -> u32 {
        if views.len() as u32 >= max_texture_count {
            return views.len() as u32;
        }

        let bytes_per_pixel = 4usize;
        let row_bytes = width as usize * bytes_per_pixel;
        debug_assert!(row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize == 0);

        let data = vec![rgba[0], rgba[1], rgba[2], rgba[3]];
        let mut pixels = Vec::with_capacity(row_bytes * height as usize);
        for _ in 0..(width as usize * height as usize) {
            pixels.extend_from_slice(&data);
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("textures.texture_2d"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("textures.texture_2d_view"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes as u32),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let id = views.len() as u32;
        textures.push(texture);
        views.push(view);

        id
    }
}

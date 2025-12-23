pub struct HiZTexture {
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    mip_level_count: u32,
    sampled_full_view: wgpu::TextureView,
    sampled_views: Vec<wgpu::TextureView>,
    storage_views: Vec<wgpu::TextureView>,
}

impl HiZTexture {
    fn calc_mip_level_count(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height).max(1);
        // WebGPU mip sizes follow integer right-shift (floor division by 2).
        // Total levels = floor(log2(max_dim)) + 1.
        32 - max_dim.leading_zeros()
    }

    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mip_level_count = Self::calc_mip_level_count(width, height);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let sampled_full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hiz_full_sampled_view"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_level_count),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        let mut sampled_views = Vec::with_capacity(mip_level_count as usize);
        let mut storage_views = Vec::with_capacity(mip_level_count as usize);
        for mip in 0..mip_level_count {
            sampled_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz_mip_sampled_view"),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            }));

            storage_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz_mip_storage_view"),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            }));
        }

        Self {
            texture,
            width,
            height,
            mip_level_count,
            sampled_full_view,
            sampled_views,
            storage_views,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    pub fn sampled_view(&self, mip: u32) -> &wgpu::TextureView {
        &self.sampled_views[mip as usize]
    }

    pub fn sampled_full_view(&self) -> &wgpu::TextureView {
        &self.sampled_full_view
    }

    pub fn storage_view(&self, mip: u32) -> &wgpu::TextureView {
        &self.storage_views[mip as usize]
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

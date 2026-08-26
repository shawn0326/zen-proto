pub struct RenderTargetContext {
    pub surface_texture: wgpu::SurfaceTexture,
    pub color_view: wgpu::TextureView,
    pub depth_stencil_view: wgpu::TextureView,
}

pub struct RenderTarget {
    surface: wgpu::Surface<'static>,
    surface_configuration: wgpu::SurfaceConfiguration,
    depth_stencil_texture: wgpu::Texture,
    depth_for_hiz_view: wgpu::TextureView,
    pending_resize: Option<(u32, u32)>,
}

impl RenderTarget {
    pub fn new(device: &wgpu::Device, surface: wgpu::Surface<'static>) -> Self {
        let surface_configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: 800,
            height: 600,
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_configuration);

        let depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_stencil_texture"),
            size: wgpu::Extent3d {
                width: surface_configuration.width,
                height: surface_configuration.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_for_hiz_view = depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("depth_for_hiz_view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        Self {
            surface,
            surface_configuration,
            depth_stencil_texture,
            depth_for_hiz_view,
            pending_resize: None,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            self.pending_resize = None;
            return;
        }

        let last_width = self
            .pending_resize
            .map(|(w, _)| w)
            .unwrap_or(self.surface_configuration.width);
        let last_height = self
            .pending_resize
            .map(|(_, h)| h)
            .unwrap_or(self.surface_configuration.height);

        if last_width == width && last_height == height {
            return;
        }

        self.pending_resize = Some((width, height));
    }

    pub fn apply_pending_resize(&mut self, device: &wgpu::Device) -> bool {
        let Some((width, height)) = self.pending_resize.take() else {
            return false;
        };

        println!("Applying render target resize to {}x{}", width, height);

        self.surface_configuration.width = width;
        self.surface_configuration.height = height;

        self.surface.configure(device, &self.surface_configuration);

        self.depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_stencil_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.depth_for_hiz_view =
            self.depth_stencil_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("depth_for_hiz_view"),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                });

        true
    }

    pub fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    pub fn depth_stencil_texture(&self) -> &wgpu::Texture {
        &self.depth_stencil_texture
    }

    pub fn width(&self) -> u32 {
        self.surface_configuration.width
    }

    pub fn height(&self) -> u32 {
        self.surface_configuration.height
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.surface_configuration.format
    }

    pub fn depth_for_hiz_view(&self) -> &wgpu::TextureView {
        &self.depth_for_hiz_view
    }

    pub(crate) fn get_target_context(&self, device: &wgpu::Device) -> Option<RenderTargetContext> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                drop(surface_texture);
                self.surface.configure(device, &self.surface_configuration);
                return None;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(device, &self.surface_configuration);
                return None;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return None;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("Surface validation error while acquiring the next frame");
                return None;
            }
        };
        let color_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_stencil_view = self
            .depth_stencil_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        Some(RenderTargetContext {
            surface_texture,
            color_view,
            depth_stencil_view,
        })
    }
}

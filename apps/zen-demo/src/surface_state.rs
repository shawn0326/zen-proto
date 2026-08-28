pub struct SurfaceState {
    surface: wgpu::Surface<'static>,
    configuration: wgpu::SurfaceConfiguration,
    pending_resize: Option<(u32, u32)>,
}

impl SurfaceState {
    pub fn new(
        device: &wgpu::Device,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Self {
        let configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(device, &configuration);
        Self {
            surface,
            configuration,
            pending_resize: None,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.pending_resize = planned_resize(
            (self.configuration.width, self.configuration.height),
            self.pending_resize,
            width,
            height,
        );
    }

    pub fn acquire(&mut self, device: &wgpu::Device) -> Option<wgpu::SurfaceTexture> {
        if apply_pending_configuration(&mut self.configuration, &mut self.pending_resize) {
            println!(
                "Applying surface resize to {}x{}",
                self.configuration.width, self.configuration.height
            );
            self.surface.configure(device, &self.configuration);
        }

        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => Some(surface_texture),
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                drop(surface_texture);
                self.surface.configure(device, &self.configuration);
                None
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(device, &self.configuration);
                None
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => None,
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("Surface validation error while acquiring the next frame");
                None
            }
        }
    }

    pub fn width(&self) -> u32 {
        self.configuration.width
    }

    pub fn height(&self) -> u32 {
        self.configuration.height
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.configuration.format
    }
}

fn planned_resize(
    current: (u32, u32),
    pending: Option<(u32, u32)>,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    if width == 0 || height == 0 {
        return pending;
    }
    if current == (width, height) {
        None
    } else if pending == Some((width, height)) {
        pending
    } else {
        Some((width, height))
    }
}

fn apply_pending_configuration(
    configuration: &mut wgpu::SurfaceConfiguration,
    pending: &mut Option<(u32, u32)>,
) -> bool {
    let Some((width, height)) = pending.take() else {
        return false;
    };
    configuration.width = width;
    configuration.height = height;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configuration() -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: 800,
            height: 600,
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        }
    }

    #[test]
    fn zero_size_does_not_replace_a_valid_pending_resize() {
        assert_eq!(
            planned_resize((800, 600), Some((1024, 768)), 0, 0),
            Some((1024, 768))
        );
    }

    #[test]
    fn duplicate_effective_size_does_not_schedule_configuration() {
        assert_eq!(planned_resize((800, 600), None, 800, 600), None);
        assert_eq!(
            planned_resize((800, 600), Some((1024, 768)), 1024, 768),
            Some((1024, 768))
        );
    }

    #[test]
    fn latest_non_zero_resize_replaces_the_pending_size() {
        assert_eq!(
            planned_resize((800, 600), Some((1024, 768)), 1280, 720),
            Some((1280, 720))
        );
    }

    #[test]
    fn resizing_back_to_the_configured_size_cancels_pending_work() {
        assert_eq!(
            planned_resize((800, 600), Some((1024, 768)), 800, 600),
            None
        );
    }

    #[test]
    fn applying_pending_resize_updates_configuration_once() {
        let mut configuration = configuration();
        let mut pending = Some((1280, 720));
        assert!(apply_pending_configuration(
            &mut configuration,
            &mut pending
        ));
        assert_eq!((configuration.width, configuration.height), (1280, 720));
        assert!(!apply_pending_configuration(
            &mut configuration,
            &mut pending
        ));
    }
}

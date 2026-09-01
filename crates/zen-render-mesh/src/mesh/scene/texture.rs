#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureAddressMode {
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureMagFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureMinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

impl TextureMinFilter {
    fn wgpu_settings(self) -> (wgpu::FilterMode, wgpu::MipmapFilterMode, bool) {
        match self {
            Self::Nearest => (
                wgpu::FilterMode::Nearest,
                wgpu::MipmapFilterMode::Nearest,
                false,
            ),
            Self::Linear => (
                wgpu::FilterMode::Linear,
                wgpu::MipmapFilterMode::Nearest,
                false,
            ),
            Self::NearestMipmapNearest => (
                wgpu::FilterMode::Nearest,
                wgpu::MipmapFilterMode::Nearest,
                true,
            ),
            Self::LinearMipmapNearest => (
                wgpu::FilterMode::Linear,
                wgpu::MipmapFilterMode::Nearest,
                true,
            ),
            Self::NearestMipmapLinear => (
                wgpu::FilterMode::Nearest,
                wgpu::MipmapFilterMode::Linear,
                true,
            ),
            Self::LinearMipmapLinear => (
                wgpu::FilterMode::Linear,
                wgpu::MipmapFilterMode::Linear,
                true,
            ),
        }
    }
}

/// API-level texture sampling semantics. The variants map one-to-one to core glTF samplers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSampler {
    pub address_mode_u: TextureAddressMode,
    pub address_mode_v: TextureAddressMode,
    pub mag_filter: TextureMagFilter,
    pub min_filter: TextureMinFilter,
}

impl Default for TextureSampler {
    fn default() -> Self {
        Self {
            address_mode_u: TextureAddressMode::Repeat,
            address_mode_v: TextureAddressMode::Repeat,
            mag_filter: TextureMagFilter::Linear,
            min_filter: TextureMinFilter::LinearMipmapLinear,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSamplingConfig {
    pub max_anisotropy: u16,
}

impl Default for TextureSamplingConfig {
    fn default() -> Self {
        Self { max_anisotropy: 1 }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TextureResourceError {
    #[error("texture count {actual} exceeds the bindless capacity {capacity}")]
    TooManyTextures { actual: usize, capacity: u32 },
    #[error("sampler count {actual} exceeds the bindless capacity {capacity}")]
    TooManySamplers { actual: usize, capacity: u32 },
    #[error("texture {texture_index} has invalid dimensions {width}x{height}")]
    InvalidDimensions {
        texture_index: usize,
        width: u32,
        height: u32,
    },
    #[error("texture {texture_index} dimensions {width}x{height} exceed device limit {limit}")]
    DimensionsExceedLimit {
        texture_index: usize,
        width: u32,
        height: u32,
        limit: u32,
    },
    #[error("texture {texture_index} uses unsupported format {format:?}")]
    UnsupportedFormat {
        texture_index: usize,
        format: wgpu::TextureFormat,
    },
    #[error("texture {texture_index} pixel byte count is {actual}, expected {expected}")]
    InvalidPixelCount {
        texture_index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("max_anisotropy must be in 1..=16, got {0}")]
    InvalidAnisotropy(u16),
}

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
        if let Err(error) = self.validate_at(0) {
            panic!("{error}");
        }
    }

    fn validate_at(&self, texture_index: usize) -> Result<(), TextureResourceError> {
        if self.width == 0 || self.height == 0 {
            return Err(TextureResourceError::InvalidDimensions {
                texture_index,
                width: self.width,
                height: self.height,
            });
        }
        if self.format != wgpu::TextureFormat::Rgba8UnormSrgb {
            return Err(TextureResourceError::UnsupportedFormat {
                texture_index,
                format: self.format,
            });
        }
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|value| value.checked_mul(4))
            .map(|value| value as usize)
            .ok_or(TextureResourceError::InvalidDimensions {
                texture_index,
                width: self.width,
                height: self.height,
            })?;
        if self.pixels.len() != expected {
            return Err(TextureResourceError::InvalidPixelCount {
                texture_index,
                actual: self.pixels.len(),
                expected,
            });
        }
        Ok(())
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
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/mesh/mipmap.wgsl").into(),
            ),
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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
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
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

pub(crate) struct UploadedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    #[cfg_attr(not(test), allow(dead_code))]
    pub mip_level_count: u32,
}

/// Shared uploader used by both legacy and meshlet texture arenas.
pub(crate) struct TextureUploader {
    mipmap_generator: MipmapGenerator,
    mipmap_sampler: wgpu::Sampler,
}

impl TextureUploader {
    pub fn new(device: &wgpu::Device) -> Self {
        let mipmap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mipmap.clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 1,
            ..Default::default()
        });
        Self {
            mipmap_generator: MipmapGenerator::new(device, wgpu::TextureFormat::Rgba8UnormSrgb),
            mipmap_sampler,
        }
    }

    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Texture,
        texture_index: usize,
        label: &str,
    ) -> Result<UploadedTexture, TextureResourceError> {
        source.validate_at(texture_index)?;
        let limit = device.limits().max_texture_dimension_2d;
        if source.width > limit || source.height > limit {
            return Err(TextureResourceError::DimensionsExceedLimit {
                texture_index,
                width: source.width,
                height: source.height,
                limit,
            });
        }

        let mip_level_count = Self::mip_level_count_for_size(source.width, source.height);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: source.width,
                height: source.height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: source.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &source.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(source.width * 4),
                rows_per_image: Some(source.height),
            },
            wgpu::Extent3d {
                width: source.width,
                height: source.height,
                depth_or_array_layers: 1,
            },
        );

        if mip_level_count > 1 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("textures.mipmaps.encoder"),
            });
            self.mipmap_generator.encode_generate(
                device,
                &mut encoder,
                &self.mipmap_sampler,
                &texture,
                source.format,
                mip_level_count,
            );
            queue.submit(Some(encoder.finish()));
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("textures.all_mips_view"),
            format: Some(source.format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_level_count),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });
        Ok(UploadedTexture {
            texture,
            view,
            mip_level_count,
        })
    }

    pub(crate) fn mip_level_count_for_size(width: u32, height: u32) -> u32 {
        32 - width.max(height).max(1).leading_zeros()
    }
}

impl TextureSampler {
    pub(crate) fn create_wgpu_sampler(
        self,
        device: &wgpu::Device,
        sampling: TextureSamplingConfig,
        label: &str,
    ) -> Result<wgpu::Sampler, TextureResourceError> {
        if !(1..=16).contains(&sampling.max_anisotropy) {
            return Err(TextureResourceError::InvalidAnisotropy(
                sampling.max_anisotropy,
            ));
        }
        let address = |mode| match mode {
            TextureAddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            TextureAddressMode::Repeat => wgpu::AddressMode::Repeat,
            TextureAddressMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        };
        let mag_filter = match self.mag_filter {
            TextureMagFilter::Nearest => wgpu::FilterMode::Nearest,
            TextureMagFilter::Linear => wgpu::FilterMode::Linear,
        };
        let (min_filter, mipmap_filter, uses_mips) = self.min_filter.wgpu_settings();
        let can_use_anisotropy = mag_filter == wgpu::FilterMode::Linear
            && min_filter == wgpu::FilterMode::Linear
            && mipmap_filter == wgpu::MipmapFilterMode::Linear;
        Ok(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: address(self.address_mode_u),
            address_mode_v: address(self.address_mode_v),
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter,
            min_filter,
            mipmap_filter,
            lod_min_clamp: 0.0,
            lod_max_clamp: if uses_mips { 32.0 } else { 0.0 },
            anisotropy_clamp: if can_use_anisotropy {
                sampling.max_anisotropy
            } else {
                1
            },
            ..Default::default()
        }))
    }
}

pub(crate) struct TextureStorage {
    max_texture_count: u32,
    _textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    samplers: Vec<wgpu::Sampler>,
}

impl TextureStorage {
    pub const DEFAULT_MAX_TEXTURE_COUNT: u32 = 1024;
    pub const DEFAULT_MAX_SAMPLER_COUNT: u32 = 32;

    pub fn from_resources(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &[Texture],
        samplers: &[TextureSampler],
        sampling: TextureSamplingConfig,
    ) -> Result<Self, TextureResourceError> {
        let max_sampler_count = Self::DEFAULT_MAX_SAMPLER_COUNT.min(
            device
                .limits()
                .max_binding_array_sampler_elements_per_shader_stage,
        );

        let fallback_texture;
        let textures = if textures.is_empty() {
            fallback_texture = Texture::white_1x1();
            std::slice::from_ref(&fallback_texture)
        } else {
            textures
        };
        let default_sampler;
        let samplers = if samplers.is_empty() {
            default_sampler = TextureSampler::default();
            std::slice::from_ref(&default_sampler)
        } else {
            samplers
        };
        if samplers.len() as u32 > max_sampler_count {
            return Err(TextureResourceError::TooManySamplers {
                actual: samplers.len(),
                capacity: max_sampler_count,
            });
        }
        if !(1..=16).contains(&sampling.max_anisotropy) {
            return Err(TextureResourceError::InvalidAnisotropy(
                sampling.max_anisotropy,
            ));
        }
        let max_texture_count = Self::DEFAULT_MAX_TEXTURE_COUNT.min(
            device
                .limits()
                .max_binding_array_elements_per_shader_stage
                .saturating_sub(samplers.len() as u32),
        );
        if textures.len() as u32 > max_texture_count {
            return Err(TextureResourceError::TooManyTextures {
                actual: textures.len(),
                capacity: max_texture_count,
            });
        }
        let uploader = TextureUploader::new(device);
        let mut gpu_textures = Vec::with_capacity(textures.len());
        let mut views = Vec::with_capacity(textures.len());
        for (index, texture) in textures.iter().enumerate() {
            let uploaded = uploader.upload(device, queue, texture, index, "textures.texture_2d")?;
            gpu_textures.push(uploaded.texture);
            views.push(uploaded.view);
        }
        let gpu_samplers = samplers
            .iter()
            .copied()
            .enumerate()
            .map(|(index, sampler)| {
                sampler.create_wgpu_sampler(device, sampling, &format!("textures.sampler.{index}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            max_texture_count,
            _textures: gpu_textures,
            views,
            samplers: gpu_samplers,
        })
    }

    pub fn max_texture_count(&self) -> u32 {
        self.max_texture_count
    }

    pub fn max_sampler_count(&self) -> u32 {
        self.samplers.len() as u32
    }

    pub fn texture_views(&self) -> &[wgpu::TextureView] {
        &self.views
    }

    pub fn samplers(&self) -> &[wgpu::Sampler] {
        &self.samplers
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Texture, TextureMinFilter, TextureResourceError, TextureSampler, TextureSamplingConfig,
        TextureStorage, TextureUploader,
    };

    #[test]
    #[should_panic(expected = "invalid dimensions")]
    fn zero_width_is_rejected_before_gpu_resource_creation() {
        Texture {
            width: 0,
            height: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            pixels: Vec::new(),
        }
        .validate();
    }

    #[test]
    fn mip_count_covers_full_non_power_of_two_chain() {
        assert_eq!(TextureUploader::mip_level_count_for_size(1, 1), 1);
        assert_eq!(TextureUploader::mip_level_count_for_size(7, 3), 3);
        assert_eq!(TextureUploader::mip_level_count_for_size(8, 5), 4);
    }

    #[test]
    fn gltf_default_sampler_is_repeat_and_trilinear() {
        let sampler = TextureSampler::default();
        assert_eq!(sampler.min_filter, TextureMinFilter::LinearMipmapLinear);
        assert_eq!(sampler.address_mode_u, super::TextureAddressMode::Repeat);
        assert_eq!(sampler.address_mode_v, super::TextureAddressMode::Repeat);
    }

    #[test]
    fn minification_modes_preserve_filter_and_mipmap_semantics() {
        use wgpu::{FilterMode as F, MipmapFilterMode as M};
        assert_eq!(
            TextureMinFilter::Nearest.wgpu_settings(),
            (F::Nearest, M::Nearest, false)
        );
        assert_eq!(
            TextureMinFilter::Linear.wgpu_settings(),
            (F::Linear, M::Nearest, false)
        );
        assert_eq!(
            TextureMinFilter::NearestMipmapNearest.wgpu_settings(),
            (F::Nearest, M::Nearest, true)
        );
        assert_eq!(
            TextureMinFilter::LinearMipmapNearest.wgpu_settings(),
            (F::Linear, M::Nearest, true)
        );
        assert_eq!(
            TextureMinFilter::NearestMipmapLinear.wgpu_settings(),
            (F::Nearest, M::Linear, true)
        );
        assert_eq!(
            TextureMinFilter::LinearMipmapLinear.wgpu_settings(),
            (F::Linear, M::Linear, true)
        );
    }

    #[test]
    fn uploader_allocates_and_reports_the_complete_mip_chain() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let source = Texture {
            width: 7,
            height: 3,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            pixels: vec![255; 7 * 3 * 4],
        };
        let uploaded = TextureUploader::new(&device)
            .upload(&device, &queue, &source, 0, "mip-test")
            .unwrap();
        assert_eq!(uploaded.mip_level_count, 3);
        assert_eq!(uploaded.texture.mip_level_count(), 3);
    }

    #[test]
    fn resource_table_capacity_errors_are_returned_before_gpu_creation() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 3,
                max_binding_array_sampler_elements_per_shader_stage: 1,
                ..Default::default()
            },
            ..Default::default()
        });
        let textures = [
            Texture::white_1x1(),
            Texture::white_1x1(),
            Texture::white_1x1(),
        ];
        assert!(matches!(
            TextureStorage::from_resources(
                &device,
                &queue,
                &textures,
                &[],
                TextureSamplingConfig::default(),
            ),
            Err(TextureResourceError::TooManyTextures {
                actual: 3,
                capacity: 2,
            })
        ));
        assert!(matches!(
            TextureStorage::from_resources(
                &device,
                &queue,
                &[],
                &[TextureSampler::default(), TextureSampler::default()],
                TextureSamplingConfig::default(),
            ),
            Err(TextureResourceError::TooManySamplers {
                actual: 2,
                capacity: 1,
            })
        ));
    }
}

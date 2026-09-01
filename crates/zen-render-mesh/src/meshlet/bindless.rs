use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::mesh::scene::TextureUploader;
use crate::mesh::{Texture, TextureSampler, TextureSamplingConfig};

/// Stable CPU-side reference to one bindless texture slot.
///
/// Shaders only store [`slot`](Self::slot). The generation protects CPU users from stale handles
/// when a slot is recycled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle {
    pub slot: u32,
    pub generation: u32,
}

/// Stable CPU-side reference to one bindless sampler slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerHandle {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FallbackTextureHandles {
    pub white: TextureHandle,
    pub black: TextureHandle,
    pub flat_normal: TextureHandle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FallbackSamplerHandles {
    pub linear: SamplerHandle,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BindlessTextureError {
    #[error(
        "bindless texture capacity {capacity} is smaller than the three reserved fallback slots"
    )]
    CapacityTooSmall { capacity: u32 },
    #[error("bindless texture arena is full (capacity {capacity})")]
    Full { capacity: u32 },
    #[error("texture handle {handle:?} is stale or invalid")]
    InvalidHandle { handle: TextureHandle },
    #[error("texture dimensions must be non-zero, got {width}x{height}")]
    ZeroExtent { width: u32, height: u32 },
    #[error("texture dimensions {width}x{height} exceed device limit {maximum}")]
    DimensionsExceeded {
        width: u32,
        height: u32,
        maximum: u32,
    },
    #[error("meshlet bindless textures require Rgba8UnormSrgb, got {actual:?}")]
    UnsupportedFormat { actual: wgpu::TextureFormat },
    #[error("texture byte size overflow for {width}x{height} RGBA8 data")]
    SizeOverflow { width: u32, height: u32 },
    #[error("texture has {actual} bytes, expected {expected} bytes of RGBA8 data")]
    PixelSizeMismatch { actual: usize, expected: usize },
    #[error(
        "bindless sampler capacity {capacity} cannot fit {actual} initial samplers plus fallback"
    )]
    InitialSamplerCapacityExceeded { actual: usize, capacity: u32 },
    #[error("failed to create or upload a bindless resource: {message}")]
    Resource { message: String },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BindlessSamplerError {
    #[error("bindless sampler arena is full (capacity {capacity})")]
    Full { capacity: u32 },
    #[error("sampler handle {handle:?} is stale or invalid")]
    InvalidHandle { handle: SamplerHandle },
    #[error("failed to create bindless sampler: {message}")]
    Resource { message: String },
}

#[derive(Debug)]
struct Slot {
    generation: u32,
    texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    reserved: bool,
}

#[derive(Debug)]
struct SamplerSlot {
    generation: u32,
    sampler: Option<wgpu::Sampler>,
    reserved: bool,
}

#[derive(Debug)]
struct BindlessEpoch {
    id: u64,
    bind_group: wgpu::BindGroup,
    last_submission: AtomicU64,
}

/// Fixed-capacity bindless texture/sampler table with frame-boundary bind-group epochs.
///
/// Updates never mutate the bind group currently referenced by recorded work. A new bind group is
/// published by [`begin_frame`](Self::begin_frame); retired epochs are released only after wgpu
/// reports that the submission which last used them has completed.
pub(crate) struct BindlessTextureArena {
    max_textures: u32,
    slots: Vec<Slot>,
    free_slots: Vec<u32>,
    max_samplers: u32,
    sampler_slots: Vec<SamplerSlot>,
    free_sampler_slots: Vec<u32>,
    layout: wgpu::BindGroupLayout,
    current: Arc<BindlessEpoch>,
    retired: Vec<Arc<BindlessEpoch>>,
    dirty: bool,
    next_epoch: u64,
    next_submission: u64,
    completed_submission: Arc<AtomicU64>,
    fallbacks: FallbackTextureHandles,
    fallback_samplers: FallbackSamplerHandles,
    uploader: TextureUploader,
    sampling: TextureSamplingConfig,
}

impl BindlessTextureArena {
    pub(crate) const RESERVED_FALLBACK_COUNT: u32 = 3;

    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_textures: u32,
        max_samplers: u32,
        initial: &[Texture],
        initial_samplers: &[TextureSampler],
        sampling: TextureSamplingConfig,
    ) -> Result<Self, BindlessTextureError> {
        if max_textures < Self::RESERVED_FALLBACK_COUNT {
            return Err(BindlessTextureError::CapacityTooSmall {
                capacity: max_textures,
            });
        }
        let max_samplers = max_samplers.max(1);
        if initial_samplers.len() > max_samplers.saturating_sub(1) as usize {
            return Err(BindlessTextureError::InitialSamplerCapacityExceeded {
                actual: initial_samplers.len(),
                capacity: max_samplers.saturating_sub(1),
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet.bindless.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: std::num::NonZeroU32::new(max_textures),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: std::num::NonZeroU32::new(max_samplers),
                },
            ],
        });

        let sampler = TextureSampler::default()
            .create_wgpu_sampler(device, sampling, "meshlet.bindless.fallback-sampler")
            .map_err(|error| BindlessTextureError::Resource {
                message: error.to_string(),
            })?;
        let uploader = TextureUploader::new(device);

        let mut slots = Vec::with_capacity(max_textures as usize);
        for _ in 0..max_textures {
            slots.push(Slot {
                generation: 1,
                texture: None,
                view: None,
                reserved: false,
            });
        }

        let fallback_sources = [
            Texture::white_1x1(),
            Texture::black_1x1(),
            Texture {
                width: 1,
                height: 1,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                pixels: vec![128, 128, 255, 255],
            },
        ];
        for (slot, source) in fallback_sources.iter().enumerate() {
            validate_texture(device, source)?;
            let uploaded = uploader
                .upload(device, queue, source, slot, "meshlet.bindless.fallback")
                .map_err(|error| BindlessTextureError::Resource {
                    message: error.to_string(),
                })?;
            slots[slot].texture = Some(uploaded.texture);
            slots[slot].view = Some(uploaded.view);
            slots[slot].reserved = true;
        }

        let fallbacks = FallbackTextureHandles {
            white: TextureHandle {
                slot: 0,
                generation: 1,
            },
            black: TextureHandle {
                slot: 1,
                generation: 1,
            },
            flat_normal: TextureHandle {
                slot: 2,
                generation: 1,
            },
        };
        let mut sampler_slots = Vec::with_capacity(max_samplers as usize);
        sampler_slots.push(SamplerSlot {
            generation: 1,
            sampler: Some(sampler),
            reserved: true,
        });
        for _ in 1..max_samplers {
            sampler_slots.push(SamplerSlot {
                generation: 1,
                sampler: None,
                reserved: false,
            });
        }
        for (index, descriptor) in initial_samplers.iter().copied().enumerate() {
            sampler_slots[index + 1].sampler = Some(
                descriptor
                    .create_wgpu_sampler(
                        device,
                        sampling,
                        &format!("meshlet.bindless.sampler.{index}"),
                    )
                    .map_err(|error| BindlessTextureError::Resource {
                        message: error.to_string(),
                    })?,
            );
        }
        let fallback_samplers = FallbackSamplerHandles {
            linear: SamplerHandle {
                slot: 0,
                generation: 1,
            },
        };
        let placeholder = create_bind_group(
            device,
            &layout,
            &slots,
            &sampler_slots,
            fallbacks.white.slot,
            fallback_samplers.linear.slot,
        );
        let mut arena = Self {
            max_textures,
            slots,
            free_slots: (Self::RESERVED_FALLBACK_COUNT..max_textures)
                .rev()
                .collect(),
            max_samplers,
            sampler_slots,
            free_sampler_slots: ((1 + initial_samplers.len() as u32)..max_samplers)
                .rev()
                .collect(),
            layout,
            current: Arc::new(BindlessEpoch {
                id: 0,
                bind_group: placeholder,
                last_submission: AtomicU64::new(0),
            }),
            retired: Vec::new(),
            dirty: false,
            next_epoch: 1,
            next_submission: 1,
            completed_submission: Arc::new(AtomicU64::new(0)),
            fallbacks,
            fallback_samplers,
            uploader,
            sampling,
        };
        for texture in initial {
            arena.insert(device, queue, texture)?;
        }
        arena.rebuild_epoch(device);
        Ok(arena)
    }

    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.current.bind_group
    }

    pub(crate) fn current_epoch_id(&self) -> u64 {
        self.current.id
    }

    pub(crate) fn fallbacks(&self) -> FallbackTextureHandles {
        self.fallbacks
    }

    pub(crate) fn fallback_samplers(&self) -> FallbackSamplerHandles {
        self.fallback_samplers
    }

    pub(crate) fn insert(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Texture,
    ) -> Result<TextureHandle, BindlessTextureError> {
        let slot_index = *self.free_slots.last().ok_or(BindlessTextureError::Full {
            capacity: self.max_textures,
        })?;
        validate_texture(device, source)?;
        let uploaded = self
            .uploader
            .upload(
                device,
                queue,
                source,
                slot_index as usize,
                "meshlet.bindless.texture",
            )
            .map_err(|error| BindlessTextureError::Resource {
                message: error.to_string(),
            })?;
        let removed = self
            .free_slots
            .pop()
            .expect("the previously inspected free slot is still present");
        debug_assert_eq!(removed, slot_index);
        let slot = &mut self.slots[slot_index as usize];
        debug_assert!(slot.texture.is_none());
        slot.texture = Some(uploaded.texture);
        slot.view = Some(uploaded.view);
        self.dirty = true;
        Ok(TextureHandle {
            slot: slot_index,
            generation: slot.generation,
        })
    }

    pub(crate) fn remove(&mut self, handle: TextureHandle) -> Result<(), BindlessTextureError> {
        let Some(slot) = self.slots.get_mut(handle.slot as usize) else {
            return Err(BindlessTextureError::InvalidHandle { handle });
        };
        if slot.reserved || slot.generation != handle.generation || slot.texture.is_none() {
            return Err(BindlessTextureError::InvalidHandle { handle });
        }
        slot.texture = None;
        slot.view = None;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free_slots.push(handle.slot);
        self.dirty = true;
        Ok(())
    }

    pub(crate) fn insert_sampler(
        &mut self,
        device: &wgpu::Device,
        descriptor: TextureSampler,
    ) -> Result<SamplerHandle, BindlessSamplerError> {
        let slot_index = *self
            .free_sampler_slots
            .last()
            .ok_or(BindlessSamplerError::Full {
                capacity: self.max_samplers,
            })?;
        let removed = self
            .free_sampler_slots
            .pop()
            .expect("the previously inspected free sampler slot is still present");
        debug_assert_eq!(removed, slot_index);
        let slot = &mut self.sampler_slots[slot_index as usize];
        debug_assert!(slot.sampler.is_none());
        slot.sampler = Some(
            descriptor
                .create_wgpu_sampler(device, self.sampling, "meshlet.bindless.dynamic-sampler")
                .map_err(|error| BindlessSamplerError::Resource {
                    message: error.to_string(),
                })?,
        );
        self.dirty = true;
        Ok(SamplerHandle {
            slot: slot_index,
            generation: slot.generation,
        })
    }

    pub(crate) fn remove_sampler(
        &mut self,
        handle: SamplerHandle,
    ) -> Result<(), BindlessSamplerError> {
        let Some(slot) = self.sampler_slots.get_mut(handle.slot as usize) else {
            return Err(BindlessSamplerError::InvalidHandle { handle });
        };
        if slot.reserved || slot.generation != handle.generation || slot.sampler.is_none() {
            return Err(BindlessSamplerError::InvalidHandle { handle });
        }
        slot.sampler = None;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free_sampler_slots.push(handle.slot);
        self.dirty = true;
        Ok(())
    }

    /// Publishes pending slot changes and returns the epoch captured by this frame.
    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) -> u64 {
        self.collect_retired();
        if self.dirty {
            self.rebuild_epoch(device);
        }
        self.current.id
    }

    /// Marks the current epoch as referenced by a successfully submitted frame.
    pub(crate) fn after_submit(&mut self, queue: &wgpu::Queue, epoch_id: u64) {
        assert_eq!(
            epoch_id, self.current.id,
            "submitted meshlet frame no longer owns the active bindless epoch"
        );
        let submission = self.next_submission;
        self.next_submission = self.next_submission.wrapping_add(1).max(1);
        self.current
            .last_submission
            .fetch_max(submission, Ordering::Release);
        let completed = Arc::clone(&self.completed_submission);
        queue.on_submitted_work_done(move || {
            completed.fetch_max(submission, Ordering::AcqRel);
        });
    }

    fn rebuild_epoch(&mut self, device: &wgpu::Device) {
        let bind_group = create_bind_group(
            device,
            &self.layout,
            &self.slots,
            &self.sampler_slots,
            self.fallbacks.white.slot,
            self.fallback_samplers.linear.slot,
        );
        let next = Arc::new(BindlessEpoch {
            id: self.next_epoch,
            bind_group,
            last_submission: AtomicU64::new(0),
        });
        self.next_epoch = self.next_epoch.wrapping_add(1).max(1);
        self.retired
            .push(std::mem::replace(&mut self.current, next));
        self.dirty = false;
        self.collect_retired();
    }

    fn collect_retired(&mut self) {
        let completed = self.completed_submission.load(Ordering::Acquire);
        self.retired.retain(|epoch| {
            let last = epoch.last_submission.load(Ordering::Acquire);
            last != 0 && last > completed
        });
    }
}
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    slots: &[Slot],
    sampler_slots: &[SamplerSlot],
    fallback_slot: u32,
    fallback_sampler_slot: u32,
) -> wgpu::BindGroup {
    let fallback = slots[fallback_slot as usize]
        .view
        .as_ref()
        .expect("fallback slot is always resident");
    let highest_resident = slots
        .iter()
        .rposition(|slot| slot.view.is_some())
        .unwrap_or(fallback_slot as usize);
    let views = slots[..=highest_resident]
        .iter()
        .map(|slot| slot.view.as_ref().unwrap_or(fallback))
        .collect::<Vec<_>>();
    let fallback_sampler = sampler_slots[fallback_sampler_slot as usize]
        .sampler
        .as_ref()
        .expect("fallback sampler slot is always resident");
    let highest_resident_sampler = sampler_slots
        .iter()
        .rposition(|slot| slot.sampler.is_some())
        .unwrap_or(fallback_sampler_slot as usize);
    let sampler_refs = sampler_slots[..=highest_resident_sampler]
        .iter()
        .map(|slot| slot.sampler.as_ref().unwrap_or(fallback_sampler))
        .collect::<Vec<_>>();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("meshlet.bindless.epoch"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&views),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::SamplerArray(&sampler_refs),
            },
        ],
    })
}

fn validate_texture(device: &wgpu::Device, source: &Texture) -> Result<(), BindlessTextureError> {
    if source.width == 0 || source.height == 0 {
        return Err(BindlessTextureError::ZeroExtent {
            width: source.width,
            height: source.height,
        });
    }
    let maximum = device.limits().max_texture_dimension_2d;
    if source.width > maximum || source.height > maximum {
        return Err(BindlessTextureError::DimensionsExceeded {
            width: source.width,
            height: source.height,
            maximum,
        });
    }
    if source.format != wgpu::TextureFormat::Rgba8UnormSrgb {
        return Err(BindlessTextureError::UnsupportedFormat {
            actual: source.format,
        });
    }
    let expected = usize::try_from(source.width)
        .ok()
        .and_then(|width| {
            usize::try_from(source.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(BindlessTextureError::SizeOverflow {
            width: source.width,
            height: source.height,
        })?;
    if source.pixels.len() != expected {
        return Err(BindlessTextureError::PixelSizeMismatch {
            actual: source.pixels.len(),
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_generation_cannot_remove_a_recycled_slot() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY,
            required_limits: wgpu::Limits {
                // The stage-wide limit includes both the texture and sampler arrays.
                max_binding_array_elements_per_shader_stage: 9,
                max_binding_array_sampler_elements_per_shader_stage: 1,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut arena = BindlessTextureArena::new(
            &device,
            &queue,
            8,
            1,
            &[],
            &[],
            TextureSamplingConfig::default(),
        )
        .unwrap();
        let old = arena
            .insert(&device, &queue, &Texture::white_1x1())
            .unwrap();
        arena.remove(old).unwrap();
        let replacement = arena
            .insert(&device, &queue, &Texture::black_1x1())
            .unwrap();
        assert_eq!(old.slot, replacement.slot);
        assert_ne!(old.generation, replacement.generation);
        assert_eq!(
            arena.remove(old),
            Err(BindlessTextureError::InvalidHandle { handle: old })
        );
    }

    #[test]
    fn malformed_texture_is_reported_without_panicking() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 5,
                max_binding_array_sampler_elements_per_shader_stage: 1,
                ..Default::default()
            },
            ..Default::default()
        });
        let malformed = Texture {
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            pixels: Vec::new(),
        };
        assert!(matches!(
            BindlessTextureArena::new(
                &device,
                &queue,
                4,
                1,
                &[malformed],
                &[],
                TextureSamplingConfig::default(),
            ),
            Err(BindlessTextureError::PixelSizeMismatch {
                actual: 0,
                expected: 4,
            })
        ));
    }

    #[test]
    fn failed_insert_does_not_consume_a_free_slot() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 5,
                max_binding_array_sampler_elements_per_shader_stage: 1,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut arena = BindlessTextureArena::new(
            &device,
            &queue,
            4,
            1,
            &[],
            &[],
            TextureSamplingConfig::default(),
        )
        .unwrap();
        let malformed = Texture {
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            pixels: Vec::new(),
        };

        assert!(matches!(
            arena.insert(&device, &queue, &malformed),
            Err(BindlessTextureError::PixelSizeMismatch { .. })
        ));
        assert!(arena.insert(&device, &queue, &Texture::white_1x1()).is_ok());
    }

    #[test]
    fn sampler_slots_use_generation_checked_recycling() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY,
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 6,
                max_binding_array_sampler_elements_per_shader_stage: 2,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut arena = BindlessTextureArena::new(
            &device,
            &queue,
            4,
            2,
            &[],
            &[],
            TextureSamplingConfig::default(),
        )
        .unwrap();
        let old = arena
            .insert_sampler(&device, TextureSampler::default())
            .unwrap();
        arena.remove_sampler(old).unwrap();
        let replacement = arena
            .insert_sampler(&device, TextureSampler::default())
            .unwrap();
        assert_eq!(old.slot, replacement.slot);
        assert_ne!(old.generation, replacement.generation);
        assert_eq!(
            arena.remove_sampler(old),
            Err(BindlessSamplerError::InvalidHandle { handle: old })
        );
    }
}

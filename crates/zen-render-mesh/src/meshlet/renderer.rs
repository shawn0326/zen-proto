use std::{
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};

use super::{
    asset::{MeshletAssetError, MeshletSceneAsset},
    bindless::{
        BindlessSamplerError, BindlessTextureArena, BindlessTextureError, FallbackSamplerHandles,
        FallbackTextureHandles, SamplerHandle, TextureHandle,
    },
    capabilities::MeshletDeviceRequirements,
    config::{
        MESHLET_MAX_TRIANGLES, MESHLET_MAX_VERTICES, MeshletBackend, MeshletCapacityConfig,
        MeshletConfigError, MeshletRendererConfig, TASK_MESHLETS_PER_WORKGROUP,
    },
    gpu_scene::{MeshletGpuScene, MeshletGpuSceneUpload},
    gpu_types::{
        BackendWorkCounts, CandidateWork, DispatchIndirectArgs, DrawIndexedIndirectArgs,
        FrameUniform, GpuCounters, GpuLodRecord, GpuMeshRecord, GpuMeshletRecord, GpuVertex,
        InstanceClassification, PSO_BIN_COUNT, RasterUniform, VisibleMeshletWork,
        prefix_scan_block_count,
    },
    graph_recorder::MeshletGraphRecorder,
    passes::MeshletPassSet,
    stats::{MeshletGpuTimingError, MeshletRenderStats},
    stats_readback::MeshletStatsReadback,
};
use crate::{
    Camera, MeshRenderTargets,
    mesh::{
        Instance, Material, MaterialTextureBinding, Texture, TextureSampler, TextureSamplingConfig,
        visibility::HiZStage,
    },
};
use zen_frame_graph::{Frame, FrameGraphError, TextureDesc};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const DEFAULT_LOD_THRESHOLD_PIXELS: f32 = 1.0;
const DEFAULT_LOD_HYSTERESIS: f32 = 0.1;
static NEXT_RENDERER_ID: AtomicU64 = AtomicU64::new(1);

/// Selects how visible meshlets are shaded by every meshlet raster backend.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MeshletRenderMode {
    /// Samples the material textures and applies the normal forward lighting path.
    #[default]
    Shaded = 0,
    /// Draws each global meshlet ID with a stable, unlit debug color.
    MeshletId = 1,
}

/// Per-frame controls shared by all three concrete Vulkan meshlet backends.
#[derive(Clone, Copy, Debug)]
pub struct MeshletRenderInput {
    /// Identity used to join delayed counter readback with an independently completed FrameGraph
    /// timestamp report. Applications should pass the enclosing `RenderFrameInput::frame_index`.
    pub frame_index: u64,
    pub camera: Camera,
    pub enable_occlusion_culling: bool,
    /// Maximum projected geometric error used by GPU LOD selection.
    pub lod_error_threshold_pixels: f32,
    /// One-level LOD hysteresis ratio, clamped by the shader to `0..0.49`.
    pub lod_hysteresis: f32,
    /// Shading mode used by the indexed, mesh-only, and task-mesh raster backends.
    pub render_mode: MeshletRenderMode,
}

impl Default for MeshletRenderInput {
    fn default() -> Self {
        Self {
            frame_index: 0,
            camera: Camera::default(),
            enable_occlusion_culling: true,
            lod_error_threshold_pixels: DEFAULT_LOD_THRESHOLD_PIXELS,
            lod_hysteresis: DEFAULT_LOD_HYSTERESIS,
            render_mode: MeshletRenderMode::Shaded,
        }
    }
}

/// Single-use transaction ticket produced by [`MeshletRenderer::prepare_frame`].
#[must_use = "a prepared meshlet frame must be submitted or discarded"]
#[derive(Debug)]
pub struct PreparedMeshletFrame {
    renderer_id: u64,
    frame_id: u64,
    pub(crate) enable_occlusion_culling: bool,
    pub(crate) extent: wgpu::Extent3d,
    pub(crate) bindless_epoch: u64,
    pub(crate) readback_index: Option<usize>,
    pub(crate) stats_frame_index: u64,
}

impl PreparedMeshletFrame {
    #[must_use]
    pub const fn extent(&self) -> wgpu::Extent3d {
        self.extent
    }
}

/// Independent Vulkan-only meshlet renderer.
///
/// `MeshRenderer` remains untouched as the legacy indexed-indirect baseline. This renderer owns a
/// separate GPU Scene V2 and chooses exactly one concrete raster backend at construction time.
pub struct MeshletRenderer {
    renderer_id: u64,
    next_frame_id: u64,
    prepared_frame: Option<(u64, u64)>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    color_format: wgpu::TextureFormat,
    backend: MeshletBackend,
    scene: MeshletGpuScene,
    bindless: BindlessTextureArena,
    bindless_texture_count: u32,
    passes: MeshletPassSet,
    hiz_stage: HiZStage,
    stats: MeshletStatsReadback,
    max_dispatch_dimension: u32,
}

impl MeshletRenderer {
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor makes the immutable asset, scene, resource table, and device tier explicit"
    )]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        config: MeshletRendererConfig,
        requirements: &MeshletDeviceRequirements,
        asset: &MeshletSceneAsset,
        materials: &[Material],
        instances: &[Instance],
        textures: &[Texture],
        samplers: &[TextureSampler],
        sampling: TextureSamplingConfig,
    ) -> Result<Self, MeshletRendererError> {
        config.validate()?;
        asset.validate()?;
        validate_requirements(device, config, requirements)?;
        validate_asset_shader_contract(asset)?;

        if instances.len() > config.capacities.max_instances as usize {
            return Err(MeshletRendererError::InstanceCapacityExceeded {
                actual: instances.len(),
                capacity: config.capacities.max_instances,
            });
        }
        let bindless_capacity = requirements.bindless_capacity();
        let available_texture_slots = bindless_capacity
            .textures
            .saturating_sub(BindlessTextureArena::RESERVED_FALLBACK_COUNT);
        if textures.len() > available_texture_slots as usize {
            return Err(MeshletRendererError::TextureCapacityExceeded {
                actual: textures.len(),
                capacity: available_texture_slots,
            });
        }
        let available_sampler_slots = bindless_capacity.samplers.saturating_sub(1);
        if samplers.len() > available_sampler_slots as usize {
            return Err(MeshletRendererError::SamplerCapacityExceeded {
                actual: samplers.len(),
                capacity: available_sampler_slots,
            });
        }
        validate_material_resources(materials, textures.len(), samplers.len())?;

        validate_gpu_buffer_limits(
            device,
            asset,
            materials.len().max(1),
            instances.len(),
            config.capacities,
        )?;
        let max_dispatch_dimension = backend_dispatch_limit(device, requirements.backend());

        let bindless = BindlessTextureArena::new(
            device,
            queue,
            bindless_capacity.textures,
            bindless_capacity.samplers,
            textures,
            samplers,
            sampling,
        )?;
        let upload = build_gpu_upload(
            asset,
            materials,
            instances,
            textures.len(),
            samplers.len(),
            &bindless,
        )?;
        let scene = MeshletGpuScene::new(device, upload, config.capacities);
        let passes = MeshletPassSet::new(
            device,
            requirements.backend(),
            color_format,
            DEPTH_FORMAT,
            bindless.layout(),
            max_dispatch_dimension,
        );
        let instance_groups = u64::from(scene.instance_count.div_ceil(64));
        let maximum_groups = u64::from(max_dispatch_dimension) * u64::from(max_dispatch_dimension);
        if instance_groups > maximum_groups {
            return Err(MeshletRendererError::InstanceDispatchExceeded {
                groups: instance_groups,
                max_dimension: max_dispatch_dimension,
            });
        }
        let total_instances = scene.instance_count;

        Ok(Self {
            renderer_id: NEXT_RENDERER_ID.fetch_add(1, Ordering::Relaxed).max(1),
            next_frame_id: 1,
            prepared_frame: None,
            device: device.clone(),
            queue: queue.clone(),
            color_format,
            backend: requirements.backend(),
            scene,
            bindless,
            bindless_texture_count: bindless_capacity.textures,
            passes,
            hiz_stage: HiZStage::new(device),
            stats: MeshletStatsReadback::new(device, total_instances, requirements.backend()),
            max_dispatch_dimension,
        })
    }

    #[must_use]
    pub const fn backend(&self) -> MeshletBackend {
        self.backend
    }

    #[must_use]
    pub fn fallback_texture_handles(&self) -> FallbackTextureHandles {
        self.bindless.fallbacks()
    }

    #[must_use]
    pub fn fallback_sampler_handles(&self) -> FallbackSamplerHandles {
        self.bindless.fallback_samplers()
    }

    /// Queues a texture-table change. A new bind-group epoch is published at the next frame
    /// boundary; already prepared frames retain their old epoch.
    pub fn insert_texture(
        &mut self,
        texture: &Texture,
    ) -> Result<TextureHandle, BindlessTextureError> {
        self.bindless.insert(&self.device, &self.queue, texture)
    }

    pub fn remove_texture(&mut self, handle: TextureHandle) -> Result<(), BindlessTextureError> {
        self.bindless.remove(handle)
    }

    /// Queues a sampler-table change for publication at the next frame boundary.
    pub fn insert_sampler(
        &mut self,
        sampler: TextureSampler,
    ) -> Result<SamplerHandle, BindlessSamplerError> {
        self.bindless.insert_sampler(&self.device, sampler)
    }

    pub fn remove_sampler(&mut self, handle: SamplerHandle) -> Result<(), BindlessSamplerError> {
        self.bindless.remove_sampler(handle)
    }

    pub fn request_stats(&mut self) {
        self.stats.request();
    }

    /// Returns whether the next frame can reserve a counter readback immediately.
    ///
    /// This is useful when a counter copy must be paired atomically with another one-shot request,
    /// such as a FrameGraph timestamp capture.
    #[must_use]
    pub fn can_request_stats_immediately(&self) -> bool {
        self.stats.can_request_immediately()
    }

    /// Attempts to reserve a counter copy for the next frame without queueing behind a busy ring.
    #[must_use]
    pub fn try_request_stats(&mut self) -> bool {
        self.stats.try_request()
    }

    /// Requests counters whose delivery waits for an explicitly associated GPU timing report from
    /// the same frame. This avoids returning a default timing block merely because counter mapping
    /// completed before timestamp mapping. Returns `false` instead of queueing when the three-slot
    /// readback ring is busy, so a caller never captures timing for a different future frame.
    #[must_use]
    pub fn request_stats_with_gpu_timing(&mut self) -> bool {
        self.stats.try_request_with_gpu_timing()
    }

    pub fn take_stats(&mut self, _device: &wgpu::Device) -> Option<MeshletRenderStats> {
        self.stats.take_ready(&self.device)
    }

    /// Explicitly associates a FrameGraph timestamp report with a requested counter snapshot.
    ///
    /// Arrival order is intentionally ignored; the report is keyed by the frame identity supplied
    /// through [`MeshletRenderInput::frame_index`].
    pub fn associate_gpu_timing(
        &mut self,
        report: &zen_frame_graph::GpuTimingReport,
    ) -> Result<(), MeshletGpuTimingError> {
        self.stats.associate_gpu_timing(report)
    }

    pub fn prepare_frame(
        &mut self,
        _queue: &wgpu::Queue,
        input: MeshletRenderInput,
        extent: wgpu::Extent3d,
    ) -> PreparedMeshletFrame {
        assert!(
            self.prepared_frame.is_none(),
            "MeshletRenderer supports one prepared frame at a time; call after_submit or after_discard first"
        );
        let width = extent.width.max(1);
        let height = extent.height.max(1);
        let pyramid = crate::mesh::visibility::HiZPyramidDesc::new(width, height);
        let bindless_epoch = self.bindless.begin_frame(&self.device);

        let frame = self.frame_uniform(input, width, height, pyramid.mip_level_count());
        let mut coarse_frame = frame;
        coarse_frame.parameters[3] = 0.0;
        coarse_frame.hiz_mip_count = 0;
        let raster = self.raster_uniforms(input.camera, input.render_mode);
        self.scene
            .update_uniforms(&self.queue, &frame, &coarse_frame, &raster);

        let frame_id = self.next_frame_id;
        self.next_frame_id = self.next_frame_id.wrapping_add(1).max(1);
        self.prepared_frame = Some((frame_id, bindless_epoch));

        PreparedMeshletFrame {
            renderer_id: self.renderer_id,
            frame_id,
            enable_occlusion_culling: input.enable_occlusion_culling,
            extent: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            bindless_epoch,
            readback_index: self.stats.planned_buffer_index(),
            stats_frame_index: input.frame_index,
        }
    }

    pub fn record_frame_graph<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        prepared: &PreparedMeshletFrame,
    ) -> Result<(), FrameGraphError> {
        self.validate_prepared(prepared)
            .map_err(|error| FrameGraphError::Internal {
                message: error.to_string(),
            })?;
        self.validate_render_targets(frame, targets, prepared)?;
        MeshletGraphRecorder::new(self).record(frame, targets, prepared)
    }

    pub fn after_submit(&mut self, _device: &wgpu::Device, prepared: PreparedMeshletFrame) {
        self.assert_prepared(&prepared);
        if let Some(index) = prepared.readback_index {
            self.stats
                .commit_submitted(index, prepared.stats_frame_index);
        }
        self.stats.after_submit(&self.device);
        self.bindless
            .after_submit(&self.queue, prepared.bindless_epoch);
        self.prepared_frame = None;
    }

    /// Ends a prepared frame which never reached queue submission.
    pub fn after_discard(&mut self, prepared: PreparedMeshletFrame) {
        self.assert_prepared(&prepared);
        self.prepared_frame = None;
    }

    fn validate_prepared(
        &self,
        prepared: &PreparedMeshletFrame,
    ) -> Result<(), MeshletRendererError> {
        if prepared.renderer_id != self.renderer_id {
            return Err(MeshletRendererError::ForeignPreparedFrame);
        }
        if self.prepared_frame != Some((prepared.frame_id, prepared.bindless_epoch))
            || self.bindless.current_epoch_id() != prepared.bindless_epoch
        {
            return Err(MeshletRendererError::StalePreparedFrame);
        }
        Ok(())
    }

    fn assert_prepared(&self, prepared: &PreparedMeshletFrame) {
        self.validate_prepared(prepared)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    fn validate_render_targets<'frame>(
        &self,
        frame: &Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        prepared: &PreparedMeshletFrame,
    ) -> Result<(), FrameGraphError> {
        let color = frame.texture_desc(targets.color)?;
        validate_render_target("color", color, prepared.extent, self.color_format)?;
        let depth = frame.texture_desc(targets.depth)?;
        validate_render_target("depth", depth, prepared.extent, DEPTH_FORMAT)
    }

    fn frame_uniform(
        &self,
        input: MeshletRenderInput,
        width: u32,
        height: u32,
        hiz_mip_count: u32,
    ) -> FrameUniform {
        let camera = input.camera;
        let frustum = camera.frustum().map(|plane| plane.to_array());
        let camera_position = camera.transform().w_axis.to_array();
        FrameUniform {
            view_projection: camera.view_projection().to_cols_array_2d(),
            view: camera.view().to_cols_array_2d(),
            frustum_planes: frustum,
            camera_position,
            viewport: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            parameters: [
                input.lod_error_threshold_pixels.max(0.01),
                input.lod_hysteresis.clamp(0.0, 0.49),
                1.0e-4,
                u32::from(input.enable_occlusion_culling) as f32,
            ],
            counts: [
                self.scene.instance_count,
                self.scene.mesh_count,
                self.scene.capacities.max_candidate_meshlets,
                self.scene.capacities.max_indirect_draws_per_bin,
            ],
            hiz_mip_count: if input.enable_occlusion_culling {
                hiz_mip_count
            } else {
                0
            },
            max_dispatch_dimension: self.max_dispatch_dimension,
            perspective_projection: u32::from(is_perspective_projection(camera.projection())),
            _pad: 0,
        }
    }

    fn raster_uniforms(
        &self,
        camera: Camera,
        render_mode: MeshletRenderMode,
    ) -> [RasterUniform; 2] {
        let view_projection = camera.view_projection().to_cols_array_2d();
        std::array::from_fn(|bin| RasterUniform {
            view_projection,
            visible_base: bin as u32 * self.scene.capacities.max_indirect_draws_per_bin,
            render_mode: render_mode as u32,
            pso_bin: bin as u32,
            _pad: 0,
        })
    }

    pub(crate) const fn scene(&self) -> &MeshletGpuScene {
        &self.scene
    }

    pub(crate) const fn passes(&self) -> &MeshletPassSet {
        &self.passes
    }

    pub(crate) const fn hiz_stage(&self) -> &HiZStage {
        &self.hiz_stage
    }

    pub(crate) const fn bindless(&self) -> &BindlessTextureArena {
        &self.bindless
    }

    pub(crate) const fn bindless_texture_count(&self) -> u32 {
        self.bindless_texture_count
    }

    pub(crate) fn readback_buffer(&self, index: usize) -> &wgpu::Buffer {
        self.stats.staging_buffer(index)
    }
}

fn is_perspective_projection(projection: glam::Mat4) -> bool {
    projection.is_finite()
        && projection.w_axis.w.abs() <= 1.0e-6
        && projection.z_axis.w.abs() > 1.0e-6
}

fn validate_requirements(
    device: &wgpu::Device,
    config: MeshletRendererConfig,
    requirements: &MeshletDeviceRequirements,
) -> Result<(), MeshletRendererError> {
    if requirements.adapter_backend() != wgpu::Backend::Vulkan {
        return Err(MeshletRendererError::UnsupportedWgpuBackend {
            actual: requirements.adapter_backend(),
        });
    }
    if requirements.source_config() != config {
        return Err(MeshletRendererError::RequirementsConfigMismatch);
    }
    if !requirements.backend().is_resolved() {
        return Err(MeshletRendererError::UnresolvedBackend);
    }
    if config.backend.is_resolved() && config.backend != requirements.backend() {
        return Err(MeshletRendererError::BackendMismatch {
            configured: config.backend,
            requested: requirements.backend(),
        });
    }
    let missing = requirements.features() - device.features();
    if !missing.is_empty() {
        return Err(MeshletRendererError::DeviceMissingFeatures { missing });
    }
    if !requirements.limits().check_limits(&device.limits()) {
        return Err(MeshletRendererError::DeviceLimitsMismatch);
    }
    Ok(())
}

fn validate_render_target(
    target: &'static str,
    actual: &TextureDesc,
    expected_extent: wgpu::Extent3d,
    expected_format: wgpu::TextureFormat,
) -> Result<(), FrameGraphError> {
    if actual.size != expected_extent
        || actual.sample_count != 1
        || actual.format != expected_format
    {
        return Err(FrameGraphError::InvalidResourceDescriptor {
            message: format!(
                "meshlet {target} target must be {:?}, 1 sample, {expected_format:?}; got {:?}, {} samples, {:?}",
                expected_extent, actual.size, actual.sample_count, actual.format
            ),
        });
    }
    Ok(())
}

fn validate_asset_shader_contract(asset: &MeshletSceneAsset) -> Result<(), MeshletRendererError> {
    let build = asset.config();
    if build.max_meshlet_vertices != MESHLET_MAX_VERTICES
        || build.max_meshlet_triangles != MESHLET_MAX_TRIANGLES
        || build.task_workgroup_meshlets != TASK_MESHLETS_PER_WORKGROUP
    {
        return Err(MeshletRendererError::UnsupportedAssetBuild {
            vertices: build.max_meshlet_vertices,
            triangles: build.max_meshlet_triangles,
            task_workgroup_meshlets: build.task_workgroup_meshlets,
        });
    }
    Ok(())
}

fn build_gpu_upload(
    asset: &MeshletSceneAsset,
    source_materials: &[Material],
    source_instances: &[Instance],
    texture_count: usize,
    sampler_count: usize,
    bindless: &BindlessTextureArena,
) -> Result<MeshletGpuSceneUpload, MeshletRendererError> {
    let vertices = asset
        .positions()
        .iter()
        .zip(asset.attributes())
        .map(|(position, attributes)| GpuVertex {
            position: *position,
            normal_oct: attributes.normal_oct,
            uv: attributes.uv,
            color: attributes.color_rgba8,
            _pad: 0,
        })
        .collect();
    let meshes = asset
        .meshes()
        .iter()
        .map(|mesh| GpuMeshRecord {
            first_lod: mesh.first_lod,
            lod_count: mesh.lod_count,
            _pad: [0; 2],
            sphere: [
                mesh.bounds.center[0],
                mesh.bounds.center[1],
                mesh.bounds.center[2],
                conservative_gpu_radius(mesh.bounds.radius),
            ],
        })
        .collect();
    let lods = asset
        .lods()
        .iter()
        .map(|lod| GpuLodRecord {
            first_meshlet: lod.first_meshlet,
            meshlet_count: lod.meshlet_count,
            geometric_error: lod.geometric_error,
            _pad: 0,
            sphere: [
                lod.bounds.center[0],
                lod.bounds.center[1],
                lod.bounds.center[2],
                conservative_gpu_radius(lod.bounds.radius),
            ],
        })
        .collect();
    let meshlets = asset
        .meshlets()
        .iter()
        .map(|meshlet| GpuMeshletRecord {
            vertex_offset: meshlet.vertex_offset,
            vertex_count: meshlet.vertex_count,
            triangle_offset: meshlet.triangle_offset,
            triangle_count: meshlet.triangle_count,
            fallback_first_index: meshlet.fallback_first_index,
            fallback_index_count: meshlet.fallback_index_count,
            _pad: [0; 2],
            sphere: [
                meshlet.bounds.center[0],
                meshlet.bounds.center[1],
                meshlet.bounds.center[2],
                conservative_gpu_radius(meshlet.bounds.radius),
            ],
            cone: [
                meshlet.normal_cone.axis[0],
                meshlet.normal_cone.axis[1],
                meshlet.normal_cone.axis[2],
                conservative_gpu_cone_cutoff(meshlet.normal_cone.cutoff),
            ],
        })
        .collect::<Vec<_>>();

    let meshlet_vertices = asset.meshlet_vertex_refs().to_vec();
    let micro_indices = asset
        .micro_indices()
        .iter()
        .map(|&index| u32::from(index))
        .collect::<Vec<_>>();

    let fallbacks = bindless.fallbacks();
    let fallback_sampler = bindless.fallback_samplers().linear;
    let remap_texture = |source: u32, fallback: TextureHandle| {
        if texture_count == 0 {
            fallback.slot
        } else {
            BindlessTextureArena::RESERVED_FALLBACK_COUNT + source
        }
    };
    let remap_sampler = |source: u32| {
        if sampler_count == 0 {
            fallback_sampler.slot
        } else {
            1 + source
        }
    };
    let mut materials = if source_materials.is_empty() {
        vec![Material {
            albedo_factor: glam::Vec4::ONE,
            emissive_ao: glam::Vec4::W,
            albedo: MaterialTextureBinding {
                texture_id: fallbacks.white.slot,
                sampler_id: fallback_sampler.slot,
            },
            emissive: MaterialTextureBinding {
                texture_id: fallbacks.black.slot,
                sampler_id: fallback_sampler.slot,
            },
            occlusion: MaterialTextureBinding {
                texture_id: fallbacks.white.slot,
                sampler_id: fallback_sampler.slot,
            },
            _padding: [0; 2],
        }]
    } else {
        source_materials.to_vec()
    };
    for material in &mut materials {
        material.albedo.texture_id = remap_texture(material.albedo.texture_id, fallbacks.white);
        material.albedo.sampler_id = remap_sampler(material.albedo.sampler_id);
        material.emissive.texture_id = remap_texture(material.emissive.texture_id, fallbacks.black);
        material.emissive.sampler_id = remap_sampler(material.emissive.sampler_id);
        material.occlusion.texture_id =
            remap_texture(material.occlusion.texture_id, fallbacks.white);
        material.occlusion.sampler_id = remap_sampler(material.occlusion.sampler_id);
    }

    let mut instances = Vec::with_capacity(source_instances.len());
    for (index, source) in source_instances.iter().enumerate() {
        if !is_finite_affine(source.transform) {
            return Err(MeshletRendererError::InvalidInstanceTransform { instance: index });
        }
        let mesh = asset.meshes().get(source.mesh_id as usize).ok_or(
            MeshletRendererError::InvalidInstanceMesh {
                instance: index,
                mesh_id: source.mesh_id,
                mesh_count: asset.meshes().len(),
            },
        )?;
        if source.material_id as usize >= materials.len() {
            return Err(MeshletRendererError::InvalidInstanceMaterial {
                instance: index,
                material_id: source.material_id,
                material_count: materials.len(),
            });
        }
        let transform = canonicalize_affine(source.transform);
        let pso = instance_pso_class(mesh.pso_class()?, transform);
        let mut instance = *source;
        instance.transform = transform;
        instance._pad = [pso as u32, 0];
        instances.push(instance);
    }

    Ok(MeshletGpuSceneUpload {
        vertices,
        meshes,
        lods,
        meshlets,
        meshlet_vertices,
        micro_indices,
        fallback_indices: asset.fallback_indices().to_vec(),
        instances,
        materials,
    })
}

fn validate_material_resources(
    materials: &[Material],
    texture_count: usize,
    sampler_count: usize,
) -> Result<(), MeshletRendererError> {
    let effective_texture_count = texture_count.max(1);
    let effective_sampler_count = sampler_count.max(1);
    for (material_index, material) in materials.iter().enumerate() {
        for (slot, binding) in [
            ("albedo", material.albedo),
            ("emissive", material.emissive),
            ("occlusion", material.occlusion),
        ] {
            if binding.texture_id as usize >= effective_texture_count {
                return Err(MeshletRendererError::InvalidMaterialTexture {
                    material: material_index,
                    slot,
                    texture_id: binding.texture_id,
                    texture_count: effective_texture_count,
                });
            }
            if binding.sampler_id as usize >= effective_sampler_count {
                return Err(MeshletRendererError::InvalidMaterialSampler {
                    material: material_index,
                    slot,
                    sampler_id: binding.sampler_id,
                    sampler_count: effective_sampler_count,
                });
            }
        }
    }
    Ok(())
}

fn is_finite_affine(transform: glam::Mat4) -> bool {
    transform.is_finite()
        && transform.x_axis.w.abs() <= 1.0e-6
        && transform.y_axis.w.abs() <= 1.0e-6
        && transform.z_axis.w.abs() <= 1.0e-6
        && (transform.w_axis.w - 1.0).abs() <= 1.0e-6
}

fn canonicalize_affine(mut transform: glam::Mat4) -> glam::Mat4 {
    transform.x_axis.w = 0.0;
    transform.y_axis.w = 0.0;
    transform.z_axis.w = 0.0;
    transform.w_axis.w = 1.0;
    transform
}

fn conservative_gpu_radius(radius: f32) -> f32 {
    // Asset validation intentionally tolerates small f32 round-off. Compensate in the actual GPU
    // ABI so a tolerated serialized bound can never become an under-bound at a culling boundary.
    radius + radius.max(1.0) * 2.0e-4
}

fn conservative_gpu_cone_cutoff(cutoff: f32) -> f32 {
    if cutoff > 1.0 {
        cutoff
    } else {
        // A larger cutoff makes the normal-cone rejection condition harder to satisfy. Values
        // above one deliberately disable cone culling in the shaders.
        cutoff + 2.0e-4
    }
}

fn instance_pso_class(
    mesh_class: super::asset::MeshletPsoClass,
    transform: glam::Mat4,
) -> super::asset::MeshletPsoClass {
    let determinant = transform.determinant();
    if mesh_class == super::asset::MeshletPsoClass::OpaqueBackface
        && (!determinant.is_finite() || determinant <= 0.0)
    {
        // Both raster pipelines use a fixed CCW front face. Mirrored, singular, or malformed
        // instance transforms cannot safely use back-face culling without a per-instance winding
        // variant, so route them through the conservative two-sided bin.
        super::asset::MeshletPsoClass::OpaqueTwoSided
    } else {
        mesh_class
    }
}

fn validate_gpu_buffer_limits(
    device: &wgpu::Device,
    asset: &MeshletSceneAsset,
    material_count: usize,
    instance_count: usize,
    capacities: MeshletCapacityConfig,
) -> Result<(), MeshletRendererError> {
    let two_bins = u64::from(PSO_BIN_COUNT);
    let storage_buffers = [
        (
            "vertices",
            element_bytes::<GpuVertex>(asset.positions().len()),
        ),
        (
            "meshes",
            element_bytes::<GpuMeshRecord>(asset.meshes().len()),
        ),
        ("lods", element_bytes::<GpuLodRecord>(asset.lods().len())),
        (
            "meshlets",
            element_bytes::<GpuMeshletRecord>(asset.meshlets().len()),
        ),
        (
            "meshlet vertex references",
            element_bytes::<u32>(asset.meshlet_vertex_refs().len()),
        ),
        (
            "expanded micro-indices",
            element_bytes::<u32>(asset.micro_indices().len()),
        ),
        (
            "fallback indices",
            element_bytes::<u32>(asset.fallback_indices().len()),
        ),
        ("instances", element_bytes::<Instance>(instance_count)),
        ("materials", element_bytes::<Material>(material_count)),
        (
            "instance classifications",
            count_bytes::<InstanceClassification>(u64::from(capacities.max_instances)),
        ),
        (
            "prefix scan block sums",
            count_bytes::<u32>(u64::from(prefix_scan_block_count(capacities.max_instances))),
        ),
        (
            "LOD history",
            count_bytes::<u32>(u64::from(capacities.max_instances)),
        ),
        (
            "candidate work",
            count_bytes::<CandidateWork>(u64::from(capacities.max_candidate_meshlets)),
        ),
        (
            "visible work",
            count_bytes::<VisibleMeshletWork>(u64::from(capacities.max_visible_meshlets)),
        ),
        (
            "indexed indirect arguments",
            count_bytes::<DrawIndexedIndirectArgs>(
                u64::from(capacities.max_indirect_draws_per_bin) * two_bins,
            ),
        ),
        ("counters", count_bytes::<GpuCounters>(1)),
        ("backend work counts", count_bytes::<BackendWorkCounts>(1)),
        (
            "mesh dispatch arguments",
            count_bytes::<DispatchIndirectArgs>(two_bins),
        ),
        (
            "task dispatch arguments",
            count_bytes::<DispatchIndirectArgs>(two_bins),
        ),
        (
            "candidate dispatch arguments",
            count_bytes::<DispatchIndirectArgs>(1),
        ),
    ];
    let limits = device.limits();
    for (buffer, required) in storage_buffers {
        let required = required?;
        if required > limits.max_buffer_size || required > limits.max_storage_buffer_binding_size {
            return Err(MeshletRendererError::BufferLimitExceeded {
                buffer,
                required,
                max_buffer_size: limits.max_buffer_size,
                max_storage_binding_size: limits.max_storage_buffer_binding_size,
            });
        }
    }
    Ok(())
}

fn element_bytes<T>(count: usize) -> Result<u64, MeshletRendererError> {
    let count = u64::try_from(count).map_err(|_| MeshletRendererError::BufferSizeOverflow)?;
    count_bytes::<T>(count)
}

fn count_bytes<T>(count: u64) -> Result<u64, MeshletRendererError> {
    count
        .checked_mul(size_of::<T>() as u64)
        .map(|size| size.max(4))
        .ok_or(MeshletRendererError::BufferSizeOverflow)
}

fn backend_dispatch_limit(device: &wgpu::Device, backend: MeshletBackend) -> u32 {
    let limits = device.limits();
    let mut maximum = limits.max_compute_workgroups_per_dimension.max(1);
    if backend.uses_mesh_shaders() {
        maximum = maximum.min(limits.max_mesh_workgroups_per_dimension.max(1));
    }
    if backend.uses_task_shaders() {
        maximum = maximum.min(limits.max_task_workgroups_per_dimension.max(1));
    }
    maximum
}

#[derive(Debug, thiserror::Error)]
pub enum MeshletRendererError {
    #[error(transparent)]
    InvalidConfig(#[from] MeshletConfigError),
    #[error(transparent)]
    InvalidAsset(#[from] MeshletAssetError),
    #[error(transparent)]
    Bindless(#[from] BindlessTextureError),
    #[error("MeshletRenderer currently supports only Vulkan; requirements came from {actual}")]
    UnsupportedWgpuBackend { actual: wgpu::Backend },
    #[error("MeshletDeviceRequirements were resolved from a different renderer configuration")]
    RequirementsConfigMismatch,
    #[error("MeshletBackend::Auto must be resolved before constructing MeshletRenderer")]
    UnresolvedBackend,
    #[error("configured meshlet backend {configured} does not match device request {requested}")]
    BackendMismatch {
        configured: MeshletBackend,
        requested: MeshletBackend,
    },
    #[error("the created device is missing required meshlet features: {missing:?}")]
    DeviceMissingFeatures { missing: wgpu::Features },
    #[error("the created device limits do not satisfy MeshletDeviceRequirements")]
    DeviceLimitsMismatch,
    #[error("prepared meshlet frame belongs to another renderer")]
    ForeignPreparedFrame,
    #[error("prepared meshlet frame is stale or no longer active")]
    StalePreparedFrame,
    #[error("meshlet GPU buffer byte-size calculation overflowed")]
    BufferSizeOverflow,
    #[error(
        "meshlet {buffer} buffer needs {required} bytes, but device limits are max_buffer_size={max_buffer_size} and max_storage_buffer_binding_size={max_storage_binding_size}"
    )]
    BufferLimitExceeded {
        buffer: &'static str,
        required: u64,
        max_buffer_size: u64,
        max_storage_binding_size: u64,
    },
    #[error(
        "asset was built for {vertices} vertices/{triangles} triangles/{task_workgroup_meshlets} task-workgroup meshlets; this renderer requires 64/64/32"
    )]
    UnsupportedAssetBuild {
        vertices: u32,
        triangles: u32,
        task_workgroup_meshlets: u32,
    },
    #[error("scene has {actual} instances but renderer capacity is {capacity}")]
    InstanceCapacityExceeded { actual: usize, capacity: u32 },
    #[error("scene has {actual} textures but bindless user capacity is {capacity}")]
    TextureCapacityExceeded { actual: usize, capacity: u32 },
    #[error("scene has {actual} samplers but bindless user capacity is {capacity}")]
    SamplerCapacityExceeded { actual: usize, capacity: u32 },
    #[error(
        "material {material} {slot} texture index {texture_id} is out of range for {texture_count} textures"
    )]
    InvalidMaterialTexture {
        material: usize,
        slot: &'static str,
        texture_id: u32,
        texture_count: usize,
    },
    #[error(
        "material {material} {slot} sampler index {sampler_id} is out of range for {sampler_count} samplers"
    )]
    InvalidMaterialSampler {
        material: usize,
        slot: &'static str,
        sampler_id: u32,
        sampler_count: usize,
    },
    #[error(
        "instance dispatch needs {groups} workgroups, exceeding a {max_dimension}x{max_dimension} dispatch"
    )]
    InstanceDispatchExceeded { groups: u64, max_dimension: u32 },
    #[error("instance {instance} references mesh {mesh_id}, but the asset has {mesh_count} meshes")]
    InvalidInstanceMesh {
        instance: usize,
        mesh_id: u32,
        mesh_count: usize,
    },
    #[error(
        "instance {instance} references material {material_id}, but the renderer has {material_count} materials"
    )]
    InvalidInstanceMaterial {
        instance: usize,
        material_id: u32,
        material_count: usize,
    },
    #[error("instance {instance} transform must be a finite affine matrix")]
    InvalidInstanceTransform { instance: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet::{
        MeshletBindlessConfig, MeshletCapabilities, MeshletCapacityConfig, RawStaticMesh,
    };
    use zen_frame_graph::{
        AccessRole, CompileOptions, FrameGraph, FullCompilationReport, ResourceKind, RootReason,
        TextureDesc, UsagePolicy,
    };

    #[test]
    fn instance_transform_contract_rejects_projective_and_non_finite_matrices() {
        assert!(is_finite_affine(glam::Mat4::IDENTITY));
        assert!(is_finite_affine(
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(-1.0, 2.0, 0.5),
                glam::Quat::from_rotation_y(0.7),
                glam::Vec3::new(3.0, -2.0, 1.0),
            )
        ));
        assert!(!is_finite_affine(glam::Mat4::perspective_rh(
            1.0, 1.0, 0.1, 100.0
        )));
        let mut non_finite = glam::Mat4::IDENTITY;
        non_finite.x_axis.x = f32::NAN;
        assert!(!is_finite_affine(non_finite));

        let mut tolerated = glam::Mat4::IDENTITY;
        tolerated.x_axis.w = 5.0e-7;
        tolerated.w_axis.w = 1.0 - 5.0e-7;
        assert!(is_finite_affine(tolerated));
        let canonical = canonicalize_affine(tolerated);
        assert_eq!(canonical.x_axis.w, 0.0);
        assert_eq!(canonical.w_axis.w, 1.0);
    }

    fn indexed_renderer_for(
        asset: &MeshletSceneAsset,
        instances: &[Instance],
    ) -> (wgpu::Device, wgpu::Queue, MeshletRenderer) {
        let features =
            MeshletDeviceRequirements::required_features(MeshletBackend::IndexedIndirect).unwrap();
        let limits = wgpu::Limits {
            max_binding_array_elements_per_shader_stage: 5,
            max_binding_array_sampler_elements_per_shader_stage: 1,
            max_storage_buffers_per_shader_stage: 8,
            ..Default::default()
        };
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            label: Some("meshlet.renderer.test-device"),
            required_features: features,
            required_limits: limits.clone(),
            ..Default::default()
        });
        let config = MeshletRendererConfig {
            backend: MeshletBackend::IndexedIndirect,
            bindless: MeshletBindlessConfig {
                max_textures: 4,
                max_samplers: 1,
            },
            capacities: MeshletCapacityConfig {
                max_instances: 4,
                max_candidate_meshlets: 8,
                max_visible_meshlets: 8,
                max_indirect_draws_per_bin: 4,
            },
            auto_benchmark_profile: None,
        };
        let requirements = MeshletCapabilities::from_parts_with_downlevel(
            wgpu::Backend::Vulkan,
            features,
            limits,
            wgpu::DownlevelFlags::INDIRECT_EXECUTION,
        )
        .device_requirements(&config)
        .unwrap();
        let renderer = MeshletRenderer::new(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
            config,
            &requirements,
            asset,
            &[],
            instances,
            &[],
            &[],
            TextureSamplingConfig::default(),
        )
        .unwrap();
        (device, queue, renderer)
    }

    fn indexed_renderer() -> (wgpu::Device, wgpu::Queue, MeshletRenderer) {
        let asset = MeshletSceneAsset::build(&[], Default::default()).unwrap();
        indexed_renderer_for(&asset, &[])
    }

    fn execute_empty_frame(occlusion: bool) {
        let (device, queue, mut renderer) = indexed_renderer();
        let extent = wgpu::Extent3d {
            width: 16,
            height: 8,
            depth_or_array_layers: 1,
        };
        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let color = frame
            .create_texture(TextureDesc {
                label: "meshlet.test-color".into(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_formats: Vec::new(),
                usage: UsagePolicy::Infer,
            })
            .unwrap();
        let depth = frame
            .create_texture(TextureDesc {
                label: "meshlet.test-depth".into(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                view_formats: Vec::new(),
                usage: UsagePolicy::Infer,
            })
            .unwrap();
        let prepared = renderer.prepare_frame(
            &queue,
            MeshletRenderInput {
                enable_occlusion_culling: occlusion,
                ..Default::default()
            },
            extent,
        );
        renderer
            .record_frame_graph(&mut frame, MeshRenderTargets::new(color, depth), &prepared)
            .unwrap();
        frame
            .mark_texture_root(color, RootReason::DebugCapture)
            .unwrap();
        frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute(&queue)
            .unwrap();
        renderer.after_submit(&device, prepared);
    }

    fn empty_frame_topology(occlusion: bool) -> FullCompilationReport {
        let (device, queue, mut renderer) = indexed_renderer();
        let extent = wgpu::Extent3d {
            width: 16,
            height: 8,
            depth_or_array_layers: 1,
        };
        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let color = frame
            .create_texture(TextureDesc::new_2d(
                "meshlet.topology-color",
                extent.width,
                extent.height,
                wgpu::TextureFormat::Rgba8Unorm,
            ))
            .unwrap();
        let depth = frame
            .create_texture(TextureDesc::new_2d(
                "meshlet.topology-depth",
                extent.width,
                extent.height,
                DEPTH_FORMAT,
            ))
            .unwrap();
        let prepared = renderer.prepare_frame(
            &queue,
            MeshletRenderInput {
                enable_occlusion_culling: occlusion,
                ..Default::default()
            },
            extent,
        );
        renderer
            .record_frame_graph(&mut frame, MeshRenderTargets::new(color, depth), &prepared)
            .unwrap();
        frame
            .mark_texture_root(color, RootReason::DebugCapture)
            .unwrap();
        let compiled = frame.compile(CompileOptions::full_report()).unwrap();
        let report = compiled.report().unwrap().full.as_ref().unwrap().clone();
        drop(compiled);
        renderer.after_discard(prepared);
        report
    }

    #[test]
    fn empty_indexed_frame_executes_with_and_without_current_frame_hiz() {
        execute_empty_frame(false);
        execute_empty_frame(true);
    }

    #[test]
    fn indexed_frame_graph_topology_is_stable_and_registers_one_texture_set() {
        let basic = empty_frame_topology(false);
        assert_eq!(
            basic
                .nodes
                .iter()
                .map(|node| node.label.as_str())
                .collect::<Vec<_>>(),
            [
                "meshlet.clear-frame-counters",
                "meshlet.instance-classify-lod-count",
                "meshlet.prefix-scan",
                "meshlet.candidate-scatter",
                "meshlet.coarse-cull",
                "meshlet.opaque-occluder-depth",
                "meshlet.indirect-prepare",
                "meshlet.backend-raster.indexed",
            ]
        );

        let occluded = empty_frame_topology(true);
        assert_eq!(
            occluded
                .nodes
                .iter()
                .map(|node| node.label.as_str())
                .collect::<Vec<_>>(),
            [
                "meshlet.clear-frame-counters",
                "meshlet.instance-classify-lod-count",
                "meshlet.prefix-scan",
                "meshlet.candidate-scatter",
                "meshlet.coarse-cull",
                "meshlet.opaque-occluder-depth",
                "meshlet.hiz-depth-to-mip0",
                "meshlet.hiz-mip0-to-mip1",
                "meshlet.hiz-mip1-to-mip2",
                "meshlet.hiz-mip2-to-mip3",
                "meshlet.hiz-mip3-to-mip4",
                "meshlet.clear-coarse-results",
                "meshlet.final-cull",
                "meshlet.indirect-prepare",
                "meshlet.backend-raster.indexed",
            ]
        );
        let texture_sets = occluded
            .resources
            .iter()
            .filter(|resource| resource.kind == ResourceKind::TextureSet)
            .collect::<Vec<_>>();
        assert_eq!(texture_sets.len(), 1);
        assert_eq!(texture_sets[0].label, "meshlet.bindless-textures");
        let scan_blocks = occluded
            .resources
            .iter()
            .find(|resource| resource.label == "meshlet.prefix-scan-blocks")
            .expect("the hierarchical scan scratch buffer must be registered once");
        let prefix_scan = occluded
            .nodes
            .iter()
            .find(|node| node.label == "meshlet.prefix-scan")
            .expect("the prefix-scan pass must survive graph compilation");
        assert!(occluded.accesses.iter().any(|access| {
            access.pass == prefix_scan.id
                && access.resource == scan_blocks.id
                && access.role == AccessRole::StorageBufferWrite
        }));
        assert_eq!(
            occluded
                .accesses
                .iter()
                .filter(|access| access.role == AccessRole::BindlessTextureSet)
                .count(),
            2,
            "the one resident texture-set is read by depth and final raster passes"
        );
    }

    #[test]
    fn mirrored_or_indeterminate_instances_use_the_two_sided_bin() {
        use crate::meshlet::MeshletPsoClass;

        assert_eq!(
            instance_pso_class(MeshletPsoClass::OpaqueBackface, glam::Mat4::IDENTITY),
            MeshletPsoClass::OpaqueBackface
        );
        for transform in [
            glam::Mat4::from_scale(glam::Vec3::new(-1.0, 1.0, 1.0)),
            glam::Mat4::from_scale(glam::Vec3::new(0.0, 1.0, 1.0)),
            glam::Mat4::from_cols(
                glam::Vec4::new(f32::NAN, 0.0, 0.0, 0.0),
                glam::Vec4::Y,
                glam::Vec4::Z,
                glam::Vec4::W,
            ),
        ] {
            assert_eq!(
                instance_pso_class(MeshletPsoClass::OpaqueBackface, transform),
                MeshletPsoClass::OpaqueTwoSided
            );
        }
        assert_eq!(
            instance_pso_class(
                MeshletPsoClass::OpaqueTwoSided,
                glam::Mat4::from_scale(glam::Vec3::splat(-1.0)),
            ),
            MeshletPsoClass::OpaqueTwoSided
        );
    }

    #[test]
    fn gpu_culling_bounds_compensate_asset_validation_tolerance() {
        let radius = 10.0;
        assert!(conservative_gpu_radius(radius) >= radius + radius * 1.0e-4);
        let cutoff = 0.5;
        assert!(conservative_gpu_cone_cutoff(cutoff) >= cutoff + 1.0e-4);
        assert_eq!(conservative_gpu_cone_cutoff(2.0), 2.0);
    }

    #[test]
    fn device_requirements_are_bound_to_the_exact_renderer_config() {
        let features =
            MeshletDeviceRequirements::required_features(MeshletBackend::IndexedIndirect).unwrap();
        let limits = wgpu::Limits {
            max_binding_array_elements_per_shader_stage: 5,
            max_binding_array_sampler_elements_per_shader_stage: 1,
            ..Default::default()
        };
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: limits.clone(),
            ..Default::default()
        });
        let config = MeshletRendererConfig {
            backend: MeshletBackend::IndexedIndirect,
            bindless: MeshletBindlessConfig {
                max_textures: 4,
                max_samplers: 1,
            },
            ..Default::default()
        };
        let requirements = MeshletCapabilities::from_parts_with_downlevel(
            wgpu::Backend::Vulkan,
            features,
            limits,
            wgpu::DownlevelFlags::INDIRECT_EXECUTION,
        )
        .device_requirements(&config)
        .unwrap();
        let mut different = config;
        different.bindless.max_textures = 3;

        assert!(matches!(
            validate_requirements(&device, different, &requirements),
            Err(MeshletRendererError::RequirementsConfigMismatch)
        ));
    }

    #[test]
    fn gpu_buffer_sizes_are_rejected_before_resource_creation() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let asset = MeshletSceneAsset::build(&[], Default::default()).unwrap();
        let capacities = MeshletCapacityConfig {
            max_instances: 1,
            max_candidate_meshlets: u32::MAX,
            max_visible_meshlets: 2,
            max_indirect_draws_per_bin: 1,
        };
        let error = validate_gpu_buffer_limits(&device, &asset, 1, 0, capacities).unwrap_err();
        assert!(matches!(
            error,
            MeshletRendererError::BufferLimitExceeded {
                buffer: "candidate work",
                ..
            }
        ));
    }

    #[test]
    fn target_contract_reports_extent_sample_and_format_mismatches() {
        let extent = wgpu::Extent3d {
            width: 16,
            height: 8,
            depth_or_array_layers: 1,
        };
        let base = TextureDesc::new_2d(
            "meshlet.target-contract",
            extent.width,
            extent.height,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        assert!(
            validate_render_target("color", &base, extent, wgpu::TextureFormat::Rgba8Unorm).is_ok()
        );

        let mut wrong_extent = base.clone();
        wrong_extent.size.width += 1;
        let mut wrong_samples = base.clone();
        wrong_samples.sample_count = 4;
        let mut wrong_format = base;
        wrong_format.format = wgpu::TextureFormat::Rgba16Float;
        for descriptor in [wrong_extent, wrong_samples, wrong_format] {
            assert!(matches!(
                validate_render_target(
                    "color",
                    &descriptor,
                    extent,
                    wgpu::TextureFormat::Rgba8Unorm
                ),
                Err(FrameGraphError::InvalidResourceDescriptor { .. })
            ));
        }
    }

    #[test]
    fn record_rejects_targets_that_do_not_match_the_prepared_frame() {
        let (device, queue, mut renderer) = indexed_renderer();
        let extent = wgpu::Extent3d {
            width: 16,
            height: 8,
            depth_or_array_layers: 1,
        };
        let prepared = renderer.prepare_frame(&queue, MeshletRenderInput::default(), extent);
        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let color = frame
            .create_texture(TextureDesc::new_2d(
                "meshlet.wrong-size-color",
                extent.width + 1,
                extent.height,
                wgpu::TextureFormat::Rgba8Unorm,
            ))
            .unwrap();
        let depth = frame
            .create_texture(TextureDesc::new_2d(
                "meshlet.target-depth",
                extent.width,
                extent.height,
                DEPTH_FORMAT,
            ))
            .unwrap();
        assert!(matches!(
            renderer.record_frame_graph(
                &mut frame,
                MeshRenderTargets::new(color, depth),
                &prepared
            ),
            Err(FrameGraphError::InvalidResourceDescriptor { .. })
        ));
        drop(frame);
        renderer.after_discard(prepared);
    }

    #[test]
    fn bindless_changes_publish_only_after_the_active_epoch_is_submitted() {
        let (device, queue, mut renderer) = indexed_renderer();
        let extent = wgpu::Extent3d {
            width: 16,
            height: 8,
            depth_or_array_layers: 1,
        };
        let prepared = renderer.prepare_frame(&queue, MeshletRenderInput::default(), extent);
        let active_epoch = prepared.bindless_epoch;
        let inserted = renderer.insert_texture(&Texture::white_1x1()).unwrap();
        renderer.remove_texture(inserted).unwrap();
        assert_eq!(renderer.bindless.current_epoch_id(), active_epoch);

        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let color = frame
            .create_texture(TextureDesc::new_2d(
                "meshlet.epoch-color",
                extent.width,
                extent.height,
                wgpu::TextureFormat::Rgba8Unorm,
            ))
            .unwrap();
        let depth = frame
            .create_texture(TextureDesc::new_2d(
                "meshlet.epoch-depth",
                extent.width,
                extent.height,
                DEPTH_FORMAT,
            ))
            .unwrap();
        renderer
            .record_frame_graph(&mut frame, MeshRenderTargets::new(color, depth), &prepared)
            .unwrap();
        frame
            .mark_texture_root(color, RootReason::DebugCapture)
            .unwrap();
        frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute(&queue)
            .unwrap();
        renderer.after_submit(&device, prepared);

        let next = renderer.prepare_frame(&queue, MeshletRenderInput::default(), extent);
        assert_ne!(next.bindless_epoch, active_epoch);
        renderer.after_discard(next);
    }

    #[test]
    fn discarded_frame_keeps_meshlet_stats_request() {
        let (_device, queue, mut renderer) = indexed_renderer();
        renderer.request_stats();
        let discarded = renderer.prepare_frame(
            &queue,
            MeshletRenderInput::default(),
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        assert_eq!(discarded.readback_index, Some(0));
        renderer.after_discard(discarded);
        let retry = renderer.prepare_frame(
            &queue,
            MeshletRenderInput::default(),
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        assert_eq!(retry.readback_index, Some(0));
    }

    #[test]
    fn one_meshlet_records_gpu_driven_work_and_maps_delayed_stats() {
        let source = RawStaticMesh::new(
            vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            vec![0, 1, 2],
        );
        let asset = MeshletSceneAsset::build(&[source], Default::default()).unwrap();
        let instances = [Instance {
            transform: glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -2.0)),
            mesh_id: 0,
            material_id: 0,
            _pad: [u32::MAX; 2],
        }];
        let (device, queue, mut renderer) = indexed_renderer_for(&asset, &instances);
        let extent = wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        };
        let mut graph = FrameGraph::with_device(&device);
        renderer.request_stats();

        let mut result = None;
        for _ in 0..8 {
            let mut frame = graph.begin_frame();
            let color = frame
                .create_texture(TextureDesc::new_2d(
                    "meshlet.triangle-color",
                    extent.width,
                    extent.height,
                    wgpu::TextureFormat::Rgba8Unorm,
                ))
                .unwrap();
            let depth = frame
                .create_texture(TextureDesc::new_2d(
                    "meshlet.triangle-depth",
                    extent.width,
                    extent.height,
                    DEPTH_FORMAT,
                ))
                .unwrap();
            let prepared = renderer.prepare_frame(
                &queue,
                MeshletRenderInput {
                    enable_occlusion_culling: false,
                    ..Default::default()
                },
                extent,
            );
            renderer
                .record_frame_graph(&mut frame, MeshRenderTargets::new(color, depth), &prepared)
                .unwrap();
            frame
                .mark_texture_root(color, RootReason::DebugCapture)
                .unwrap();
            frame
                .compile(CompileOptions::default())
                .unwrap()
                .execute(&queue)
                .unwrap();
            renderer.after_submit(&device, prepared);
            result = renderer.take_stats(&device).or(result);
        }

        let stats = result.expect("three-frame-delayed stats should eventually map");
        assert_eq!(stats.total_instances, 1);
        // wgpu's noop backend validates commands and mapping, but deliberately does not execute
        // shader arithmetic. Counter correctness is covered on Vulkan integration hardware.
        assert!(stats.overflow.is_empty());
    }
}

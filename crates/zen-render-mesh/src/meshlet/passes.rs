use std::{mem::size_of, num::NonZeroU64};

use super::{
    asset::MeshletPsoClass,
    config::MeshletBackend,
    gpu_scene::MeshletGpuScene,
    gpu_types::{
        DispatchIndirectArgs, DrawIndexedIndirectArgs, FRAME_UNIFORM_SIZE, GpuCounters,
        PREFIX_SCAN_WORKGROUP_SIZE, RASTER_UNIFORM_STRIDE, RasterUniform, prefix_scan_block_count,
    },
};

const CLASSIFY_WORKGROUP_SIZE: u32 = 64;

/// Pipelines and layouts for the shared meshlet front end and the selected raster backend.
///
/// Bind groups are intentionally assembled by the encode methods. This keeps the pass set
/// independent from frame-graph resource lifetimes and, importantly, lets coarse and final culling
/// bind different `FrameUniform` buffers. A renderer that wants to cache bind groups can use the
/// exposed layouts without changing the pipeline ABI.
pub(crate) struct MeshletPassSet {
    max_dispatch_dimension: u32,

    classify: ComputeStage,
    prefix_scan: PrefixScanStages,
    scatter: ComputeStage,
    cull: ComputeStage,
    indirect_prepare: ComputeStage,

    indexed_layout: wgpu::BindGroupLayout,
    indexed_depth: PsoPipelines,
    indexed_final: PsoPipelines,
    mesh: Option<MeshRasterPipelines>,

    hiz_sampler: wgpu::Sampler,
    dummy_hiz_texture: wgpu::Texture,
}

struct ComputeStage {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

struct PrefixScanStages {
    layout: wgpu::BindGroupLayout,
    local: wgpu::ComputePipeline,
    block_sums: wgpu::ComputePipeline,
    add_offsets: wgpu::ComputePipeline,
}

struct PsoPipelines {
    backface: wgpu::RenderPipeline,
    two_sided: wgpu::RenderPipeline,
}

impl PsoPipelines {
    fn get(&self, bin: MeshletPsoClass) -> &wgpu::RenderPipeline {
        match bin {
            MeshletPsoClass::OpaqueBackface => &self.backface,
            MeshletPsoClass::OpaqueTwoSided => &self.two_sided,
        }
    }
}

struct MeshRasterPipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: PsoPipelines,
    uses_task_shader: bool,
}

impl MeshletPassSet {
    #[expect(
        clippy::too_many_lines,
        reason = "pipeline creation keeps every WGSL binding adjacent to its explicit wgpu layout"
    )]
    pub(crate) fn new(
        device: &wgpu::Device,
        backend: MeshletBackend,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        bindless_layout: &wgpu::BindGroupLayout,
        max_dispatch_dimension: u32,
    ) -> Self {
        assert!(
            backend.is_resolved(),
            "MeshletBackend::Auto must be resolved before constructing MeshletPassSet"
        );

        let classify = create_compute_stage(
            device,
            "meshlet.classify",
            include_str!("../../shaders/meshlet/classify.wgsl"),
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                uniform_entry(6, false, FRAME_UNIFORM_SIZE),
            ],
        );
        let prefix_scan = create_prefix_scan_stages(device);
        let scatter_constants = [(
            "ENABLE_TASK_PACKETS",
            f64::from(u8::from(task_packets_enabled(backend))),
        )];
        let scatter = create_compute_stage_with_constants(
            device,
            "meshlet.scatter",
            include_str!("../../shaders/meshlet/scatter.wgsl"),
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                // Binding 2 was deliberately retired: PSO class is InstanceData._pad.x rather
                // than a material texture bit. Keeping the hole makes that ABI change obvious.
                storage_entry_at(3, wgpu::ShaderStages::COMPUTE, true),
                storage_entry_at(4, wgpu::ShaderStages::COMPUTE, false),
                storage_entry_at(5, wgpu::ShaderStages::COMPUTE, false),
                storage_entry_at(6, wgpu::ShaderStages::COMPUTE, false),
                uniform_entry(7, false, FRAME_UNIFORM_SIZE),
            ],
            &scatter_constants,
        );
        let cull = create_compute_stage(
            device,
            "meshlet.cull",
            include_str!("../../shaders/meshlet/cull.wgsl"),
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                uniform_entry(6, false, FRAME_UNIFORM_SIZE),
                texture_entry(7, wgpu::ShaderStages::COMPUTE),
                sampler_entry(8, wgpu::ShaderStages::COMPUTE),
            ],
        );
        let device_limits = device.limits();
        let (mesh_workgroup_total_limit, task_workgroup_total_limit) =
            indirect_workgroup_total_limits(
                backend,
                device_limits.max_mesh_workgroup_total_count,
                device_limits.max_task_workgroup_total_count,
            );
        let (mesh_dispatch_width, mesh_dispatch_capacity) =
            rectangular_dispatch_limit(mesh_workgroup_total_limit, max_dispatch_dimension);
        let (task_dispatch_width, task_dispatch_capacity) =
            rectangular_dispatch_limit(task_workgroup_total_limit, max_dispatch_dimension);
        let indirect_constants = [
            (
                "MAX_MESH_WORKGROUP_TOTAL_COUNT",
                f64::from(mesh_dispatch_capacity),
            ),
            (
                "MAX_TASK_WORKGROUP_TOTAL_COUNT",
                f64::from(task_dispatch_capacity),
            ),
            ("MESH_DISPATCH_WIDTH", f64::from(mesh_dispatch_width)),
            ("TASK_DISPATCH_WIDTH", f64::from(task_dispatch_width)),
        ];
        let indirect_prepare = create_compute_stage_with_constants(
            device,
            "meshlet.indirect-prepare",
            include_str!("../../shaders/meshlet/indirect_prepare.wgsl"),
            &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                uniform_entry(3, false, FRAME_UNIFORM_SIZE),
                storage_entry(4, false),
            ],
            &indirect_constants,
        );

        let indexed_shader = checked_shader_module(
            device,
            "meshlet.indexed.shader",
            include_str!("../../shaders/meshlet/indexed.wgsl"),
        );
        let indexed_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet.indexed.scene-layout"),
            entries: &[
                storage_entry_at(0, wgpu::ShaderStages::VERTEX, true),
                storage_entry_at(1, wgpu::ShaderStages::FRAGMENT, true),
                storage_entry_at(2, wgpu::ShaderStages::VERTEX, true),
                storage_entry_at(3, wgpu::ShaderStages::VERTEX, true),
                uniform_entry_at(
                    4,
                    wgpu::ShaderStages::VERTEX,
                    true,
                    size_of::<RasterUniform>() as u64,
                ),
            ],
        });
        let indexed_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("meshlet.indexed.pipeline-layout"),
                bind_group_layouts: &[Some(&indexed_layout), Some(bindless_layout)],
                immediate_size: 0,
            });

        let indexed_depth = create_indexed_pso_pipelines(
            device,
            &indexed_pipeline_layout,
            &indexed_shader,
            color_format,
            depth_format,
            true,
        );
        let indexed_final = create_indexed_pso_pipelines(
            device,
            &indexed_pipeline_layout,
            &indexed_shader,
            color_format,
            depth_format,
            false,
        );

        // Do not even create a mesh shader module for IndexedIndirect. This is what allows the
        // non-experimental tier to use an ordinary device without ExperimentalFeatures enabled.
        let mesh = backend.uses_mesh_shaders().then(|| {
            create_mesh_raster_pipelines(
                device,
                backend,
                color_format,
                depth_format,
                bindless_layout,
            )
        });

        let hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("meshlet.hiz.nearest-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let dummy_hiz_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("meshlet.hiz.disabled-dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            // This value is shared with FrameUniform. For mesh backends it is the minimum of the
            // compute, mesh, and (where applicable) task limits, so every 2-D dispatch stays legal.
            max_dispatch_dimension: max_dispatch_dimension.max(1),
            classify,
            prefix_scan,
            scatter,
            cull,
            indirect_prepare,
            indexed_layout,
            indexed_depth,
            indexed_final,
            mesh,
            hiz_sampler,
            dummy_hiz_texture,
        }
    }

    /// Hi-Z-disabled placeholder used by coarse culling and non-occlusion frames.
    pub(crate) fn dummy_hiz_texture(&self) -> &wgpu::Texture {
        &self.dummy_hiz_texture
    }

    pub(crate) fn encode_classify(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        scene: &MeshletGpuScene,
        frame_uniform: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet.classify.bind-group"),
            layout: &self.classify.layout,
            entries: &[
                buffer_entry(0, &scene.meshes),
                buffer_entry(1, &scene.lods),
                buffer_entry(2, &scene.instances),
                buffer_entry(3, &scene.classifications),
                buffer_entry(4, &scene.lod_history),
                buffer_entry(5, &scene.counters),
                sized_buffer_entry(6, frame_uniform, FRAME_UNIFORM_SIZE),
            ],
        });
        let (x, y) = dispatch_2d(
            scene.instance_count.div_ceil(CLASSIFY_WORKGROUP_SIZE),
            self.max_dispatch_dimension,
        );
        pass.set_pipeline(&self.classify.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    pub(crate) fn encode_prefix_scan(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        scene: &MeshletGpuScene,
        frame_uniform: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet.prefix-scan.bind-group"),
            layout: &self.prefix_scan.layout,
            entries: &[
                buffer_entry(0, &scene.classifications),
                buffer_entry(1, &scene.counters),
                sized_buffer_entry(2, frame_uniform, FRAME_UNIFORM_SIZE),
                buffer_entry(3, &scene.candidate_dispatch),
                buffer_entry(4, &scene.scan_blocks),
            ],
        });
        pass.set_bind_group(0, &bind_group, &[]);
        let block_count = prefix_scan_block_count(scene.instance_count);
        if block_count != 0 {
            let (x, y) = dispatch_2d(block_count, self.max_dispatch_dimension);
            pass.set_pipeline(&self.prefix_scan.local);
            pass.dispatch_workgroups(x, y, 1);
        }
        pass.set_pipeline(&self.prefix_scan.block_sums);
        pass.dispatch_workgroups(1, 1, 1);
        if block_count != 0 {
            let (x, y) = dispatch_2d(block_count, self.max_dispatch_dimension);
            pass.set_pipeline(&self.prefix_scan.add_offsets);
            pass.dispatch_workgroups(x, y, 1);
        }
    }

    pub(crate) fn encode_scatter(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        scene: &MeshletGpuScene,
        frame_uniform: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet.scatter.bind-group"),
            layout: &self.scatter.layout,
            entries: &[
                buffer_entry(0, &scene.lods),
                buffer_entry(1, &scene.instances),
                buffer_entry(3, &scene.classifications),
                buffer_entry(4, &scene.candidates),
                buffer_entry(5, &scene.task_packets),
                buffer_entry(6, &scene.counters),
                sized_buffer_entry(7, frame_uniform, FRAME_UNIFORM_SIZE),
            ],
        });
        let (x, y) = dispatch_2d(
            scene.instance_count.div_ceil(CLASSIFY_WORKGROUP_SIZE),
            self.max_dispatch_dimension,
        );
        pass.set_pipeline(&self.scatter.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    /// Encodes either coarse culling (with a Hi-Z-disabled uniform and the dummy view) or final
    /// culling (with the final uniform and current-frame pyramid). Candidate dispatch dimensions
    /// are consumed entirely on GPU.
    pub(crate) fn encode_cull(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        scene: &MeshletGpuScene,
        frame_uniform: &wgpu::Buffer,
        hiz_view: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet.cull.bind-group"),
            layout: &self.cull.layout,
            entries: &[
                buffer_entry(0, &scene.meshlets),
                buffer_entry(1, &scene.instances),
                buffer_entry(2, &scene.candidates),
                buffer_entry(3, &scene.visible),
                buffer_entry(4, &scene.draw_args),
                buffer_entry(5, &scene.counters),
                sized_buffer_entry(6, frame_uniform, FRAME_UNIFORM_SIZE),
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                },
            ],
        });
        pass.set_pipeline(&self.cull.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups_indirect(&scene.candidate_dispatch, 0);
    }

    pub(crate) fn encode_indirect_prepare(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        scene: &MeshletGpuScene,
        frame_uniform: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet.indirect-prepare.bind-group"),
            layout: &self.indirect_prepare.layout,
            entries: &[
                buffer_entry(0, &scene.counters),
                buffer_entry(1, &scene.mesh_dispatch),
                buffer_entry(2, &scene.task_dispatch),
                sized_buffer_entry(3, frame_uniform, FRAME_UNIFORM_SIZE),
                buffer_entry(4, &scene.backend_work_counts),
            ],
        });
        pass.set_pipeline(&self.indirect_prepare.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Draws the coarse occluder list with indexed fallback geometry. Both mesh backends use this
    /// same path so the Hi-Z source is independent from the backend being measured.
    pub(crate) fn encode_indexed_depth(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass<'_>,
        scene: &MeshletGpuScene,
        bindless: &wgpu::BindGroup,
        bin: MeshletPsoClass,
    ) {
        let bind_group = self.create_indexed_bind_group(device, scene);
        pass.set_pipeline(self.indexed_depth.get(bin));
        pass.set_bind_group(0, &bind_group, &[raster_dynamic_offset(bin)]);
        pass.set_bind_group(1, bindless, &[]);
        pass.set_index_buffer(scene.fallback_indices.slice(..), wgpu::IndexFormat::Uint32);
        encode_indexed_indirect(pass, scene, bin);
    }

    pub(crate) fn encode_indexed_raster(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass<'_>,
        scene: &MeshletGpuScene,
        bindless: &wgpu::BindGroup,
        bin: MeshletPsoClass,
    ) {
        let bind_group = self.create_indexed_bind_group(device, scene);
        pass.set_pipeline(self.indexed_final.get(bin));
        pass.set_bind_group(0, &bind_group, &[raster_dynamic_offset(bin)]);
        pass.set_bind_group(1, bindless, &[]);
        pass.set_index_buffer(scene.fallback_indices.slice(..), wgpu::IndexFormat::Uint32);
        encode_indexed_indirect(pass, scene, bin);
    }

    /// Draws the selected mesh-only or task+mesh backend. The renderer still chooses which pass to
    /// record at startup; this method contains no per-frame backend fallback.
    #[expect(
        clippy::too_many_arguments,
        reason = "mesh rasterization binds the explicit scene, bindless, frame, and Hi-Z inputs"
    )]
    pub(crate) fn encode_mesh_raster(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass<'_>,
        scene: &MeshletGpuScene,
        bindless: &wgpu::BindGroup,
        frame_uniform: &wgpu::Buffer,
        hiz_view: &wgpu::TextureView,
        bin: MeshletPsoClass,
    ) {
        let mesh = self
            .mesh
            .as_ref()
            .expect("encode_mesh_raster requires MeshOnly or TaskMesh");
        let bind_group =
            self.create_mesh_bind_group(device, &mesh.layout, scene, frame_uniform, hiz_view);
        pass.set_pipeline(mesh.pipelines.get(bin));
        pass.set_bind_group(0, &bind_group, &[raster_dynamic_offset(bin)]);
        pass.set_bind_group(1, bindless, &[]);
        let dispatch = if mesh.uses_task_shader {
            &scene.task_dispatch
        } else {
            &scene.mesh_dispatch
        };
        pass.draw_mesh_tasks_indirect(
            dispatch,
            bin as u64 * size_of::<DispatchIndirectArgs>() as u64,
        );
    }

    fn create_indexed_bind_group(
        &self,
        device: &wgpu::Device,
        scene: &MeshletGpuScene,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet.indexed.scene-bind-group"),
            layout: &self.indexed_layout,
            entries: &[
                buffer_entry(0, &scene.vertices),
                buffer_entry(1, &scene.materials),
                buffer_entry(2, &scene.instances),
                buffer_entry(3, &scene.visible),
                sized_buffer_entry(4, &scene.raster_uniform, size_of::<RasterUniform>() as u64),
            ],
        })
    }

    fn create_mesh_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        scene: &MeshletGpuScene,
        frame_uniform: &wgpu::Buffer,
        hiz_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet.mesh.scene-bind-group"),
            layout,
            entries: &[
                buffer_entry(0, &scene.vertices),
                buffer_entry(1, &scene.materials),
                buffer_entry(2, &scene.instances),
                buffer_entry(3, &scene.meshlets),
                buffer_entry(4, &scene.meshlet_vertices),
                buffer_entry(5, &scene.micro_indices),
                buffer_entry(6, &scene.visible),
                buffer_entry(7, &scene.counters),
                sized_buffer_entry(8, &scene.raster_uniform, size_of::<RasterUniform>() as u64),
                buffer_entry(9, &scene.task_packets),
                sized_buffer_entry(10, frame_uniform, FRAME_UNIFORM_SIZE),
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                },
                buffer_entry(13, &scene.backend_work_counts),
            ],
        })
    }
}

fn create_compute_stage(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> ComputeStage {
    create_compute_stage_with_constants(device, label, source, entries, &[])
}

fn create_compute_stage_with_constants(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
    entries: &[wgpu::BindGroupLayoutEntry],
    constants: &[(&str, f64)],
) -> ComputeStage {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}.layout")),
        entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}.pipeline-layout")),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = checked_shader_module(device, &format!("{label}.shader"), source);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}.pipeline")),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    });
    ComputeStage { layout, pipeline }
}

fn create_prefix_scan_stages(device: &wgpu::Device) -> PrefixScanStages {
    debug_assert!(
        device.limits().max_compute_invocations_per_workgroup >= PREFIX_SCAN_WORKGROUP_SIZE
    );
    debug_assert!(device.limits().max_compute_workgroup_size_x >= PREFIX_SCAN_WORKGROUP_SIZE);

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("meshlet.prefix-scan.layout"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2, false, FRAME_UNIFORM_SIZE),
            storage_entry(3, false),
            storage_entry(4, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("meshlet.prefix-scan.pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = checked_shader_module(
        device,
        "meshlet.prefix-scan.shader",
        include_str!("../../shaders/meshlet/prefix_scan.wgsl"),
    );
    let create_pipeline = |label: &'static str, entry_point: &'static str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    PrefixScanStages {
        layout,
        local: create_pipeline("meshlet.prefix-scan.local.pipeline", "scan_blocks_local"),
        block_sums: create_pipeline("meshlet.prefix-scan.block-sums.pipeline", "scan_block_sums"),
        add_offsets: create_pipeline(
            "meshlet.prefix-scan.add-offsets.pipeline",
            "add_block_offsets",
        ),
    }
}

fn checked_shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &'static str,
) -> wgpu::ShaderModule {
    // The safe API is deliberate: it enables ShaderRuntimeChecks::checked(), including mesh
    // primitive-index clamping, instead of the often-recommended unchecked mesh-shader fast path.
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn create_indexed_pso_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    depth_only: bool,
) -> PsoPipelines {
    let create = |label: &'static str, cull_mode: Option<wgpu::Face>| {
        let targets = [Some(wgpu::ColorTargetState {
            format: color_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(depth_state(depth_format, depth_only)),
            multisample: wgpu::MultisampleState::default(),
            fragment: (!depth_only).then_some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    PsoPipelines {
        backface: create(
            if depth_only {
                "meshlet.indexed.depth.backface"
            } else {
                "meshlet.indexed.final.backface"
            },
            Some(wgpu::Face::Back),
        ),
        two_sided: create(
            if depth_only {
                "meshlet.indexed.depth.two-sided"
            } else {
                "meshlet.indexed.final.two-sided"
            },
            None,
        ),
    }
}

fn create_mesh_raster_pipelines(
    device: &wgpu::Device,
    backend: MeshletBackend,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    bindless_layout: &wgpu::BindGroupLayout,
) -> MeshRasterPipelines {
    debug_assert!(backend.uses_mesh_shaders());
    let task_mesh = backend.uses_task_shaders();
    let mesh_stage = wgpu::ShaderStages::MESH;
    let task_stage = wgpu::ShaderStages::TASK;
    let task_and_mesh = task_stage | mesh_stage;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("meshlet.mesh.scene-layout"),
        entries: &[
            storage_entry_at(0, mesh_stage, true),
            storage_entry_at(1, wgpu::ShaderStages::FRAGMENT, true),
            storage_entry_at(2, task_and_mesh, true),
            storage_entry_at(3, task_and_mesh, true),
            storage_entry_at(4, mesh_stage, true),
            storage_entry_at(5, mesh_stage, true),
            storage_entry_at(6, task_and_mesh, true),
            storage_entry_at(7, task_and_mesh, false),
            uniform_entry_at(8, task_and_mesh, true, size_of::<RasterUniform>() as u64),
            storage_entry_at(9, task_stage, true),
            uniform_entry_at(10, task_stage, false, FRAME_UNIFORM_SIZE),
            texture_entry(11, task_stage),
            sampler_entry(12, task_stage),
            storage_entry_at(13, task_and_mesh, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("meshlet.mesh.pipeline-layout"),
        bind_group_layouts: &[Some(&layout), Some(bindless_layout)],
        immediate_size: 0,
    });
    let shader = checked_shader_module(
        device,
        "meshlet.mesh.shader",
        include_str!("../../shaders/meshlet/mesh.wgsl"),
    );
    let task_state = || {
        task_mesh.then_some(wgpu::TaskState {
            module: &shader,
            entry_point: Some("ts_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    };
    let mesh_entry = if task_mesh { "ms_task_main" } else { "ms_main" };
    let create = |label: &'static str, cull_mode: Option<wgpu::Face>| {
        let targets = [Some(wgpu::ColorTargetState {
            format: color_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            task: task_state(),
            mesh: wgpu::MeshState {
                module: &shader,
                entry_point: Some(mesh_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(depth_state(depth_format, false)),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            multiview: None,
            cache: None,
        })
    };
    let (backface_label, two_sided_label) = if task_mesh {
        (
            "meshlet.task-mesh.final.backface",
            "meshlet.task-mesh.final.two-sided",
        )
    } else {
        (
            "meshlet.mesh-only.final.backface",
            "meshlet.mesh-only.final.two-sided",
        )
    };
    MeshRasterPipelines {
        layout,
        pipelines: PsoPipelines {
            backface: create(backface_label, Some(wgpu::Face::Back)),
            two_sided: create(two_sided_label, None),
        },
        uses_task_shader: task_mesh,
    }
}

fn depth_state(format: wgpu::TextureFormat, depth_only: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format,
        depth_write_enabled: Some(depth_only),
        depth_compare: Some(if depth_only {
            wgpu::CompareFunction::Less
        } else {
            wgpu::CompareFunction::LessEqual
        }),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn encode_indexed_indirect(
    pass: &mut wgpu::RenderPass<'_>,
    scene: &MeshletGpuScene,
    bin: MeshletPsoClass,
) {
    let draw_stride = size_of::<DrawIndexedIndirectArgs>() as u64;
    let draw_offset = bin as u64 * scene.capacities.max_indirect_draws_per_bin as u64 * draw_stride;
    let count_offset = match bin {
        MeshletPsoClass::OpaqueBackface => GpuCounters::VISIBLE_BACKFACE_OFFSET,
        MeshletPsoClass::OpaqueTwoSided => GpuCounters::VISIBLE_TWO_SIDED_OFFSET,
    };
    pass.multi_draw_indexed_indirect_count(
        &scene.draw_args,
        draw_offset,
        &scene.counters,
        count_offset,
        scene.capacities.max_indirect_draws_per_bin,
    );
}

fn dispatch_2d(group_count: u32, max_dimension: u32) -> (u32, u32) {
    if group_count == 0 {
        return (0, 1);
    }
    let x = group_count.min(max_dimension);
    let y = group_count.div_ceil(x).min(max_dimension);
    debug_assert!(
        u64::from(group_count) <= u64::from(max_dimension) * u64::from(max_dimension),
        "meshlet instance dispatch exceeds the device's two-dimensional dispatch capacity"
    );
    (x, y)
}

fn indirect_workgroup_total_limits(
    backend: MeshletBackend,
    mesh_limit: u32,
    task_limit: u32,
) -> (u32, u32) {
    match backend {
        MeshletBackend::IndexedIndirect => (0, 0),
        MeshletBackend::MeshOnly => (mesh_limit, 0),
        MeshletBackend::TaskMesh => (0, task_limit),
        MeshletBackend::Auto => unreachable!("Auto is resolved before pipeline construction"),
    }
}

/// Chooses the narrowest legal row width that can span the total limit within the Y dimension.
/// Keeping X small bounds duplicate-last padding to strictly fewer than `width` workgroups.
fn rectangular_dispatch_limit(total_limit: u32, max_dimension: u32) -> (u32, u32) {
    if total_limit == 0 {
        return (1, 0);
    }
    let max_dimension = max_dimension.max(1);
    let width = total_limit.div_ceil(max_dimension).min(max_dimension);
    let rows = (total_limit / width).min(max_dimension);
    (width, width * rows)
}

fn task_packets_enabled(_backend: MeshletBackend) -> bool {
    // The Vulkan compatibility path feeds TaskMesh from the compute-produced visible list. Keep
    // the retired packet binding ABI stable, but do not spend scatter time or overflow capacity on
    // packets that no backend consumes.
    false
}

fn raster_dynamic_offset(bin: MeshletPsoClass) -> u32 {
    u32::try_from(bin as u64 * RASTER_UNIFORM_STRIDE)
        .expect("the fixed two-bin raster uniform offsets fit in DynamicOffset")
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    storage_entry_at(binding, wgpu::ShaderStages::COMPUTE, read_only)
}

fn storage_entry_at(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            // Runtime-sized arrays may legally be backed by the renderer's four-byte empty
            // sentinel buffer, so their minimum is enforced by checked shader indexing instead.
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, dynamic: bool, size: u64) -> wgpu::BindGroupLayoutEntry {
    uniform_entry_at(binding, wgpu::ShaderStages::COMPUTE, dynamic, size)
}

fn uniform_entry_at(
    binding: u32,
    visibility: wgpu::ShaderStages,
    dynamic: bool,
    size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
        count: None,
    }
}

fn buffer_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn sized_buffer_entry<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    size: u64,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(size),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet::config::TASK_PACKET_MESHLET_COUNT;

    fn bounded_dispatch_for_test(
        count: u32,
        maximum_dimension: u32,
        total_limit: u32,
    ) -> ((u32, u32), bool) {
        if count == 0 || total_limit == 0 {
            return ((0, 1), false);
        }
        let (width, capacity) = rectangular_dispatch_limit(total_limit, maximum_dimension);
        let safe_count = count.min(capacity);
        if safe_count <= width {
            return ((safe_count, 1), safe_count < count);
        }
        let rows = safe_count.div_ceil(width);
        ((width, rows), safe_count < count)
    }

    fn hierarchical_prefix_scan_for_test(counts: &[u32], capacity: u32) -> (Vec<u32>, u32, bool) {
        let mut overflow = false;
        let mut saturating_add = |left: u32, right: u32| match left.checked_add(right) {
            Some(sum) => sum,
            None => {
                overflow = true;
                u32::MAX
            }
        };
        let block_size = PREFIX_SCAN_WORKGROUP_SIZE as usize;
        let mut offsets = vec![0; counts.len()];
        let mut block_sums = Vec::with_capacity(counts.len().div_ceil(block_size));
        for (block_id, block) in counts.chunks(block_size).enumerate() {
            let mut running = 0u32;
            for (lane, count) in block.iter().copied().enumerate() {
                offsets[block_id * block_size + lane] = running.min(capacity);
                running = saturating_add(running, count);
            }
            block_sums.push(running);
        }

        let mut block_running = 0u32;
        for block_sum in &mut block_sums {
            let sum = *block_sum;
            *block_sum = block_running;
            block_running = saturating_add(block_running, sum);
        }
        for (block_id, block) in offsets.chunks_mut(block_size).enumerate() {
            for local_offset in block {
                *local_offset = saturating_add(block_sums[block_id], *local_offset).min(capacity);
            }
        }
        if block_running > capacity {
            overflow = true;
        }
        (offsets, block_running.min(capacity), overflow)
    }

    fn reference_prefix_scan(counts: &[u32], capacity: u32) -> (Vec<u32>, u32, bool) {
        let mut running = 0u128;
        let offsets = counts
            .iter()
            .map(|count| {
                let offset = running.min(u128::from(u32::MAX)).min(u128::from(capacity)) as u32;
                running += u128::from(*count);
                offset
            })
            .collect();
        let total = running.min(u128::from(u32::MAX)).min(u128::from(capacity)) as u32;
        let overflow = running > u128::from(u32::MAX) || running > u128::from(capacity);
        (offsets, total, overflow)
    }

    fn projected_cube_bounds(
        view_projection: glam::Mat4,
        center: glam::Vec3,
        radius: f32,
    ) -> Option<(glam::Vec3, glam::Vec3)> {
        let mut minimum = glam::Vec3::splat(f32::INFINITY);
        let mut maximum = glam::Vec3::splat(f32::NEG_INFINITY);
        for corner in 0..8_u32 {
            let offset = glam::Vec3::new(
                if corner & 1 == 0 { -radius } else { radius },
                if corner & 2 == 0 { -radius } else { radius },
                if corner & 4 == 0 { -radius } else { radius },
            );
            let clip = view_projection * (center + offset).extend(1.0);
            if !clip.is_finite() || clip.w <= 1.0e-5 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            minimum = minimum.min(ndc);
            maximum = maximum.max(ndc);
        }
        Some((minimum, maximum))
    }

    fn vertical_focal_pixels_for_test(
        view: glam::Mat4,
        view_projection: glam::Mat4,
        viewport_height: f32,
    ) -> f32 {
        let view_y = glam::Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let clip_y = glam::Vec3::new(
            view_projection.x_axis.y,
            view_projection.y_axis.y,
            view_projection.z_axis.y,
        );
        0.5 * viewport_height * clip_y.length() / view_y.length().max(1.0e-20)
    }

    fn gershgorin_scale_for_test(model: glam::Mat3) -> f32 {
        let [column0, column1, column2] = [model.x_axis, model.y_axis, model.z_axis];
        let g00 = column0.dot(column0);
        let g11 = column1.dot(column1);
        let g22 = column2.dot(column2);
        let g01 = column0.dot(column1).abs();
        let g02 = column0.dot(column2).abs();
        let g12 = column1.dot(column2).abs();
        (g00 + g01 + g02)
            .max(g11 + g01 + g12)
            .max(g22 + g02 + g12)
            .sqrt()
    }

    fn indexed_device() -> wgpu::Device {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            label: Some("meshlet.passes.noop-device"),
            required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
            required_limits: wgpu::Limits {
                // wgpu counts both texture[8] and sampler[4] elements in this aggregate limit.
                max_binding_array_elements_per_shader_stage: 12,
                max_binding_array_sampler_elements_per_shader_stage: 4,
                max_storage_buffers_per_shader_stage: 8,
                ..Default::default()
            },
            ..Default::default()
        });
        device
    }

    fn test_bindless_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet.passes.test-bindless-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: std::num::NonZeroU32::new(8),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: std::num::NonZeroU32::new(4),
                },
            ],
        })
    }

    #[test]
    fn noop_device_constructs_every_indexed_pipeline_with_eight_storage_limit() {
        let device = indexed_device();
        let bindless = test_bindless_layout(&device);
        let passes = MeshletPassSet::new(
            &device,
            MeshletBackend::IndexedIndirect,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Depth32Float,
            &bindless,
            device.limits().max_compute_workgroups_per_dimension,
        );
        assert!(passes.mesh.is_none());
    }

    #[test]
    fn dispatch_flattens_to_two_dimensions_without_padding_the_first_row() {
        assert_eq!(dispatch_2d(0, 65_535), (0, 1));
        assert_eq!(dispatch_2d(17, 65_535), (17, 1));
        assert_eq!(dispatch_2d(65_536, 65_535), (65_535, 2));
    }

    #[test]
    fn hierarchical_prefix_scan_matches_wide_reference_at_block_boundaries() {
        for count in [0usize, 1, 255, 256, 257, 16_383, 16_384, 16_385, 262_144] {
            let counts = (0..count)
                .map(|index| ((index * 17 + 3) % 11) as u32)
                .collect::<Vec<_>>();
            let exact_total = counts.iter().map(|value| u64::from(*value)).sum::<u64>();
            for capacity in [
                1,
                exact_total.saturating_sub(1).min(u64::from(u32::MAX)) as u32,
                exact_total.min(u64::from(u32::MAX)) as u32,
                exact_total.saturating_add(1).min(u64::from(u32::MAX)) as u32,
                u32::MAX,
            ] {
                assert_eq!(
                    hierarchical_prefix_scan_for_test(&counts, capacity),
                    reference_prefix_scan(&counts, capacity),
                    "count={count}, capacity={capacity}",
                );
            }
        }
    }

    #[test]
    fn hierarchical_prefix_scan_saturates_u32_and_reports_overflow() {
        for counts in [
            vec![u32::MAX, 1],
            vec![u32::MAX - 1, 1, 1],
            vec![u32::MAX; PREFIX_SCAN_WORKGROUP_SIZE as usize + 1],
        ] {
            assert_eq!(
                hierarchical_prefix_scan_for_test(&counts, u32::MAX),
                reference_prefix_scan(&counts, u32::MAX),
            );
            assert!(hierarchical_prefix_scan_for_test(&counts, u32::MAX).2);
        }
    }

    #[test]
    fn mesh_dispatch_never_exceeds_the_total_workgroup_limit() {
        for maximum_dimension in [1, 7, 64, 1_024] {
            for total_limit in [0, 1, 31, 1_000, 1_024, 65_536] {
                for count in [0, 1, 17, 999, 1_000, 1_025, 65_536] {
                    let ((x, y), overflow) =
                        bounded_dispatch_for_test(count, maximum_dimension, total_limit);
                    assert!(x <= maximum_dimension);
                    assert!(y <= maximum_dimension);
                    assert!(u64::from(x) * u64::from(y) <= u64::from(total_limit));
                    if !overflow && total_limit != 0 {
                        assert!(u64::from(x) * u64::from(y) >= u64::from(count));
                    }
                }
            }
        }

        // Prefer the minimum legal row width. Any padded final row therefore duplicates fewer
        // than `x` meshlets, while the full launch stays under the stage total.
        assert_eq!(
            bounded_dispatch_for_test(65_536, 65_535, 70_000),
            ((2, 32_768), false)
        );
        assert_eq!(
            bounded_dispatch_for_test(1_000, 64, 1_024),
            ((16, 63), false)
        );
        assert_eq!(
            bounded_dispatch_for_test(1_000, 64, 1_000),
            ((16, 62), true)
        );
    }

    #[test]
    fn zero_specialization_limit_disables_an_unused_stage_without_overflow() {
        // IndexedIndirect specializes both mesh and task totals to zero. MeshOnly specializes only
        // task to zero. Non-zero counters left by the shared front end must not report a device
        // dispatch-limit overflow for a stage that the selected backend never executes.
        assert_eq!(bounded_dispatch_for_test(37, 64, 0), ((0, 1), false));

        // The enabled MeshOnly mesh stage still obeys its real total limit and reports genuine
        // truncation.
        assert_eq!(bounded_dispatch_for_test(37, 64, 64), ((1, 37), false));
        assert_eq!(bounded_dispatch_for_test(65, 64, 64), ((1, 64), true));
    }

    #[test]
    fn only_the_backend_consumed_indirect_dispatch_is_enabled() {
        let mesh_limit = 1_024;
        let task_limit = 2_048;
        assert_eq!(
            indirect_workgroup_total_limits(
                MeshletBackend::IndexedIndirect,
                mesh_limit,
                task_limit
            ),
            (0, 0)
        );
        assert_eq!(
            indirect_workgroup_total_limits(MeshletBackend::MeshOnly, mesh_limit, task_limit),
            (mesh_limit, 0)
        );
        assert_eq!(
            indirect_workgroup_total_limits(MeshletBackend::TaskMesh, mesh_limit, task_limit),
            (0, task_limit)
        );
    }

    #[test]
    fn compute_visible_backends_do_not_build_retired_task_packets() {
        for backend in [
            MeshletBackend::IndexedIndirect,
            MeshletBackend::MeshOnly,
            MeshletBackend::TaskMesh,
        ] {
            assert!(!task_packets_enabled(backend));
        }
    }

    #[test]
    fn projected_cube_conservatively_bounds_a_sphere_for_a_rotated_camera() {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0, 2.0, 6.0),
            glam::Vec3::new(0.25, -0.1, 0.0),
            glam::Vec3::Y,
        );
        let projection = glam::Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let view_projection = projection * view;
        let center = glam::Vec3::new(0.25, -0.1, 0.0);
        let radius = 1.25;
        let (minimum, maximum) =
            projected_cube_bounds(view_projection, center, radius).expect("sphere is in front");

        for latitude in 0..=32 {
            let phi = std::f32::consts::PI * latitude as f32 / 32.0;
            for longitude in 0..64 {
                let theta = std::f32::consts::TAU * longitude as f32 / 64.0;
                let direction =
                    glam::Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
                let clip = view_projection * (center + radius * direction).extend(1.0);
                let ndc = clip.truncate() / clip.w;
                assert!(ndc.cmpge(minimum - glam::Vec3::splat(1.0e-5)).all());
                assert!(ndc.cmple(maximum + glam::Vec3::splat(1.0e-5)).all());
            }
        }
    }

    #[test]
    fn projected_cube_rejects_occlusion_when_it_reaches_the_near_plane() {
        let projection = glam::Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        assert!(projected_cube_bounds(projection, glam::Vec3::new(0.0, 0.0, -0.05), 0.1).is_none());
    }

    #[test]
    fn focal_pixel_scale_is_invariant_under_camera_rotation() {
        let viewport_height = 1_080.0;
        let projection =
            glam::Mat4::perspective_rh(67.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1_000.0);
        let expected = 0.5 * viewport_height * projection.y_axis.y.abs();
        for eye in [
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(4.0, 2.0, 7.0),
            glam::Vec3::new(-3.0, 5.0, 2.0),
        ] {
            let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);
            let actual = vertical_focal_pixels_for_test(view, projection * view, viewport_height);
            assert!((actual - expected).abs() <= expected * 1.0e-5);
        }
    }

    #[test]
    fn gershgorin_scale_contains_sheared_sphere_directions() {
        let transform = glam::Mat3::from_cols(
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(2.5, 0.75, 0.0),
            glam::Vec3::new(-1.0, 0.5, 3.0),
        );
        let bound = gershgorin_scale_for_test(transform);
        for latitude in 0..=64 {
            let phi = std::f32::consts::PI * latitude as f32 / 64.0;
            for longitude in 0..128 {
                let theta = std::f32::consts::TAU * longitude as f32 / 128.0;
                let direction =
                    glam::Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
                assert!((transform * direction).length() <= bound * (1.0 + 1.0e-6));
            }
        }

        let orthogonal = glam::Mat3::from_diagonal(glam::Vec3::new(2.0, 3.0, 1.0));
        assert_eq!(gershgorin_scale_for_test(orthogonal), 3.0);
    }

    #[test]
    fn task_packet_compaction_matches_indexed_visibility_without_occlusion() {
        let visibility = (0..197_u32)
            .map(|index| index % 3 != 0 && index % 11 != 5)
            .collect::<Vec<_>>();
        let indexed = visibility
            .iter()
            .enumerate()
            .filter_map(|(index, &visible)| visible.then_some(index))
            .collect::<Vec<_>>();
        let mut task = Vec::new();
        for (packet_index, packet) in visibility
            .chunks(TASK_PACKET_MESHLET_COUNT as usize)
            .enumerate()
        {
            task.extend(packet.iter().enumerate().filter_map(|(lane, &visible)| {
                visible.then_some(packet_index * TASK_PACKET_MESHLET_COUNT as usize + lane)
            }));
        }
        assert_eq!(task, indexed);

        // If a defensive device-limit clamp ever lowers the task output count, only the payload
        // prefix is considered visible/output work; the dropped lanes are represented by overflow.
        let one_packet = [true; TASK_PACKET_MESHLET_COUNT as usize];
        let emitted = one_packet
            .iter()
            .enumerate()
            .filter_map(|(lane, &visible)| visible.then_some(lane))
            .take(7)
            .collect::<Vec<_>>();
        assert_eq!(emitted, (0..7).collect::<Vec<_>>());
    }
}

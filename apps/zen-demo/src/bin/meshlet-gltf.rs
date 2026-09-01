use std::{
    collections::BTreeSet,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use winit::{dpi::PhysicalSize, window::Window};
use zen_demo::{
    Example, FrameGraphSnapshotSource,
    device::{
        create_vulkan_instance, request_vulkan_legacy_device,
        request_vulkan_meshlet_device_configured,
    },
    gltf_loader::{LoadGltfOptions, LoadedGltfModel, load_gltf},
    meshlet_benchmark::{
        MESHLET_BENCHMARK_HEIGHT, MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES,
        MESHLET_BENCHMARK_WARMUP_FRAMES, MESHLET_BENCHMARK_WIDTH, MeshletAutoProfileFile,
        MeshletBenchmarkCollector, MeshletBenchmarkContext, MeshletFrameSampleNs,
    },
    meshlet_support::{
        DemoRenderer, MeshletDemoArgs, load_or_build_zenmesh, meshlet_cache_key,
        parse_meshlet_demo_args, raw_static_meshes,
    },
    orbit_camera_controller::{OrbitCameraController, OrbitCameraControllerOptions},
    rendering::{
        ForwardFrameComposer, ForwardRenderHost, MeshletForwardFrameComposer,
        MeshletForwardRenderHost,
    },
    run,
    surface_state::SurfaceState,
};
use zen_render::{GpuTimingReport, RenderFrameInput};
use zen_render_mesh::{
    Camera, MeshRenderInput, MeshRenderer, MeshletBuildConfig, MeshletGpuFrameTimings,
    MeshletRenderInput, MeshletRenderer, PerspectiveProjection, TextureSamplingConfig,
};

static DEMO_ARGS: OnceLock<MeshletDemoArgs> = OnceLock::new();

enum DemoRenderHost {
    Legacy(Box<ForwardRenderHost>),
    Meshlet(Box<MeshletForwardRenderHost>),
}

impl DemoRenderHost {
    fn set_gpu_debug_groups_enabled(&mut self, enabled: bool) {
        match self {
            Self::Legacy(host) => host.set_gpu_debug_groups_enabled(enabled),
            Self::Meshlet(host) => host.set_gpu_debug_groups_enabled(enabled),
        }
    }

    fn request_measurements(&mut self) {
        match self {
            Self::Legacy(host) => {
                host.composer_mut().mesh_mut().request_stats();
                host.request_gpu_timing();
            }
            Self::Meshlet(host) => {
                if !host.composer().meshlet().can_request_stats_immediately()
                    || !host.try_request_gpu_timing()
                {
                    return;
                }
                assert!(
                    host.composer_mut()
                        .meshlet_mut()
                        .request_stats_with_gpu_timing(),
                    "counter availability cannot change during a single-threaded request handshake"
                );
            }
        }
    }

    fn try_request_benchmark_measurements(&mut self) -> bool {
        match self {
            Self::Legacy(host) => host.try_request_gpu_timing(),
            Self::Meshlet(host) => {
                if !host.composer().meshlet().can_request_stats_immediately()
                    || !host.try_request_gpu_timing()
                {
                    return false;
                }
                assert!(
                    host.composer_mut().meshlet_mut().try_request_stats(),
                    "counter availability cannot change during a single-threaded request handshake"
                );
                true
            }
        }
    }

    fn uses_meshlet_renderer(&self) -> bool {
        matches!(self, Self::Meshlet(_))
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        surface: &wgpu::Texture,
        camera: Camera,
    ) {
        let result = match self {
            Self::Legacy(host) => host.render_frame(
                device,
                queue,
                RenderFrameInput::new(
                    frame_index,
                    surface,
                    MeshRenderInput {
                        camera,
                        debug_camera: None,
                        enable_occlusion_culling: true,
                    },
                ),
            ),
            Self::Meshlet(host) => host.render_frame(
                device,
                queue,
                RenderFrameInput::new(
                    frame_index,
                    surface,
                    MeshletRenderInput {
                        frame_index,
                        camera,
                        ..Default::default()
                    },
                ),
            ),
        };
        if let Err(error) = result {
            fatal(format_args!("FrameGraph rendering failed: {error}"));
        }
    }

    fn print_ready_stats(&mut self, device: &wgpu::Device) {
        match self {
            Self::Legacy(host) => {
                if let Some(stats) = host.composer_mut().mesh_mut().take_stats(device) {
                    println!(
                        "legacy stats: total={} visible={} drawn={}",
                        stats.total_instances, stats.visible_after_main_cull, stats.drawn_instances,
                    );
                }
            }
            Self::Meshlet(host) => {
                if let Some(stats) = host.composer_mut().meshlet_mut().take_stats(device) {
                    println!(
                        "meshlet stats: instances={} candidates={} visible={} (backface={} two-sided={}) culled(frustum={} cone={} hiz={}) overflow={:?}",
                        stats.total_instances,
                        stats.candidate_meshlets,
                        stats.visible_meshlets,
                        stats.visible_meshlets_per_bin.opaque_backface,
                        stats.visible_meshlets_per_bin.opaque_two_sided,
                        stats.frustum_culled_meshlets,
                        stats.normal_cone_culled_meshlets,
                        stats.hiz_culled_meshlets,
                        stats.overflow,
                    );
                }
            }
        }
    }

    fn take_meshlet_stats(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<zen_render_mesh::MeshletRenderStats> {
        match self {
            Self::Legacy(_) => None,
            Self::Meshlet(host) => host.composer_mut().meshlet_mut().take_stats(device),
        }
    }

    fn take_gpu_timing(&mut self) -> Option<GpuTimingReport> {
        match self {
            Self::Legacy(host) => host.take_gpu_timing(),
            Self::Meshlet(host) => host.take_gpu_timing(),
        }
    }

    fn associate_gpu_timing(&mut self, timing: &GpuTimingReport) {
        if let Self::Meshlet(host) = self
            && let Err(error) = host
                .composer_mut()
                .meshlet_mut()
                .associate_gpu_timing(timing)
        {
            eprintln!("Meshlet stats timing association failed: {error}");
        }
    }

    fn take_cpu_timings(&mut self) -> Option<(Duration, Duration)> {
        let timings = match self {
            Self::Legacy(host) => host.take_cpu_timings(),
            Self::Meshlet(host) => host.take_cpu_timings(),
        }?;
        Some((timings.encode, timings.submit))
    }

    fn snapshot_source(&mut self) -> &mut dyn FrameGraphSnapshotSource {
        match self {
            Self::Legacy(host) => host.as_mut(),
            Self::Meshlet(host) => host.as_mut(),
        }
    }
}

struct PendingBenchmarkSample {
    frame_index: u64,
    cpu_frame_ns: u64,
    cpu_encode_ns: u64,
    cpu_submit_ns: u64,
}

struct ActiveBenchmark {
    collector: Option<MeshletBenchmarkCollector>,
    output: PathBuf,
    pending: Option<PendingBenchmarkSample>,
    expected_stats: BTreeSet<u64>,
    completed: bool,
}

impl ActiveBenchmark {
    fn new(collector: MeshletBenchmarkCollector, output: PathBuf) -> Self {
        Self {
            collector: Some(collector),
            output,
            pending: None,
            expected_stats: BTreeSet::new(),
            completed: false,
        }
    }

    fn collector(&self) -> &MeshletBenchmarkCollector {
        self.collector
            .as_ref()
            .expect("completed benchmark no longer has a collector")
    }

    fn collector_mut(&mut self) -> &mut MeshletBenchmarkCollector {
        self.collector
            .as_mut()
            .expect("completed benchmark no longer has a collector")
    }

    fn track_frame(&self) -> u32 {
        let collector = self.collector();
        collector
            .warmup_frames_recorded()
            .saturating_add(collector.sample_frames_recorded() as u32)
    }

    fn is_warming_up(&self) -> bool {
        self.collector().warmup_frames_recorded() < MESHLET_BENCHMARK_WARMUP_FRAMES
    }

    fn should_capture(&self) -> bool {
        !self.is_warming_up() && !self.collector().is_ready() && self.pending.is_none()
    }
}

struct Demo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: SurfaceState,
    render_host: DemoRenderHost,
    projection: PerspectiveProjection,
    camera: Camera,
    camera_controller: OrbitCameraController,
    frame_index: u64,
    benchmark_target: Option<wgpu::Texture>,
    benchmark: Option<ActiveBenchmark>,
    model_center: glam::Vec3,
    model_radius: f32,
}

impl Example for Demo {
    const NAME: &'static str = "Vulkan Meshlet GPU-Driven";

    fn window_attributes() -> winit::window::WindowAttributes {
        Window::default_attributes()
            .with_title(Self::NAME)
            .with_inner_size(PhysicalSize::new(1920, 1080))
    }

    async fn init(window: Arc<Window>) -> Self {
        let arguments = DEMO_ARGS.get().cloned().unwrap_or_default();
        let instance = create_vulkan_instance();
        let size = window.inner_size();
        let surface = instance.create_surface(window).unwrap_or_else(|error| {
            fatal(format_args!("failed to create Vulkan surface: {error}"))
        });

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_dir = manifest_dir
            .parent()
            .and_then(Path::parent)
            .unwrap_or_else(|| fatal("zen-demo must be located under apps/ in the workspace"));
        let default_model = manifest_dir
            .join("assets")
            .join("DamagedHelmet")
            .join("glTF")
            .join("DamagedHelmet.gltf");
        let model_path = arguments
            .model
            .as_deref()
            .map(|path| resolve_model_path(workspace_dir, manifest_dir, path))
            .unwrap_or(default_model);
        let model = load_gltf(
            &model_path,
            LoadGltfOptions {
                global_scale: 1.0,
                flip_v: false,
                bake_node_transform: false,
            },
        )
        .unwrap_or_else(|error| {
            fatal(format_args!(
                "failed to load {}: {error}",
                model_path.display()
            ))
        });
        let (center, radius) = compute_model_bounds(&model);

        let needs_meshlet_source = arguments.renderer != DemoRenderer::Legacy
            || arguments.benchmark_out.is_some()
            || arguments.auto_profile.is_some();
        let raw_meshes = needs_meshlet_source
            .then(|| raw_static_meshes(&model).unwrap_or_else(|error| fatal(error)));
        let scene_identity = raw_meshes.as_ref().map(|meshes| {
            benchmark_scene_identity(
                &model,
                meshes,
                MeshletBuildConfig::default(),
                TextureSamplingConfig::default(),
            )
        });
        let auto_profile = arguments.auto_profile.as_ref().map(|path| {
            MeshletAutoProfileFile::read_json_file(path)
                .unwrap_or_else(|error| fatal(format_args!("failed to load Auto profile: {error}")))
        });

        let (device, queue, adapter_info, meshlet_requirements, meshlet_config, device_sampling) =
            if let Some(config) = arguments.renderer.meshlet_config() {
                let expected_scene = scene_identity
                    .as_deref()
                    .expect("meshlet paths always compute a scene identity");
                let requested = request_vulkan_meshlet_device_configured(
                    &instance,
                    &surface,
                    config,
                    |adapter, config| {
                        if let Some(profile) = auto_profile.as_ref() {
                            match profile.validate_for(adapter, expected_scene) {
                                Ok(()) => {
                                    config.auto_benchmark_profile =
                                        Some(profile.renderer_profile());
                                    println!(
                                        "Loaded qualifying Auto profile for {} ({})",
                                        adapter.name, adapter.driver_info
                                    );
                                }
                                Err(error) => println!(
                                    "Ignoring stale or non-qualifying Auto profile; IndexedIndirect remains the default: {error}"
                                ),
                            }
                        }
                        Ok(())
                    },
                )
                .await
                .unwrap_or_else(|error| fatal(error));
                (
                    requested.device,
                    requested.queue,
                    requested.adapter_info,
                    Some(requested.requirements),
                    Some(requested.config),
                    requested.sampling,
                )
            } else {
                let requested = request_vulkan_legacy_device(&instance, &surface).await;
                (
                    requested.device,
                    requested.queue,
                    requested.adapter_info,
                    None,
                    None,
                    requested.sampling,
                )
            };
        let sampling = if arguments.benchmark_out.is_some() {
            TextureSamplingConfig::default()
        } else {
            device_sampling
        };
        if arguments.benchmark_out.is_some() {
            let required_timing =
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            if !device.features().contains(required_timing) {
                fatal(format_args!(
                    "benchmark requires complete GPU timing coverage ({required_timing:?}); device exposes {:?}",
                    device.features()
                ));
            }
        }
        let surface = SurfaceState::new(&device, surface, size.width, size.height);

        let mut render_host = match arguments.renderer {
            DemoRenderer::Legacy => {
                let mesh = MeshRenderer::new(
                    &device,
                    &queue,
                    surface.format(),
                    &model.meshes,
                    &model.materials,
                    &model.instances,
                    &model.textures,
                    &model.samplers,
                    sampling,
                )
                .unwrap_or_else(|error| fatal(error));
                DemoRenderHost::Legacy(Box::new(ForwardRenderHost::new(
                    &device,
                    ForwardFrameComposer::new(mesh, surface.format()),
                )))
            }
            _ => {
                let (Some(config), Some(requirements)) =
                    (meshlet_config, meshlet_requirements.as_ref())
                else {
                    fatal("meshlet renderer choice was not resolved during device creation");
                };
                let raw_meshes = raw_meshes
                    .as_deref()
                    .expect("meshlet paths always prepare static source meshes");
                let cache_path = arguments.cache.clone().unwrap_or_else(|| {
                    default_cache_path(&model_path, raw_meshes, MeshletBuildConfig::default())
                });
                let cached =
                    load_or_build_zenmesh(&cache_path, raw_meshes, MeshletBuildConfig::default())
                        .unwrap_or_else(|error| fatal(error));
                println!(
                    "Meshlet asset {:?}: {} (meshes={}, lods={}, meshlets={})",
                    cached.status,
                    cache_path.display(),
                    cached.asset.meshes().len(),
                    cached.asset.lods().len(),
                    cached.asset.meshlets().len(),
                );
                let meshlet = MeshletRenderer::new(
                    &device,
                    &queue,
                    surface.format(),
                    config,
                    requirements,
                    &cached.asset,
                    &model.materials,
                    &model.instances,
                    &model.textures,
                    &model.samplers,
                    sampling,
                )
                .unwrap_or_else(|error| fatal(error));
                DemoRenderHost::Meshlet(Box::new(MeshletForwardRenderHost::new(
                    &device,
                    MeshletForwardFrameComposer::new(meshlet, surface.format()),
                )))
            }
        };
        render_host.set_gpu_debug_groups_enabled(true);

        let benchmark_target = arguments.benchmark_out.as_ref().map(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("meshlet-benchmark-1920x1080"),
                size: wgpu::Extent3d {
                    width: MESHLET_BENCHMARK_WIDTH,
                    height: MESHLET_BENCHMARK_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: surface.format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        });
        let benchmark = arguments.benchmark_out.clone().map(|output| {
            let renderer = arguments
                .renderer
                .benchmark_renderer()
                .expect("argument validation rejects Auto benchmark runs");
            let context = MeshletBenchmarkContext::from_vulkan_adapter(
                &adapter_info,
                scene_identity
                    .as_deref()
                    .expect("benchmark paths always compute a scene identity"),
            );
            println!(
                "Benchmark: fixed {}x{}, warmup={}, samples={}, camera={}, output={}",
                MESHLET_BENCHMARK_WIDTH,
                MESHLET_BENCHMARK_HEIGHT,
                MESHLET_BENCHMARK_WARMUP_FRAMES,
                MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES,
                context.camera_track,
                output.display(),
            );
            ActiveBenchmark::new(
                MeshletBenchmarkCollector::new(renderer, context, arguments.geometry_bound),
                output,
            )
        });
        let projection = PerspectiveProjection {
            aspect: if benchmark.is_some() {
                MESHLET_BENCHMARK_WIDTH as f32 / MESHLET_BENCHMARK_HEIGHT as f32
            } else {
                surface.width() as f32 / surface.height() as f32
            },
            fovy_deg: 45.0,
            near: 0.1,
            far: 1000.0,
        };
        let camera_position = center + glam::vec3(0.0, 0.0, radius.max(0.01) * 3.0);
        let camera = Camera::new(
            glam::Mat4::look_at_rh(camera_position, center, glam::Vec3::Y).inverse(),
            projection,
        );
        let camera_controller = OrbitCameraController::new(OrbitCameraControllerOptions {
            target: center,
            position: Some(camera_position),
            ..Default::default()
        });

        println!("Started renderer path: {}", arguments.renderer);
        Self {
            device,
            queue,
            surface,
            render_host,
            projection,
            camera,
            camera_controller,
            frame_index: 0,
            benchmark_target,
            benchmark,
            model_center: center,
            model_radius: radius,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.surface.resize(width, height);
        if self.benchmark.is_some() {
            return;
        }
        if width == 0 || height == 0 {
            return;
        }
        self.projection.aspect = width as f32 / height as f32;
        self.camera.set_projection(self.projection);
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        self.frame_index += 1;
        if self.benchmark.is_some() {
            self.render_benchmark_frame();
            return;
        }
        if self.frame_index.is_multiple_of(120) {
            self.render_host.request_measurements();
        }
        let Some(surface_texture) = self.surface.acquire(&self.device) else {
            return;
        };
        self.render_host.render(
            &self.device,
            &self.queue,
            self.frame_index,
            &surface_texture.texture,
            self.camera,
        );
        self.queue.present(surface_texture);
        if let Some(timing) = self.render_host.take_gpu_timing() {
            self.render_host.associate_gpu_timing(&timing);
            print_gpu_timing(timing);
        }
        self.render_host.print_ready_stats(&self.device);
    }

    fn mouse_drag(&mut self, dx: f32, dy: f32) {
        if self.benchmark.is_some() {
            return;
        }
        self.camera_controller.orbit(dx * 0.01, dy * 0.01);
        self.camera.set_view(self.camera_controller.view_matrix());
    }

    fn mouse_wheel(&mut self, delta_y: f32) {
        if self.benchmark.is_some() {
            return;
        }
        self.camera_controller.dolly(delta_y);
        self.camera.set_view(self.camera_controller.view_matrix());
    }

    fn frame_graph_snapshot_source(&mut self) -> Option<&mut dyn FrameGraphSnapshotSource> {
        Some(self.render_host.snapshot_source())
    }

    fn should_exit(&self) -> bool {
        self.benchmark
            .as_ref()
            .is_some_and(|benchmark| benchmark.completed)
    }
}

impl Demo {
    fn render_benchmark_frame(&mut self) {
        let (track_frame, warming_up, wants_capture) = {
            let benchmark = self.benchmark.as_ref().unwrap();
            (
                benchmark.track_frame(),
                benchmark.is_warming_up(),
                benchmark.should_capture(),
            )
        };
        self.camera.set_view(benchmark_view(
            self.model_center,
            self.model_radius,
            track_frame,
        ));
        let capture = wants_capture && self.render_host.try_request_benchmark_measurements();

        let target = self
            .benchmark_target
            .as_ref()
            .expect("active benchmark has a fixed offscreen target");
        let cpu_frame_start = Instant::now();
        self.render_host.render(
            &self.device,
            &self.queue,
            self.frame_index,
            target,
            self.camera,
        );
        let cpu_frame_ns = duration_ns(cpu_frame_start.elapsed());
        let (cpu_encode, cpu_submit) = self
            .render_host
            .take_cpu_timings()
            .unwrap_or_else(|| fatal("successful benchmark frame did not expose CPU timings"));
        let cpu_encode_ns = duration_ns(cpu_encode);
        let cpu_submit_ns = duration_ns(cpu_submit);

        if warming_up {
            self.benchmark
                .as_mut()
                .unwrap()
                .collector_mut()
                .record_frame(MeshletFrameSampleNs::split(
                    cpu_frame_ns,
                    cpu_encode_ns,
                    cpu_submit_ns,
                    0,
                ));
            if self
                .benchmark
                .as_ref()
                .unwrap()
                .collector()
                .warmup_frames_recorded()
                == MESHLET_BENCHMARK_WARMUP_FRAMES
            {
                println!("Benchmark warm-up complete; collecting GPU timestamp samples");
            }
        } else if capture {
            let benchmark = self.benchmark.as_mut().unwrap();
            benchmark.pending = Some(PendingBenchmarkSample {
                frame_index: self.frame_index,
                cpu_frame_ns,
                cpu_encode_ns,
                cpu_submit_ns,
            });
            if self.render_host.uses_meshlet_renderer() {
                assert!(benchmark.expected_stats.insert(self.frame_index));
            }
        }

        if let Some(timing) = self.render_host.take_gpu_timing() {
            self.accept_benchmark_gpu_timing(timing);
        }
        while let Some(stats) = self.render_host.take_meshlet_stats(&self.device) {
            let benchmark = self.benchmark.as_mut().unwrap();
            if !benchmark.expected_stats.remove(&stats.frame_index) {
                fatal(format_args!(
                    "counter readback for unexpected benchmark frame {}",
                    stats.frame_index
                ));
            }
            if !stats.overflow.is_empty() {
                fatal(format_args!(
                    "benchmark frame {} overflowed meshlet GPU capacity: {:?}",
                    stats.frame_index, stats.overflow
                ));
            }
        }
        self.try_finish_benchmark();
    }

    fn accept_benchmark_gpu_timing(&mut self, timing: GpuTimingReport) {
        let decoded = MeshletGpuFrameTimings::from_gpu_timing_report(&timing)
            .unwrap_or_else(|error| fatal(format_args!("GPU timestamp sample failed: {error}")));
        let frame_index = decoded.frame_index;
        let gpu_frame_ns = decoded.frame_total_ns;
        let benchmark = self.benchmark.as_mut().unwrap();
        let pending = benchmark
            .pending
            .take()
            .unwrap_or_else(|| fatal("GPU timing completed without a pending benchmark sample"));
        if pending.frame_index != frame_index {
            fatal(format_args!(
                "GPU timing frame mismatch: expected {}, got {frame_index}",
                pending.frame_index
            ));
        }
        benchmark.collector_mut().record_frame(
            MeshletFrameSampleNs::split(
                pending.cpu_frame_ns,
                pending.cpu_encode_ns,
                pending.cpu_submit_ns,
                gpu_frame_ns,
            )
            .with_gpu_passes(decoded.passes),
        );
        let samples = benchmark.collector().sample_frames_recorded();
        if samples.is_multiple_of(60) || samples == MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES as usize {
            println!(
                "Benchmark samples: {samples}/{}",
                MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES
            );
        }
    }

    fn try_finish_benchmark(&mut self) {
        let ready = self.benchmark.as_ref().is_some_and(|benchmark| {
            benchmark.collector().is_ready()
                && benchmark.pending.is_none()
                && benchmark.expected_stats.is_empty()
        });
        if !ready {
            return;
        }
        let benchmark = self.benchmark.as_mut().unwrap();
        let report = benchmark
            .collector
            .take()
            .expect("ready benchmark has a collector")
            .finish()
            .unwrap_or_else(|error| fatal(error));
        report
            .write_json_file(&benchmark.output)
            .unwrap_or_else(|error| fatal(error));
        println!(
            "Benchmark complete: renderer={}, GPU median={}ns p95={}ns, JSON={}",
            report.renderer,
            report.gpu_frame_ns.median,
            report.gpu_frame_ns.p95,
            benchmark.output.display(),
        );
        benchmark.completed = true;
    }
}

fn benchmark_view(center: glam::Vec3, radius: f32, frame: u32) -> glam::Mat4 {
    let track_length = MESHLET_BENCHMARK_WARMUP_FRAMES + MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES;
    let phase = (frame % track_length) as f32 / track_length as f32;
    let azimuth = phase * std::f32::consts::TAU;
    let elevation = 0.22 + (phase * std::f32::consts::TAU * 2.0).sin() * 0.08;
    let distance = radius.max(0.01) * 3.0;
    let horizontal = elevation.cos() * distance;
    let position = center
        + glam::vec3(
            azimuth.sin() * horizontal,
            elevation.sin() * distance,
            azimuth.cos() * horizontal,
        );
    glam::Mat4::look_at_rh(position, center, glam::Vec3::Y).inverse()
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn default_cache_path(
    model_path: &Path,
    meshes: &[zen_render_mesh::meshlet::RawStaticMesh],
    config: MeshletBuildConfig,
) -> PathBuf {
    let key = meshlet_cache_key(meshes, config);
    let stem = model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("scene");
    std::env::temp_dir()
        .join("zen-proto")
        .join("zenmesh-v1")
        .join(format!(
            "{stem}-{}-{}.zenmesh",
            key.source_hash, key.build_hash
        ))
}

fn benchmark_scene_identity(
    model: &LoadedGltfModel,
    meshes: &[zen_render_mesh::meshlet::RawStaticMesh],
    config: MeshletBuildConfig,
    sampling: TextureSamplingConfig,
) -> String {
    let key = meshlet_cache_key(meshes, config);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.source_hash.to_string().hash(&mut hasher);
    key.build_hash.to_string().hash(&mut hasher);
    model.instances.len().hash(&mut hasher);
    for instance in &model.instances {
        instance.mesh_id.hash(&mut hasher);
        instance.material_id.hash(&mut hasher);
        for component in instance.transform.to_cols_array() {
            component.to_bits().hash(&mut hasher);
        }
    }
    model.materials.len().hash(&mut hasher);
    for material in &model.materials {
        for component in material.albedo_factor.to_array() {
            component.to_bits().hash(&mut hasher);
        }
        for component in material.emissive_ao.to_array() {
            component.to_bits().hash(&mut hasher);
        }
        material.albedo.texture_id.hash(&mut hasher);
        material.albedo.sampler_id.hash(&mut hasher);
        material.emissive.texture_id.hash(&mut hasher);
        material.emissive.sampler_id.hash(&mut hasher);
        material.occlusion.texture_id.hash(&mut hasher);
        material.occlusion.sampler_id.hash(&mut hasher);
    }
    model.samplers.hash(&mut hasher);
    sampling.hash(&mut hasher);
    model.textures.len().hash(&mut hasher);
    for texture in &model.textures {
        texture.width.hash(&mut hasher);
        texture.height.hash(&mut hasher);
        texture.format.hash(&mut hasher);
        texture.pixels.hash(&mut hasher);
    }
    format!(
        "zenmesh-v1:{}:{}:scene-{:016x}",
        key.source_hash,
        key.build_hash,
        hasher.finish()
    )
}

fn compute_model_bounds(model: &LoadedGltfModel) -> (glam::Vec3, f32) {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for instance in &model.instances {
        let Some(mesh) = model.meshes.get(instance.mesh_id as usize) else {
            continue;
        };
        for vertex in &mesh.vertices {
            let point = instance
                .transform
                .transform_point3(vertex.position.truncate());
            min = min.min(point);
            max = max.max(point);
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return (glam::Vec3::ZERO, 1.0);
    }
    let center = (min + max) * 0.5;
    (center, ((max - min) * 0.5).length().max(0.01))
}

fn resolve_model_path(workspace_dir: &Path, manifest_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else if let Ok(asset_path) = path.strip_prefix("assets") {
        manifest_dir.join("assets").join(asset_path)
    } else {
        workspace_dir.join(path)
    }
}

fn print_gpu_timing(timing: GpuTimingReport) {
    match timing {
        GpuTimingReport::Available {
            frame_index,
            frame_duration,
            nodes,
            ..
        } => {
            println!("GPU timing frame {frame_index}: total {frame_duration:?}");
            for node in nodes {
                println!("  {} ({:?}): {:?}", node.label, node.kind, node.duration);
            }
        }
        GpuTimingReport::Unavailable {
            frame_index,
            reason,
        } => println!("GPU timing frame {frame_index} unavailable: {reason:?}"),
        _ => println!("GPU timing report uses an unknown future format"),
    }
}

fn fatal(error: impl std::fmt::Display) -> ! {
    eprintln!("meshlet-gltf: {error}");
    process::exit(1)
}

fn main() {
    let arguments = parse_meshlet_demo_args(std::env::args_os().skip(1))
        .unwrap_or_else(|error| fatal(format_args!("{error}\nusage: meshlet-gltf [MODEL] [--renderer legacy|indexed|mesh|task-mesh|auto] [--cache PATH] [--benchmark-out REPORT.json] [--geometry-bound] [--auto-profile PROFILE.json]")));
    let benchmark = arguments.benchmark_out.is_some();
    if DEMO_ARGS.set(arguments).is_err() {
        fatal("demo arguments were initialized more than once");
    }
    run::<Demo>(benchmark.then_some(u8::MAX));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_cache_path_is_keyed_by_source_and_build() {
        let mesh = zen_render_mesh::meshlet::RawStaticMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        );
        let first = default_cache_path(
            Path::new("scene.gltf"),
            std::slice::from_ref(&mesh),
            MeshletBuildConfig::default(),
        );
        let changed = default_cache_path(
            Path::new("scene.gltf"),
            &[zen_render_mesh::meshlet::RawStaticMesh::new(
                mesh.positions,
                vec![0, 2, 1],
            )],
            MeshletBuildConfig::default(),
        );
        assert_ne!(first, changed);
        assert_eq!(
            first.extension().and_then(|value| value.to_str()),
            Some("zenmesh")
        );
    }

    #[test]
    fn model_path_resolution_matches_the_legacy_demo() {
        let workspace = Path::new("workspace");
        let manifest = workspace.join("apps/zen-demo");
        assert_eq!(
            resolve_model_path(workspace, &manifest, Path::new("assets/scene/model.gltf")),
            manifest.join("assets/scene/model.gltf")
        );
        assert_eq!(
            resolve_model_path(workspace, &manifest, Path::new("models/a.glb")),
            workspace.join("models/a.glb")
        );
    }

    #[test]
    fn benchmark_camera_track_is_deterministic_finite_and_periodic() {
        let center = glam::vec3(1.0, 2.0, 3.0);
        let first = benchmark_view(center, 2.0, 137);
        assert_eq!(first, benchmark_view(center, 2.0, 137));
        assert_eq!(
            first,
            benchmark_view(
                center,
                2.0,
                137 + MESHLET_BENCHMARK_WARMUP_FRAMES + MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES,
            )
        );
        assert!(first.is_finite());
        assert_ne!(first, benchmark_view(center, 2.0, 138));
    }
}

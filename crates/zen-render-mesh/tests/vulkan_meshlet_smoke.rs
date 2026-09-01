//! Opt-in real-hardware smoke test for legacy plus all three Vulkan meshlet backends.
//!
//! Run with:
//! `cargo test -p zen-render-mesh --test vulkan_meshlet_smoke -- --ignored --nocapture`

use std::sync::mpsc;

use zen_frame_graph::{
    CompileOptions, FrameGraph, ImportTextureOptions, InitialContents, RootReason, TextureDesc,
    UsagePolicy,
};
use zen_render_mesh::{
    Camera, Instance, Material, MaterialTextureBinding, Mesh, MeshRenderInput, MeshRenderTargets,
    MeshRenderer, MeshletBackend, MeshletBindlessConfig, MeshletCapabilities,
    MeshletCapacityConfig, MeshletRenderInput, MeshletRenderMode, MeshletRenderer,
    MeshletRendererConfig, MeshletSceneAsset, RawStaticMesh, Texture, TextureAddressMode,
    TextureMagFilter, TextureMinFilter, TextureSampler, TextureSamplingConfig, Vertex,
};

const EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: 256,
    height: 256,
    depth_or_array_layers: 1,
};
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[derive(Clone, Copy)]
struct SamplingCase {
    name: &'static str,
    address: TextureAddressMode,
    uv: [f32; 2],
    expected_channel: usize,
}

const SAMPLING_CASES: [SamplingCase; 3] = [
    SamplingCase {
        name: "repeat",
        address: TextureAddressMode::Repeat,
        uv: [1.1, 0.1],
        expected_channel: 0,
    },
    SamplingCase {
        name: "clamp",
        address: TextureAddressMode::ClampToEdge,
        uv: [1.1, 0.1],
        expected_channel: 1,
    },
    SamplingCase {
        name: "mirrored-repeat",
        address: TextureAddressMode::MirroredRepeat,
        uv: [1.1, 0.1],
        expected_channel: 1,
    },
];

#[test]
#[ignore = "requires a Vulkan adapter with descriptor indexing and EXT_mesh_shader"]
fn multiple_meshlets_render_through_indexed_mesh_and_task_mesh_backends() {
    pollster::block_on(run());
}

async fn run() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("no Vulkan adapter available for meshlet hardware smoke test");
    eprintln!("Vulkan meshlet smoke adapter: {:?}", adapter.get_info());

    for instance_count in [2, 33] {
        for sampling_case in SAMPLING_CASES {
            let reference = render_legacy(&adapter, instance_count, sampling_case).await;
            let mut debug_reference: Option<Vec<FrameCapture>> = None;
            for backend in [
                MeshletBackend::IndexedIndirect,
                MeshletBackend::MeshOnly,
                MeshletBackend::TaskMesh,
            ] {
                let captures = render_instances(
                    &adapter,
                    backend,
                    instance_count,
                    sampling_case,
                    MeshletRenderMode::Shaded,
                )
                .await;
                compare_captures(&reference, &captures, backend, instance_count);
                let debug_captures = render_instances(
                    &adapter,
                    backend,
                    instance_count,
                    sampling_case,
                    MeshletRenderMode::MeshletId,
                )
                .await;
                if let Some(reference) = debug_reference.as_ref() {
                    compare_captures(reference, &debug_captures, backend, instance_count);
                } else {
                    debug_reference = Some(debug_captures);
                }
                eprintln!(
                    "Vulkan meshlet smoke passed: {backend}, shaded+meshlet-id, sampling={}, instances={instance_count}",
                    sampling_case.name
                );
            }
            compare_captures(
                &reference[0..1],
                &reference[1..2],
                MeshletBackend::IndexedIndirect,
                instance_count,
            );
        }
    }
}

async fn render_legacy(
    adapter: &wgpu::Adapter,
    instance_count: u32,
    sampling_case: SamplingCase,
) -> Vec<FrameCapture> {
    let features = MeshRenderer::required_features();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("zen.meshlet.vulkan-smoke.legacy"),
            required_features: features,
            required_limits: MeshRenderer::required_limits(&adapter.limits()),
            ..Default::default()
        })
        .await
        .expect("failed to request legacy smoke device");
    let uv = glam::Vec2::from_array(sampling_case.uv);
    let mesh = Mesh {
        vertices: vec![
            Vertex {
                position: glam::Vec4::new(-0.65, -0.55, 0.0, 1.0),
                normal: glam::Vec4::Z,
                color: glam::Vec4::ONE,
                uv,
            },
            Vertex {
                position: glam::Vec4::new(0.65, -0.55, 0.0, 1.0),
                normal: glam::Vec4::Z,
                color: glam::Vec4::ONE,
                uv,
            },
            Vertex {
                position: glam::Vec4::new(0.0, 0.65, 0.0, 1.0),
                normal: glam::Vec4::Z,
                color: glam::Vec4::ONE,
                uv,
            },
        ],
        indices: vec![0, 1, 2],
    };
    let instances = smoke_instances(instance_count);
    let (materials, textures, samplers) = sampling_resources(sampling_case.address);
    let mut renderer = MeshRenderer::new(
        &device,
        &queue,
        COLOR_FORMAT,
        &[mesh],
        &materials,
        &instances,
        &textures,
        &samplers,
        TextureSamplingConfig::default(),
    )
    .unwrap();
    render_legacy_captures(
        &device,
        &queue,
        &mut renderer,
        instance_count,
        sampling_case.expected_channel,
    )
}

fn render_legacy_captures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut MeshRenderer,
    instance_count: u32,
    expected_channel: usize,
) -> Vec<FrameCapture> {
    let color = create_texture(
        device,
        "meshlet.smoke.legacy.color",
        COLOR_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let depth = create_texture(
        device,
        "meshlet.smoke.legacy.depth",
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    );
    let camera = Camera::default();
    let mut captures = Vec::new();
    for enable_occlusion_culling in [false, true] {
        let mut graph = FrameGraph::with_device(device);
        let mut frame = graph.begin_frame();
        let color_handle = import_texture(&mut frame, &color, InitialContents::Undefined);
        let depth_handle = import_texture(&mut frame, &depth, InitialContents::Undefined);
        let prepared = renderer.prepare_frame(
            queue,
            MeshRenderInput {
                camera,
                debug_camera: None,
                enable_occlusion_culling,
            },
            EXTENT,
        );
        renderer
            .record_frame_graph(
                &mut frame,
                MeshRenderTargets::new(color_handle, depth_handle),
                &prepared,
            )
            .unwrap();
        frame
            .mark_texture_root(color_handle, RootReason::DebugCapture)
            .unwrap();
        frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute(queue)
            .unwrap();
        renderer.after_submit(device, prepared);
        let color = read_rgba8(device, queue, &color);
        let depth = read_depth32(device, queue, &depth);
        assert_instance_center_probes(
            &color,
            &depth,
            camera,
            instance_count,
            MeshletBackend::IndexedIndirect,
            expected_channel,
        );
        captures.push(FrameCapture { color, depth });
    }
    captures
}

#[derive(Clone)]
struct FrameCapture {
    color: Vec<u8>,
    depth: Vec<f32>,
}

async fn render_instances(
    adapter: &wgpu::Adapter,
    backend: MeshletBackend,
    instance_count: u32,
    sampling_case: SamplingCase,
    render_mode: MeshletRenderMode,
) -> Vec<FrameCapture> {
    let config = MeshletRendererConfig {
        backend,
        bindless: MeshletBindlessConfig {
            max_textures: 8,
            max_samplers: 4,
        },
        capacities: MeshletCapacityConfig {
            max_instances: 64,
            max_candidate_meshlets: 128,
            max_visible_meshlets: 128,
            max_task_packets: 64,
            max_indirect_draws_per_bin: 64,
        },
        auto_benchmark_profile: None,
    };
    let capabilities = MeshletCapabilities::from_adapter(adapter);
    let requirements = capabilities
        .device_requirements(&config)
        .unwrap_or_else(|error| panic!("{backend} is unavailable on the smoke adapter: {error}"));
    // SAFETY: This opt-in hardware test explicitly exercises wgpu's experimental mesh/task API.
    let experimental_features = unsafe { requirements.experimental_features_token() };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("zen.meshlet.vulkan-smoke"),
            required_features: requirements.features(),
            required_limits: requirements.limits().clone(),
            experimental_features,
            ..Default::default()
        })
        .await
        .unwrap_or_else(|error| panic!("failed to request {backend} smoke device: {error}"));

    let mut source = RawStaticMesh::new(
        vec![[-0.65, -0.55, 0.0], [0.65, -0.55, 0.0], [0.0, 0.65, 0.0]],
        vec![0, 1, 2],
    );
    source.normals = vec![[0.0, 0.0, 1.0]; 3];
    source.tex_coords = vec![sampling_case.uv; 3];
    source.colors = vec![[1.0; 4]; 3];
    let asset = MeshletSceneAsset::build(&[source], Default::default()).unwrap();
    let instances = smoke_instances(instance_count);
    let (materials, textures, samplers) = sampling_resources(sampling_case.address);
    let mut renderer = MeshletRenderer::new(
        &device,
        &queue,
        COLOR_FORMAT,
        config,
        &requirements,
        &asset,
        &materials,
        &instances,
        &textures,
        &samplers,
        TextureSamplingConfig::default(),
    )
    .unwrap();

    let color = create_texture(
        &device,
        "meshlet.smoke.color",
        COLOR_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let depth = create_texture(
        &device,
        "meshlet.smoke.depth",
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    );
    let mut captures = Vec::new();
    let camera = Camera::default();
    for enable_occlusion_culling in [false, true] {
        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let color_handle = import_texture(&mut frame, &color, InitialContents::Undefined);
        let depth_handle = import_texture(&mut frame, &depth, InitialContents::Undefined);
        let prepared = renderer.prepare_frame(
            &queue,
            MeshletRenderInput {
                camera,
                enable_occlusion_culling,
                render_mode,
                ..Default::default()
            },
            EXTENT,
        );
        renderer
            .record_frame_graph(
                &mut frame,
                MeshRenderTargets::new(color_handle, depth_handle),
                &prepared,
            )
            .unwrap();
        frame
            .mark_texture_root(color_handle, RootReason::DebugCapture)
            .unwrap();
        frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute(&queue)
            .unwrap();
        renderer.after_submit(&device, prepared);

        let pixels = read_rgba8(&device, &queue, &color);
        let pixel_chunks = pixels.as_chunks::<4>().0;
        let corner_pixel = pixel_chunks[0];
        let drawn_pixels = pixel_chunks
            .iter()
            .filter(|pixel| **pixel != corner_pixel)
            .count();
        assert!(
            drawn_pixels >= instance_count as usize,
            "{backend} rendered only {drawn_pixels} non-clear pixels for {instance_count} instances (occlusion={enable_occlusion_culling})"
        );
        let depths = read_depth32(&device, &queue, &depth);
        assert_instance_center_probes(
            &pixels,
            &depths,
            camera,
            instance_count,
            backend,
            if render_mode == MeshletRenderMode::MeshletId {
                0
            } else {
                sampling_case.expected_channel
            },
        );
        captures.push(FrameCapture {
            color: pixels,
            depth: depths,
        });
    }
    captures
}

fn smoke_instances(instance_count: u32) -> Vec<Instance> {
    (0..instance_count)
        .map(|index| {
            let translation = instance_translation(index);
            Instance {
                transform: glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::splat(0.16),
                    glam::Quat::IDENTITY,
                    translation,
                ),
                mesh_id: 0,
                material_id: 0,
                _pad: [0; 2],
            }
        })
        .collect()
}

fn sampling_resources(
    address: TextureAddressMode,
) -> (Vec<Material>, Vec<Texture>, Vec<TextureSampler>) {
    let binding = MaterialTextureBinding {
        texture_id: 0,
        sampler_id: 0,
    };
    let materials = vec![Material {
        albedo_factor: glam::Vec4::ONE,
        emissive_ao: glam::Vec4::W,
        albedo: binding,
        emissive: binding,
        occlusion: binding,
        _padding: [0; 2],
    }];
    let textures = vec![Texture {
        width: 2,
        height: 2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        pixels: vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ],
    }];
    let samplers = vec![TextureSampler {
        address_mode_u: address,
        address_mode_v: address,
        mag_filter: TextureMagFilter::Nearest,
        min_filter: TextureMinFilter::Nearest,
    }];
    (materials, textures, samplers)
}

fn compare_captures(
    reference: &[FrameCapture],
    actual: &[FrameCapture],
    backend: MeshletBackend,
    instance_count: u32,
) {
    assert_eq!(reference.len(), actual.len());
    for (frame, (reference, actual)) in reference.iter().zip(actual).enumerate() {
        assert_eq!(reference.color.len(), actual.color.len());
        assert_eq!(reference.depth.len(), actual.depth.len());
        compare_coverage_and_interior(reference, actual, backend, instance_count, frame);
    }
}

fn instance_translation(index: u32) -> glam::Vec3 {
    let column = index % 6;
    let row = index / 6;
    glam::Vec3::new(
        (column as f32 - 2.5) * 0.28,
        (row as f32 - 2.5) * 0.25,
        -2.0,
    )
}

fn coverage(color: &[u8]) -> Vec<bool> {
    let pixels = color.as_chunks::<4>().0;
    let clear = pixels[0];
    pixels.iter().map(|pixel| *pixel != clear).collect()
}

fn covered_within_one(mask: &[bool], x: usize, y: usize) -> bool {
    let width = EXTENT.width as usize;
    let height = EXTENT.height as usize;
    let x0 = x.saturating_sub(1);
    let y0 = y.saturating_sub(1);
    let x1 = (x + 1).min(width - 1);
    let y1 = (y + 1).min(height - 1);
    (y0..=y1).any(|near_y| (x0..=x1).any(|near_x| mask[near_y * width + near_x]))
}

fn compare_coverage_and_interior(
    reference: &FrameCapture,
    actual: &FrameCapture,
    backend: MeshletBackend,
    instance_count: u32,
    frame: usize,
) {
    let width = EXTENT.width as usize;
    let height = EXTENT.height as usize;
    let reference_coverage = coverage(&reference.color);
    let actual_coverage = coverage(&actual.color);
    let missing_from_actual = reference_coverage
        .iter()
        .enumerate()
        .filter(|(index, covered)| {
            **covered && !covered_within_one(&actual_coverage, index % width, index / width)
        })
        .count();
    let missing_from_reference = actual_coverage
        .iter()
        .enumerate()
        .filter(|(index, covered)| {
            **covered && !covered_within_one(&reference_coverage, index % width, index / width)
        })
        .count();
    assert_eq!(
        (missing_from_actual, missing_from_reference),
        (0, 0),
        "{backend} coverage differs beyond one pixel at capture {frame} with {instance_count} instances"
    );

    let mut compared_interior = 0usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let interior = (y - 1..=y + 1).all(|near_y| {
                (x - 1..=x + 1).all(|near_x| {
                    let index = near_y * width + near_x;
                    reference_coverage[index] && actual_coverage[index]
                })
            });
            if !interior {
                continue;
            }
            compared_interior += 1;
            let index = y * width + x;
            let color_index = index * 4;
            let color_delta = (0..4)
                .map(|channel| {
                    reference.color[color_index + channel]
                        .abs_diff(actual.color[color_index + channel])
                })
                .max()
                .unwrap();
            assert!(
                color_delta <= 1,
                "{backend} interior color differs by {color_delta} at capture {frame} with {instance_count} instances"
            );
            let depth_delta = (reference.depth[index] - actual.depth[index]).abs();
            assert!(
                depth_delta <= 1.0e-6,
                "{backend} interior depth differs by {depth_delta} at capture {frame} with {instance_count} instances"
            );
        }
    }
    assert!(
        compared_interior >= instance_count as usize,
        "{backend} had only {compared_interior} common interior pixels at capture {frame} with {instance_count} instances"
    );
}

fn assert_instance_center_probes(
    color: &[u8],
    depth: &[f32],
    camera: Camera,
    instance_count: u32,
    backend: MeshletBackend,
    expected_channel: usize,
) {
    let width = EXTENT.width as i32;
    let height = EXTENT.height as i32;
    let clear = &color[..4];
    for instance in 0..instance_count {
        let clip = camera.view_projection() * instance_translation(instance).extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let center_x = ((ndc.x * 0.5 + 0.5) * width as f32).floor() as i32;
        let center_y = ((0.5 - ndc.y * 0.5) * height as f32).floor() as i32;
        let found = (-1..=1).any(|dy| {
            (-1..=1).any(|dx| {
                let x = center_x + dx;
                let y = center_y + dy;
                if x < 0 || x >= width || y < 0 || y >= height {
                    return false;
                }
                let index = y as usize * width as usize + x as usize;
                let pixel = &color[index * 4..index * 4 + 4];
                let other_channel = usize::from(expected_channel == 0);
                pixel != clear
                    && depth[index] < 1.0
                    && pixel[expected_channel] > pixel[other_channel].saturating_add(32)
            })
        });
        assert!(
            found,
            "{backend} did not render instance {instance} near its projected center"
        );
    }
}

fn create_texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: EXTENT,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn import_texture<'frame>(
    frame: &mut zen_frame_graph::Frame<'frame>,
    native: &wgpu::Texture,
    initial_contents: InitialContents,
) -> zen_frame_graph::Texture<'frame> {
    let logical = frame
        .import_texture(
            TextureDesc {
                label: "meshlet.smoke.import".into(),
                size: native.size(),
                mip_level_count: native.mip_level_count(),
                sample_count: native.sample_count(),
                dimension: native.dimension(),
                format: native.format(),
                view_formats: Vec::new(),
                usage: UsagePolicy::Fixed(native.usage()),
            },
            ImportTextureOptions {
                initial_contents,
                exposed_usage: Some(native.usage()),
            },
        )
        .unwrap();
    frame.bind_imported_texture(logical, native).unwrap();
    logical
}

fn read_rgba8(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    read_texture_bytes(device, queue, texture, wgpu::TextureAspect::All)
}

fn read_depth32(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<f32> {
    let bytes = read_texture_bytes(device, queue, texture, wgpu::TextureAspect::DepthOnly);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn read_texture_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
) -> Vec<u8> {
    let bytes_per_row = EXTENT.width * 4;
    assert_eq!(bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("meshlet.smoke.readback"),
        size: u64::from(bytes_per_row * EXTENT.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("meshlet.smoke.readback-encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(EXTENT.height),
            },
        },
        EXTENT,
    );
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let pixels = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    pixels
}

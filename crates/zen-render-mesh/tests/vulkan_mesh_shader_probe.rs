//! Opt-in Vulkan hardware probes for the experimental mesh-shader backends.
//!
//! These tests deliberately bypass the renderer so a failure identifies the Vulkan mesh/task
//! execution contract rather than culling, FrameGraph, indirect preparation, or material code.

use std::sync::mpsc;

const GROUPS_X: u32 = 4;
const GROUPS_Y: u32 = 2;
const WORKGROUP_SIZE_X: u32 = 32;
const RECORD_WORDS: usize = 8;
const GROUP_COUNT: usize = (GROUPS_X * GROUPS_Y) as usize;
const WORD_COUNT: usize = 1 + GROUP_COUNT * RECORD_WORDS;
const MESH_DISPATCH_BUILTINS: [u32; 5] = [24, 26, 27, 28, 29];

const BUILTIN_SHADER: &str = r#"
enable wgpu_mesh_shader;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
};

@group(0) @binding(0) var<storage, read_write> probe: array<atomic<u32>, 65>;
var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output)
@workgroup_size(32, 1, 1)
fn ms_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    if (lane == 0u) {
        let record = atomicAdd(&probe[0], 1u);
        let base = 1u + record * 8u;
        atomicStore(&probe[base], workgroup_id.x);
        atomicStore(&probe[base + 1u], workgroup_id.y);
        atomicStore(&probe[base + 2u], global_invocation_id.x);
        atomicStore(&probe[base + 3u], global_invocation_id.y);
        atomicStore(&probe[base + 4u], local_invocation_id.x);
        atomicStore(&probe[base + 5u], num_workgroups.x);
        atomicStore(&probe[base + 6u], num_workgroups.y);
        atomicStore(&probe[base + 7u], lane);
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    }

    if (lane < 3u) {
        let positions = array<vec2<f32>, 3>(
            vec2<f32>(-0.5, -0.5),
            vec2<f32>(0.5, -0.5),
            vec2<f32>(0.0, 0.5),
        );
        mesh_output.vertices[lane].position = vec4<f32>(positions[lane], 0.0, 1.0);
    }
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
"#;

const EMPTY_OUTPUT_SHADER: &str = r#"
enable wgpu_mesh_shader;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) valid: u32,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
};

@group(0) @binding(0) var<storage, read_write> fragments: array<atomic<u32>, 2>;
var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output)
@workgroup_size(32)
fn ms_main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flattened = group_id.y * num_workgroups.x + group_id.x;
    let valid = flattened < 5u;
    if (lane == 0u) {
        mesh_output.vertex_count = select(0u, 3u, valid);
        mesh_output.primitive_count = select(0u, 1u, valid);
        mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    }
    if (lane < 3u) {
        let positions = array<vec2<f32>, 3>(
            vec2<f32>(-0.75, -0.75),
            vec2<f32>(0.75, -0.75),
            vec2<f32>(0.0, 0.75),
        );
        mesh_output.vertices[lane].position = vec4<f32>(positions[lane], 0.0, 1.0);
        mesh_output.vertices[lane].valid = select(0u, 1u, valid);
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    atomicAdd(&fragments[select(1u, 0u, input.valid != 0u)], 1u);
    return vec4<f32>(1.0);
}
"#;

const TASK_PAYLOAD_SHADER: &str = r#"
enable wgpu_mesh_shader;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
};

struct TaskPayload {
    values: array<vec4<u32>, 32>,
};

@group(0) @binding(0) var<storage, read_write> records: array<atomic<u32>, 145>;
var<task_payload> payload: TaskPayload;
var<workgroup> mesh_output: MeshOutput;

@task
@payload(payload)
@workgroup_size(32)
fn ts_main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) -> @builtin(mesh_task_size) vec3<u32> {
    let task_id = group_id.y * num_workgroups.x + group_id.x;
    payload.values[lane] = vec4<u32>(task_id, lane, 0x5a17u, task_id * 32u + lane);
    workgroupBarrier();
    let child_counts = array<u32, 4>(1u, 32u, 0u, 3u);
    return vec3<u32>(child_counts[task_id], 1u, 1u);
}

@mesh(mesh_output)
@payload(payload)
@workgroup_size(32)
fn ms_main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) child_id: vec3<u32>,
) {
    if (lane == 0u) {
        let record = atomicAdd(&records[0], 1u);
        let base = 1u + record * 4u;
        let value = payload.values[child_id.x];
        atomicStore(&records[base], value.x);
        atomicStore(&records[base + 1u], value.y);
        atomicStore(&records[base + 2u], value.z);
        atomicStore(&records[base + 3u], value.w);
        mesh_output.vertex_count = 0u;
        mesh_output.primitive_count = 0u;
    }
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
"#;

#[test]
fn naga_spirv_declares_distinct_mesh_builtin_inputs() {
    let module = naga::front::wgsl::parse_str(BUILTIN_SHADER).expect("probe WGSL must parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("probe WGSL must validate");
    let words = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options {
            lang_version: (1, 6),
            ..Default::default()
        },
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Mesh,
            entry_point: "ms_main".into(),
        }),
    )
    .expect("probe WGSL must lower to SPIR-V");
    assert_naga_mesh_builtin_shape(&words);
}

fn assert_naga_mesh_builtin_shape(words: &[u32]) {
    let mut builtin_variables = Vec::new();
    let mut entry_point_interfaces = Vec::new();
    let mut entry_point_function = None;
    let mut current_function = None;
    let mut function_calls = Vec::new();
    let mut function_loads = Vec::new();
    for instruction in spirv_instructions(words) {
        let opcode = instruction[0] & 0xffff;
        let operands = &instruction[1..];
        match opcode {
            15 => {
                entry_point_function = Some(operands[1]);
                let name_words = operands[2..]
                    .iter()
                    .position(|word| word.to_le_bytes().contains(&0))
                    .expect("OpEntryPoint name must be null terminated")
                    + 1;
                entry_point_interfaces.extend_from_slice(&operands[2 + name_words..]);
            }
            71 if operands.get(1) == Some(&11) => {
                builtin_variables.push((operands[0], operands[2]));
            }
            54 => current_function = Some(operands[1]),
            56 => current_function = None,
            57 => function_calls.push((
                current_function.expect("OpFunctionCall must be inside a function"),
                operands[2],
            )),
            61 => function_loads.push((
                current_function.expect("OpLoad must be inside a function"),
                operands[2],
            )),
            _ => {}
        }
    }
    builtin_variables.sort_unstable_by_key(|entry| entry.1);

    for builtin in MESH_DISPATCH_BUILTINS {
        let variables = builtin_variables
            .iter()
            .filter_map(|(variable, candidate)| (*candidate == builtin).then_some(*variable))
            .collect::<Vec<_>>();
        assert_eq!(
            variables.len(),
            1,
            "SPIR-V BuiltIn {builtin} variable count"
        );
        assert!(
            entry_point_interfaces.contains(&variables[0]),
            "SPIR-V BuiltIn {builtin} is missing from the MeshEXT entry-point interface"
        );
    }
    let mut input_ids = builtin_variables
        .iter()
        .filter_map(|(variable, builtin)| {
            MESH_DISPATCH_BUILTINS
                .contains(builtin)
                .then_some(*variable)
        })
        .collect::<Vec<_>>();
    input_ids.sort_unstable();
    input_ids.dedup();
    assert_eq!(
        input_ids.len(),
        5,
        "mesh builtins must use distinct variables"
    );

    let entry_point_function = entry_point_function.expect("missing MeshEXT entry point");
    let user_function = function_calls
        .iter()
        .find_map(|(caller, callee)| (*caller == entry_point_function).then_some(*callee))
        .expect("Naga mesh entry-point wrapper must call the WGSL user function");
    let builtins_loaded_by = |function| {
        let mut builtins = function_loads
            .iter()
            .filter_map(|(owner, pointer)| {
                (*owner == function).then(|| {
                    builtin_variables
                        .iter()
                        .find_map(|(variable, builtin)| (*variable == *pointer).then_some(*builtin))
                })?
            })
            .collect::<Vec<_>>();
        builtins.sort_unstable();
        builtins.dedup();
        builtins
    };
    assert_eq!(builtins_loaded_by(entry_point_function), vec![29]);
    assert_eq!(builtins_loaded_by(user_function), MESH_DISPATCH_BUILTINS);
}

fn spirv_instructions(words: &[u32]) -> impl Iterator<Item = &[u32]> {
    let mut offset = 5;
    std::iter::from_fn(move || {
        if offset >= words.len() {
            return None;
        }
        let word_count = (words[offset] >> 16) as usize;
        assert!(word_count > 0, "SPIR-V instruction has zero word count");
        let instruction = &words[offset..offset + word_count];
        offset += word_count;
        Some(instruction)
    })
}

#[test]
#[ignore = "requires a Vulkan adapter with EXT_mesh_shader"]
fn original_mesh_dispatch_builtins_are_correct() {
    pollster::block_on(run_builtin_probe());
}

#[test]
#[ignore = "requires a Vulkan adapter with EXT_mesh_shader"]
fn rectangular_padding_mesh_groups_emit_nothing() {
    pollster::block_on(run_empty_output_probe());
}

#[test]
#[ignore = "requires a Vulkan adapter with EXT_mesh_shader task support"]
fn dynamic_task_children_preserve_payload_and_child_mapping() {
    pollster::block_on(run_task_payload_probe());
}

async fn request_mesh_device(require_task: bool) -> (wgpu::AdapterInfo, wgpu::Device, wgpu::Queue) {
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
        .expect("no Vulkan adapter available for mesh shader probe");
    let info = adapter.get_info();
    assert!(
        adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER),
        "Vulkan adapter does not expose EXPERIMENTAL_MESH_SHADER: {info:?}"
    );
    let supported = adapter.limits();
    if require_task {
        assert!(supported.max_task_invocations_per_workgroup >= 32);
        assert!(supported.max_task_payload_size >= 512);
        assert!(supported.max_mesh_workgroup_total_count >= 32);
    }
    let required_limits = wgpu::Limits {
        max_mesh_workgroup_total_count: supported.max_mesh_workgroup_total_count,
        max_mesh_workgroups_per_dimension: supported.max_mesh_workgroups_per_dimension,
        max_mesh_invocations_per_workgroup: supported.max_mesh_invocations_per_workgroup,
        max_mesh_invocations_per_dimension: supported.max_mesh_invocations_per_dimension,
        max_mesh_output_vertices: supported.max_mesh_output_vertices,
        max_mesh_output_primitives: supported.max_mesh_output_primitives,
        max_task_workgroup_total_count: supported.max_task_workgroup_total_count,
        max_task_workgroups_per_dimension: supported.max_task_workgroups_per_dimension,
        max_task_invocations_per_workgroup: supported.max_task_invocations_per_workgroup,
        max_task_invocations_per_dimension: supported.max_task_invocations_per_dimension,
        max_task_payload_size: supported.max_task_payload_size,
        ..wgpu::Limits::default()
    };
    // SAFETY: Running an ignored hardware probe is the explicit opt-in required by wgpu's
    // experimental mesh-shader API.
    let experimental_features = unsafe { wgpu::ExperimentalFeatures::enabled() };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("zen.vulkan-mesh-shader-probe.device"),
            required_features: wgpu::Features::EXPERIMENTAL_MESH_SHADER,
            required_limits,
            experimental_features,
            ..Default::default()
        })
        .await
        .expect("failed to request Vulkan mesh shader probe device");
    (info, device, queue)
}

async fn run_builtin_probe() {
    let (info, device, queue) = request_mesh_device(false).await;
    let probe = create_storage_buffer(&device, "builtin", WORD_COUNT);
    let (readback, words) = run_mesh_probe(
        &device,
        &queue,
        BUILTIN_SHADER,
        "ms_main",
        None,
        &probe,
        WORD_COUNT,
        (GROUPS_X, GROUPS_Y, 1),
        wgpu::ShaderStages::MESH,
    );
    drop(readback);

    let mut records = words[1..].as_chunks::<RECORD_WORDS>().0.to_vec();
    records.sort_unstable();
    let mut expected = Vec::with_capacity(GROUP_COUNT);
    for y in 0..GROUPS_Y {
        for x in 0..GROUPS_X {
            expected.push([x, y, x * WORKGROUP_SIZE_X, y, 0, GROUPS_X, GROUPS_Y, 0]);
        }
    }
    expected.sort_unstable();
    eprintln!("mesh shader probe adapter: {info:?}");
    assert_eq!(words[0], GROUP_COUNT as u32, "mesh workgroup count");
    assert_eq!(records, expected, "lane-0 mesh builtin records");
}

async fn run_empty_output_probe() {
    let (info, device, queue) = request_mesh_device(false).await;
    let probe = create_storage_buffer(&device, "empty-output", 2);
    let (readback, words) = run_mesh_probe(
        &device,
        &queue,
        EMPTY_OUTPUT_SHADER,
        "ms_main",
        None,
        &probe,
        2,
        (GROUPS_X, GROUPS_Y, 1),
        wgpu::ShaderStages::MESH | wgpu::ShaderStages::FRAGMENT,
    );
    drop(readback);
    eprintln!("empty output probe adapter: {info:?}; fragment counts={words:?}");
    assert!(words[0] > 0, "valid mesh groups must rasterize fragments");
    assert_eq!(
        words[1], 0,
        "rectangular padding groups must emit no fragments"
    );
}

async fn run_task_payload_probe() {
    const CHILDREN: usize = 36;
    const TASK_WORDS: usize = 1 + CHILDREN * 4;
    let (info, device, queue) = request_mesh_device(true).await;
    let probe = create_storage_buffer(&device, "task-payload", TASK_WORDS);
    let (readback, words) = run_mesh_probe(
        &device,
        &queue,
        TASK_PAYLOAD_SHADER,
        "ms_main",
        Some("ts_main"),
        &probe,
        TASK_WORDS,
        (2, 2, 1),
        wgpu::ShaderStages::MESH,
    );
    drop(readback);

    assert_eq!(words[0], CHILDREN as u32, "dynamic task child count");
    let mut actual = words[1..]
        .as_chunks::<4>()
        .0
        .iter()
        .take(CHILDREN)
        .copied()
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = Vec::with_capacity(CHILDREN);
    for (task_id, child_count) in [(0u32, 1u32), (1, 32), (2, 0), (3, 3)] {
        for child in 0..child_count {
            expected.push([task_id, child, 0x5a17, task_id * 32 + child]);
        }
    }
    expected.sort_unstable();
    eprintln!("task payload probe adapter: {info:?}");
    assert_eq!(
        actual, expected,
        "task payload isolation and child ID mapping"
    );
}

fn create_storage_buffer(device: &wgpu::Device, suffix: &str, words: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("zen.mesh-shader-probe.{suffix}.storage")),
        size: (words * size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[expect(clippy::too_many_arguments)]
fn run_mesh_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &str,
    mesh_entry: &str,
    task_entry: Option<&str>,
    probe: &wgpu::Buffer,
    word_count: usize,
    groups: (u32, u32, u32),
    visibility: wgpu::ShaderStages,
) -> (wgpu::Buffer, Vec<u32>) {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zen.mesh-shader-probe.readback"),
        size: (word_count * size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("zen.mesh-shader-probe.bind-group-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zen.mesh-shader-probe.bind-group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: probe.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("zen.mesh-shader-probe.pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("zen.mesh-shader-probe.shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let color_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
        label: Some("zen.mesh-shader-probe.pipeline"),
        layout: Some(&pipeline_layout),
        task: task_entry.map(|entry_point| wgpu::TaskState {
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        mesh: wgpu::MeshState {
            module: &module,
            entry_point: Some(mesh_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("zen.mesh-shader-probe.color"),
        size: wgpu::Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("zen.mesh-shader-probe.encoder"),
    });
    encoder.clear_buffer(probe, 0, None);
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zen.mesh-shader-probe.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw_mesh_tasks(groups.0, groups.1, groups.2);
    }
    encoder.copy_buffer_to_buffer(
        probe,
        0,
        &readback,
        0,
        (word_count * size_of::<u32>()) as u64,
    );
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    let words = mapped
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect();
    drop(mapped);
    readback.unmap();
    (readback, words)
}

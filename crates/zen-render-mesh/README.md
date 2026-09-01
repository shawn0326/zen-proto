# zen-render-mesh

`zen-render-mesh` is the GPU-driven Mesh domain renderer for `zen-proto`. It owns Mesh scene GPU
data, visibility state, native pass pipelines, and stats readback. It contributes work to a
caller-owned `zen-frame-graph` frame; it does not acquire a surface, allocate shared frame targets,
compile or execute the graph, or present.

## Concepts

```text
MeshRenderer
  MeshGpuScene             persistent geometry, material, instance, and texture data
  MeshVisibilityState      persistent visibility lists and history
  HiZStage                 multi-node depth-pyramid construction
  MeshPassSet              cull, indirect preparation, occlusion, and draw passes
  MeshGraphRecorder        Mesh node ordering, optional branches, and roots
  MeshGraphResources       per-frame logical handles for Mesh-owned GPU resources
```

Each ordinary `*Pass::record` method contributes exactly one FrameGraph node and declares its
complete resource access contract. `HiZStage` is intentionally a stage rather than a pass: it owns
the full depth-pyramid operation and contributes one node per mip while preserving the existing
`Initial Hi-Z Pyramid` and `Final Hi-Z Pyramid` debug groups.

`MeshRenderTargets` contains only caller-registered FrameGraph handles for color and depth. This
keeps target policy in the renderer-composition layer and makes it possible for Mesh, Line,
Particle, post-processing, and presentation modules to share graph resources explicitly.

## Vulkan meshlet pipeline

The original `MeshRenderer` remains the indexed-indirect baseline and correctness oracle. The
independent `MeshletRenderer` owns the experimental Vulkan-only geometry pipeline; it does not
branch or replace `MeshRenderer` internally.

```text
MeshletRenderer
  .zenmesh v1 asset       checksummed LODs, 64v/64t meshlets, bounds/cones, fallback indices
  MeshletGpuScene         buffer + u32-offset arenas (no BDA or buffer binding arrays)
  BindlessTextureArena    fixed texture/sampler table with generation-checked handles
  shared GPU front-end    classify/LOD -> scan -> candidate scatter -> cull -> indirect prepare
  IndexedIndirect         multi_draw_indexed_indirect_count per PSO bin
  MeshOnly                one built-in-addressed mesh workgroup per visible meshlet
  TaskMesh                up to 32 payload-indexed mesh children per task workgroup
```

All three backends consume the same compute-compacted visible list. `BackendWorkCounts` is a
16-byte handoff written by indirect preparation after capacity and device-limit clipping. MeshOnly
linearizes `workgroup_id` with `num_workgroups` and reads that visible-list entry directly;
rectangular padding workgroups emit zero vertices and primitives. TaskMesh linearizes each task
workgroup, copies at most 32 consecutive visible entries into its private payload, returns the real
remaining child count, and lets each mesh child address the payload with `workgroup_id.x`. Empty
bins dispatch zero work and padded task groups return zero children. There is no driver-specific
shader path or runtime driver-version parsing.

The scan stage is a three-dispatch hierarchical exclusive prefix scan: 256-instance local scans,
a 256-lane scan of evenly partitioned block sums, then parallel block-offset addition. The default
262,144-instance capacity therefore uses 1,024 block sums (4 KiB of scratch) and never requires a
CPU readback to determine downstream work.

`MeshletBackend::Auto` is conservative: an unknown adapter/driver profile resolves to
`IndexedIndirect`; `TaskMesh` is promoted only by a matching local benchmark profile whose
geometry-bound GPU p95 is at least 10% faster. Backend selection happens before device creation
and a renderer instance owns one concrete path for its lifetime.

### Capability baseline

Hardware support is defined by Vulkan features and numeric limits, not a GPU product generation.
The application must satisfy the following adapter contract before requesting the device:

| Requirement | IndexedIndirect | MeshOnly | TaskMesh |
| --- | --- | --- | --- |
| wgpu backend | Vulkan | Vulkan | Vulkan |
| downlevel flag | `INDIRECT_EXECUTION` | `INDIRECT_EXECUTION` | `INDIRECT_EXECUTION` |
| bindless/indirect features | `TEXTURE_BINDING_ARRAY`, `PARTIALLY_BOUND_BINDING_ARRAY`, non-uniform sampled-texture/storage-buffer array indexing, `MULTI_DRAW_INDIRECT_COUNT`, `INDIRECT_FIRST_INSTANCE` | same | same |
| experimental opt-in | none | `EXPERIMENTAL_MESH_SHADER` plus the unsafe experimental token | same |
| mesh stage | n/a | at least 64 invocations, X dimension 64, 64 output vertices, 64 output primitives | same |
| task stage | n/a | n/a | at least 32 invocations, X dimension 32, 512-byte payload |
| child/dispatch capacity | n/a | non-zero mesh total and per-dimension limits | mesh total and per-dimension limits at least 32; non-zero task total and per-dimension limits |

Bindless table sizes are clamped to the adapter's binding-array and sampler limits. All shaders use
checked WGSL translated by Naga; there is no renderer HAL, passthrough SPIR-V, HLSL, MSL, BDA, or
driver-version compatibility path. Applications may still supply an exact, application-owned
`MeshletDriverBlacklist`; the built-in blacklist remains empty.

### Hardware qualification and known issues

`tests/vulkan_mesh_shader_probe.rs` is the release/merge gate for experimental backends. MeshOnly
requires the static SPIR-V interface check, original dispatch-builtin probe, and rectangular empty
output probe to pass. TaskMesh additionally requires the dynamic child-count/payload-isolation
probe. A failing backend must be disabled (and excluded from `Auto`) through an exact application
blacklist or by withholding that build; compatibility shaders are not accepted.

| Environment | Builtins | Empty mesh output | Dynamic task children/payload | Status |
| --- | --- | --- | --- | --- |
| Windows, RTX 2080 Ti, NVIDIA 616.56, wgpu/Naga 30 | pass | pass | pass | minimum NVIDIA baseline verified by this project |
| Windows, RTX 2080 Ti, NVIDIA 565.90, wgpu/Naga 30 | fail | not qualifying | not qualifying | known bad |

On 565.90, NVIDIA compiled dispatch builtins loaded in Naga's MeshEXT wrapper call tree to incorrect
values even though the emitted SPIR-V declared distinct builtin variables and entry-point interface
IDs. The same original Naga SPIR-V path passes on 616.56. The diagnostic SPIR-V binary rewriter used
to isolate that driver failure is intentionally not part of the maintained probe or renderer.

Run the static and opt-in hardware gates with:

```text
cargo test -p zen-render-mesh --test vulkan_mesh_shader_probe
cargo test -p zen-render-mesh --test vulkan_mesh_shader_probe -- --ignored --nocapture
cargo test -p zen-render-mesh --test vulkan_meshlet_smoke -- --ignored --nocapture
```

The workspace carries a source patch for `wgpu-hal` 30.0.1 because that release's Vulkan resource
barriers omit the task/mesh pipeline stages. This is a dependency-level synchronization correction;
renderer code still uses only wgpu's public API. The patch is isolated under
`third_party/wgpu-hal-30.0.1` so it can be removed when the correction ships upstream.

The initial production boundary is static opaque rigid instances with standard-Z forward shading.
Alpha mask/blend, skinning, morph targets, shadows, virtual geometry, and streaming are deliberately
outside the v1 contract.

## Frame lifecycle

```text
prepare_frame(queue, MeshRenderInput, extent) -> PreparedMeshFrame
record_frame_graph(frame, MeshRenderTargets, &prepared)
compile and submit in the caller
after_submit(device, prepared) | after_discard(prepared)
```

`PreparedMeshFrame` is a transaction ticket: recording borrows it, then exactly one terminal hook
consumes it. Stats reservations are committed only by `after_submit`; discarding a frame leaves the
request available for a later successful submission.

`PreparedMeshletFrame` follows the same transaction contract. It also pins the bindless bind-group
epoch selected at the frame boundary; retired epochs stay alive until their submitted work has
completed.

Device creation remains outside this crate. Applications aggregate `MeshRenderer::required_features()`
and `MeshRenderer::required_limits(adapter_limits)` with the requirements of their other render
modules. Mesh does not require `TIMESTAMP_QUERY`; timing is a concern of the renderer-composition
layer.

Meshlet counters and FrameGraph timestamps complete independently. Call
`request_stats_with_gpu_timing`, pass the enclosing frame index through `MeshletRenderInput`, and
feed the completed report to `associate_gpu_timing`; the renderer joins them strictly by frame
identity before `take_stats` returns the three-frame-delayed snapshot. Timestamp features remain
optional renderer-host capabilities and are not part of `MeshletDeviceRequirements`.

The crate is private to the workspace (`publish = false`).

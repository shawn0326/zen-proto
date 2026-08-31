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
  shared GPU front-end    classify/LOD -> scan -> scatter -> cull -> indirect prepare
  IndexedIndirect         multi_draw_indexed_indirect_count per PSO bin
  MeshOnly                one claimed mesh workgroup per visible meshlet, plus bounded tail padding
  TaskMesh                fixed 32-child task fanout; child mesh groups claim visible work
```

All three backends consume the same compute-compacted visible list. On the validated wgpu
30/Naga 30 NVIDIA Vulkan path, mesh-stage workgroup/global IDs, dynamically computed task child
counts, and dynamically empty mesh output are not reliable. MeshOnly and TaskMesh therefore use a
per-PSO-bin GPU atomic to assign each mesh workgroup a unique visible-list slot. Empty bins still
dispatch zero work; rectangular MeshOnly padding and TaskMesh's fixed 32-child tail repeat the last
legal entry. This preserves the logical visible set without a CPU readback. Atomic assignment and
duplicate-tail work are included in benchmark timing, while `output_vertices` and
`output_primitives` report logical visible geometry rather than compatibility duplicates.
`TaskMesh` remains an explicit experimental backend, and `Auto` cannot select it without a matching
local profile. Task-stage culling/compaction can replace this compatibility path only after the
relevant wgpu/Naga and driver matrix is proven stable.

The scan stage is a three-dispatch hierarchical exclusive prefix scan: 256-instance local scans,
a 256-lane scan of evenly partitioned block sums, then parallel block-offset addition. The default
262,144-instance capacity therefore uses 1,024 block sums (4 KiB of scratch) and never requires a
CPU readback to determine downstream work.

`MeshletBackend::Auto` is conservative: an unknown adapter/driver profile resolves to
`IndexedIndirect`; `TaskMesh` is promoted only by a matching local benchmark profile whose
geometry-bound GPU p95 is at least 10% faster. Backend selection happens before device creation
and a renderer instance owns one concrete path for its lifetime.

The meshlet renderer accepts Vulkan capabilities only. `IndexedIndirect` uses the ordinary checked
WGSL runtime path. `MeshOnly` and `TaskMesh` additionally require wgpu's experimental mesh-shader
feature and a device created with its experimental token. All shaders remain WGSL and are validated
and translated by Naga; there is no HAL, passthrough SPIR-V, HLSL, MSL, or BDA path.

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

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

Device creation remains outside this crate. Applications aggregate `MeshRenderer::required_features()`
and `MeshRenderer::required_limits(adapter_limits)` with the requirements of their other render
modules. Mesh does not require `TIMESTAMP_QUERY`; timing is a concern of the renderer-composition
layer.

The crate is private to the workspace (`publish = false`).

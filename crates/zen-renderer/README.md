# zen-renderer

`zen-renderer` is the GPU-driven rendering layer used by `zen-proto`. Its public architecture has
two explicit levels:

- `Renderer` composes a complete frame, owns the FrameGraph, registers the acquired surface color
  and transient depth target, compiles the graph, and executes it.
- `MeshRenderer` owns Mesh-domain scene resources and its visibility, Hi-Z, draw, and stats
  implementation.

The crate deliberately uses concrete composition. It does not define a universal render-module,
feature, or plugin trait. Future Line and Particle domain renderers can be added to `Renderer` and
connected with explicit typed targets or outputs. The term `Feature` is reserved for an optional
extension that injects passes into an existing renderer; `Plugin` is reserved for a registration
lifecycle.

## Frame lifecycle

```text
Renderer
  validate surface target
  MeshRenderer::prepare_frame       queue uploads and stats reservation
  begin FrameGraph frame
  register surface color + transient depth
  MeshFrameRecorder                 stage order and optional branches
    VisibilityStage                 cull, dispatch preparation, Hi-Z, occlusion
    MeshDrawStage                   indirect preparation and mesh draws
  compile + execute
  MeshRenderer::after_submit        stats commit and mapping
```

Each internal `Pass::record` method creates one FrameGraph node, declares its complete access
contract, and installs its executor. `MeshFrameRecorder` only selects ordering and branches. A
`Pipeline` name is used only for native wgpu compute/render pipelines or their creation/cache.

Recording keeps the two-pass Mesh flow at the root so cull, prepare, and draw nodes remain
continuous. The repeated mip chains are collapsed into optional `Initial Hi-Z Pyramid` and
`Final Hi-Z Pyramid` groups; `Frame Targets`, optional `Debug View`, and optional
`Stats Readback` remain independent root groups. These groups appear in full CPU reports. Applications may call
`Renderer::set_gpu_debug_groups_enabled(true)` to mirror retained group paths into GPU debug
markers; marker emission is disabled by default and does not alter graph topology.

## Public construction

```rust,no_run
use zen_renderer::{
    FrameInput, Renderer,
    mesh::{MeshFrameInput, MeshRenderer},
};

# fn render(
#     device: &wgpu::Device,
#     queue: &wgpu::Queue,
#     surface_texture: &wgpu::Texture,
#     mesh: MeshRenderer,
#     mesh_input: MeshFrameInput,
# ) -> Result<(), zen_frame_graph::FrameGraphError> {
let mut renderer = Renderer::new(device, mesh);
renderer.render(
    device,
    queue,
    FrameInput {
        frame_index: 42,
        surface_texture,
        mesh: mesh_input,
    },
)?;
# Ok(())
# }
```

Surface acquisition and presentation remain in the platform/demo layer. Mesh scene buffers,
visibility lists and history, uniforms, and stats staging buffers remain owned by `MeshRenderer`
and are imported into each frame. Per-frame depth and Hi-Z textures are transient FrameGraph
resources backed by its cross-frame resource pool.

Imported Mesh resources may continue to participate in long-lived renderer-owned bind groups,
provided each pass declares the matching access. Transient depth and Hi-Z views are resolved from
pass-scoped tokens and are never cached outside their callback. A future Line or Particle renderer
sharing a native resource with Mesh must receive the same logical handle imported once by the
top-level `Renderer`.

GPU timing is sampled explicitly. `Renderer::request_gpu_timing()` coalesces
repeated requests and times the next eligible successful frame;
`Renderer::take_gpu_timing()` non-blockingly returns its `GpuTimingReport`.
Requests made while a previous readback is pending remain queued until that
result is taken. Timing support is enabled when the selected adapter exposes
`TIMESTAMP_QUERY`; unsupported adapters still render normally and return an
`Unavailable` timing report.

With the `snapshot` feature, `Renderer::request_frame_graph_snapshot()`
coalesces repeated requests and captures the next eligible successful frame.
`Renderer::take_frame_graph_snapshot()` non-blockingly returns the Snapshot 1.0
object or a producer error after GPU timing and post-execution pool statistics
are ready. Snapshot takes priority over a new ordinary timing request; an
already-pending ordinary readback delays capture and its queued request is not
lost. Unsupported timestamp queries still produce a Snapshot with an explicit
unavailable timing result. Normal frames do not request a Full report.

The crate is currently private to the workspace (`publish = false`).

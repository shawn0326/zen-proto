# zen-proto

Prototype GPU-driven renderer in Rust using wgpu.

## Workspace

- `crates/zen-frame-graph`: renderer-agnostic, wgpu-specific compiler and GPU runtime with transient resource pooling.
- `crates/zen-render`: domain-independent `RenderHost` execution infrastructure and typed
  `FrameComposer` contract.
- `crates/zen-render-mesh`: GPU-driven Mesh domain renderer, persistent GPU scene, and FrameGraph
  passes.
- `apps/zen-demo`: concrete forward frame composition, window-surface integration, and assets.

The authoritative Meshlet hardware requirements, validation matrix, and known driver issues are in
[`crates/zen-render-mesh/README.md`](crates/zen-render-mesh/README.md). Meshlet demo, probe, and
benchmark commands are in [`apps/zen-demo/MESHLET.md`](apps/zen-demo/MESHLET.md).

Run the demos from the workspace root:

```shell
cargo run -p zen-demo --bin basic
cargo run -p zen-demo --bin load-gltf
```

An optional model path can be passed to the glTF demo:

```shell
cargo run -p zen-demo --bin load-gltf -- path/to/model.gltf
```

Relative model paths are resolved from the workspace root.

## todo

- [ ] Optimize HiZ generation process
- [ ] Optimize rendering pipeline
- [ ] Improve occlusion culling algorithm
- [ ] Automatic LOD

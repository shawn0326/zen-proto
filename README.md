# zen-proto

Prototype GPU-driven renderer in Rust using wgpu.

## Workspace

- `crates/zen-frame-graph`: renderer-agnostic, wgpu-specific compiler and GPU runtime with transient resource pooling.
- `crates/zen-renderer`: GPU-driven renderer with a frame-composition `Renderer` and an explicit
  Mesh-domain `MeshRenderer` made of internal visibility/draw stages and FrameGraph passes.
- `apps/zen-demo`: interactive demo applications, window-surface integration, and assets.

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

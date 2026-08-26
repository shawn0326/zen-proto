# zen-proto

Prototype GPU-driven renderer in Rust using wgpu.

## Workspace

- `crates/zen-renderer`: GPU-driven renderer library.
- `apps/zen-demo`: interactive demo applications and their assets.

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

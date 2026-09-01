# Vulkan meshlet demo and benchmark

The advanced demo deliberately creates a Vulkan-only wgpu instance. The five startup paths are:

```text
cargo run -p zen-demo --bin meshlet-gltf -- --renderer legacy
cargo run -p zen-demo --bin meshlet-gltf -- --renderer indexed
cargo run -p zen-demo --bin meshlet-gltf -- --renderer mesh
cargo run -p zen-demo --bin meshlet-gltf -- --renderer task-mesh
cargo run -p zen-demo --bin meshlet-gltf -- --renderer auto [--auto-profile auto.json]
```

Pass a glTF/GLB path as the first argument to override the bundled Damaged Helmet model. Meshlet
loading rejects alpha-mask and alpha-blend primitives; opaque glTF `doubleSided` primitives are
placed in the two-sided PSO bin.

The `indexed`, `mesh`, `task-mesh`, and `auto` paths support a stable per-meshlet ID color view.
Press `M` to toggle it while the demo is running, or pass `--meshlet-debug` to start in that mode:

```text
cargo run -p zen-demo --bin meshlet-gltf -- --renderer auto --meshlet-debug
```

The debug colors are unlit and bypass material texture sampling so meshlet boundaries remain
clear. `--meshlet-debug` is rejected with the legacy renderer and with `--benchmark-out`; runtime
mode switching is also disabled during a benchmark so reports always measure shaded rendering.

Build a deterministic cache explicitly with:

```text
cargo run -p zen-demo --bin build-zenmesh -- scene.gltf -o scene.zenmesh
```

The development loader verifies the v1 header, every section checksum, source/build hashes and
topology ranges. Missing, stale or corrupt entries are rebuilt and atomically replaced.

The authoritative feature, limit, driver-baseline, and known-issue documentation lives in
[`crates/zen-render-mesh/README.md`](../../crates/zen-render-mesh/README.md). Before using an
experimental backend on a new adapter/driver, run:

```text
cargo test -p zen-render-mesh --test vulkan_mesh_shader_probe
cargo test -p zen-render-mesh --test vulkan_mesh_shader_probe -- --ignored --nocapture
cargo test -p zen-render-mesh --test vulkan_meshlet_smoke -- --ignored --nocapture
```

If IndexedIndirect works but either experimental backend fails, keep using `--renderer indexed`,
capture the adapter name/driver info printed by the probe, and do not generate an Auto profile for
that driver. Validation-layer complaints involving task/mesh resource visibility should also be
checked against the local `wgpu-hal 30.0.1` barrier patch described by the authoritative README.

## Fixed benchmark

Each run renders to an offscreen 1920x1080 target, uses the versioned deterministic camera track,
warms up for 120 frames, collects 600 GPU timestamp samples, writes JSON, and exits. Debug groups
are enabled for RenderDoc/Nsight/RGP captures. `--geometry-bound` is an explicit assertion used by
the TaskMesh promotion gate. Benchmark startup fails clearly unless Vulkan timestamp queries are
available. Schema v6 pairs every timestamp sample with the same frame's delayed GPU counters and
rejects the run if any sticky capacity-overflow flag is observed. Reports retain the whole-frame
GPU median/p95 and a per-pass median/p95 block for clear, classify, scan, candidate scatter,
culling, occluder, Hi-Z, indirect preparation, backend raster, and optional stats-copy work. A pass
which is absent from the selected path is serialized as `null`. Schema v5 reports and Auto profiles
are rejected by the existing strict version check and must be regenerated.

```text
cargo run --release -p zen-demo --bin meshlet-gltf -- scene.gltf --renderer legacy --benchmark-out legacy.json
cargo run --release -p zen-demo --bin meshlet-gltf -- scene.gltf --renderer indexed --geometry-bound --benchmark-out indexed.json
cargo run --release -p zen-demo --bin meshlet-gltf -- scene.gltf --renderer task-mesh --geometry-bound --benchmark-out task-mesh.json
```

Generate the adapter/driver/build/scene-bound Auto profile and consume it:

```text
cargo run --release -p zen-demo --bin meshlet-profile -- indexed.json task-mesh.json --legacy legacy.json -o auto.json
cargo run --release -p zen-demo --bin meshlet-gltf -- scene.gltf --renderer auto --auto-profile auto.json
```

Profile generation fails unless the reports have matching identities, IndexedIndirect is within
10% of the optional legacy baseline, and TaskMesh GPU p95 is at least 10% faster. A stale or
unsupported profile never promotes Auto and it falls back to the indexed-indirect path.

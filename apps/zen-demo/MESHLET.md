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

Build a deterministic cache explicitly with:

```text
cargo run -p zen-demo --bin build-zenmesh -- scene.gltf -o scene.zenmesh
```

The development loader verifies the v1 header, every section checksum, source/build hashes and
topology ranges. Missing, stale or corrupt entries are rebuilt and atomically replaced.

`indexed`, `mesh`, and `task-mesh` share the same compute-produced visibility list. The current
Vulkan mesh paths use a per-bin GPU atomic to assign unique visible work because mesh-stage
workgroup/global IDs are not reliable on the validated wgpu 30/Naga 30 NVIDIA path. `task-mesh`
also uses a fixed 32-child fanout because dynamic task child counts are unreliable. Empty bins
dispatch zero work; rectangular and fixed-fanout tails repeat the last visible meshlet. The
benchmark measures this compatibility overhead, and `auto` still requires a matching local
promotion profile. Reported output vertex/primitive counts describe logical visible geometry and
exclude repeated compatibility-tail work.

## Fixed benchmark

Each run renders to an offscreen 1920x1080 target, uses the versioned deterministic camera track,
warms up for 120 frames, collects 600 GPU timestamp samples, writes JSON, and exits. Debug groups
are enabled for RenderDoc/Nsight/RGP captures. `--geometry-bound` is an explicit assertion used by
the TaskMesh promotion gate. Benchmark startup fails clearly unless Vulkan timestamp queries are
available. Schema v5 pairs every timestamp sample with the same frame's delayed GPU counters and
rejects the run if any sticky capacity-overflow flag is observed. Reports retain the whole-frame GPU median/p95 and a per-pass
median/p95 block for clear, classify, scan, scatter, culling, occluder, Hi-Z, indirect preparation,
backend raster, and optional stats-copy work. A pass which is absent from the selected path is
serialized as `null`.

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

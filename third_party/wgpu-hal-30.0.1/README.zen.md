# Local Vulkan mesh/task barrier fix

This directory is the crates.io `wgpu-hal` 30.0.1 source under its original MIT/Apache-2.0
licenses. The workspace patches that exact dependency instead of calling `wgpu-hal` from renderer
code.

The local delta fixes Vulkan resource synchronization for `EXPERIMENTAL_MESH_SHADER` devices:

- buffer uniform/storage barriers include `TASK_SHADER_EXT` and `MESH_SHADER_EXT`;
- sampled/storage texture barriers include those stages as well;
- the extension stage bits are added only when wgpu enabled the mesh-shader device feature.

Without this, compute-written visible work, backend work counts, counters, and Hi-Z data can remain
invisible to a following task/mesh shader even though wgpu emitted a resource barrier. Remove this
patch once the same feature-aware stage mapping ships upstream.

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
};

struct BackendWorkCounts {
    mesh: vec2<u32>,
    task: vec2<u32>,
};

struct FrameUniform {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
    parameters: vec4<f32>,
    counts: vec4<u32>,
    limits: vec4<u32>,
};

struct Counters {
    candidate_count: atomic<u32>,
    packet_count_backface: atomic<u32>,
    packet_count_two_sided: atomic<u32>,
    visible_count_backface: atomic<u32>,
    visible_count_two_sided: atomic<u32>,
    instances_visible: atomic<u32>,
    culled_frustum: atomic<u32>,
    culled_cone: atomic<u32>,
    culled_hiz: atomic<u32>,
    output_vertices: atomic<u32>,
    output_primitives: atomic<u32>,
    overflow: atomic<u32>,
    lod_histogram: array<atomic<u32>, 8>,
    lod_overflow_instances: atomic<u32>,
    conservatively_visible_meshlets: atomic<u32>,
    raster_claim_backface: atomic<u32>,
    raster_claim_two_sided: atomic<u32>,
    _pad: array<u32, 8>,
};

@group(0) @binding(0) var<storage, read_write> counters: Counters;
@group(0) @binding(1) var<storage, read_write> mesh_dispatch: array<DispatchIndirectArgs>;
@group(0) @binding(2) var<storage, read_write> task_dispatch: array<DispatchIndirectArgs>;
@group(0) @binding(3) var<uniform> frame: FrameUniform;
@group(0) @binding(4) var<storage, read_write> backend_work_counts: BackendWorkCounts;

// Specialized from the selected Vulkan device. Mesh/task draw limits constrain the product of the
// three indirect dimensions in addition to each individual dimension.
override MAX_MESH_WORKGROUP_TOTAL_COUNT: u32 = 0u;
override MAX_TASK_WORKGROUP_TOTAL_COUNT: u32 = 0u;
override MESH_DISPATCH_WIDTH: u32 = 1u;
override TASK_DISPATCH_WIDTH: u32 = 1u;

const TASK_MESHLETS_PER_WORKGROUP: u32 = 32u;

fn flattened_dispatch(count: u32, total_limit: u32, dispatch_width: u32) -> DispatchIndirectArgs {
    // A zero specialization limit means that this shader stage is not enabled for the selected
    // backend (both stages for IndexedIndirect, task only for MeshOnly). Its unused arguments stay
    // empty without turning the shared sticky overflow flag into a false positive.
    if (count == 0u || total_limit == 0u) {
        return DispatchIndirectArgs(0u, 1u, 1u);
    }
    let safe_count = min(count, total_limit);
    let width = max(dispatch_width, 1u);
    if (safe_count <= width) {
        if (safe_count < count) {
            atomicOr(&counters.overflow, 16u);
        }
        return DispatchIndirectArgs(safe_count, 1u, 1u);
    }

    // The specialized limit is an exact legal rectangle computed from the selected Vulkan
    // stage limits. Consequently the padded final row remains within the total-workgroup limit.
    let rows = safe_count / width + select(0u, 1u, (safe_count % width) != 0u);
    if (safe_count < count) {
        atomicOr(&counters.overflow, 16u);
    }
    return DispatchIndirectArgs(width, rows, 1u);
}

fn clamped_task_meshlet_count(count: u32) -> u32 {
    // The shared shader specializes this stage to zero for IndexedIndirect and MeshOnly. Ignore
    // their visible counters without turning the sticky overflow bit into a false positive.
    if (MAX_TASK_WORKGROUP_TOTAL_COUNT == 0u) {
        return 0u;
    }
    // Multiplication by 32 must remain representable even when a Vulkan implementation reports a
    // very large task-workgroup limit. The rounded task count below can then add 31 without wrap.
    let safe_group_capacity = min(MAX_TASK_WORKGROUP_TOTAL_COUNT, 0xffffffffu / TASK_MESHLETS_PER_WORKGROUP);
    let meshlet_capacity = safe_group_capacity * TASK_MESHLETS_PER_WORKGROUP;
    let safe_count = min(count, meshlet_capacity);
    if (safe_count < count) {
        atomicOr(&counters.overflow, 16u);
    }
    return safe_count;
}

fn task_workgroup_count(meshlet_count: u32) -> u32 {
    return meshlet_count / TASK_MESHLETS_PER_WORKGROUP
        + select(0u, 1u, (meshlet_count % TASK_MESHLETS_PER_WORKGROUP) != 0u);
}

@compute @workgroup_size(1)
fn main() {
    let mesh_backface_count = min(atomicLoad(&counters.visible_count_backface), frame.counts.w);
    let mesh_two_sided_count = min(atomicLoad(&counters.visible_count_two_sided), frame.counts.w);
    let task_backface_count = clamped_task_meshlet_count(mesh_backface_count);
    let task_two_sided_count = clamped_task_meshlet_count(mesh_two_sided_count);
    let task_backface_groups = task_workgroup_count(task_backface_count);
    let task_two_sided_groups = task_workgroup_count(task_two_sided_count);
    backend_work_counts.mesh = vec2<u32>(
        min(mesh_backface_count, MAX_MESH_WORKGROUP_TOTAL_COUNT),
        min(mesh_two_sided_count, MAX_MESH_WORKGROUP_TOTAL_COUNT),
    );
    backend_work_counts.task = vec2<u32>(
        task_backface_count,
        task_two_sided_count,
    );
    mesh_dispatch[0] = flattened_dispatch(
        mesh_backface_count,
        MAX_MESH_WORKGROUP_TOTAL_COUNT,
        MESH_DISPATCH_WIDTH,
    );
    mesh_dispatch[1] = flattened_dispatch(
        mesh_two_sided_count,
        MAX_MESH_WORKGROUP_TOTAL_COUNT,
        MESH_DISPATCH_WIDTH,
    );
    task_dispatch[0] = flattened_dispatch(
        task_backface_groups,
        MAX_TASK_WORKGROUP_TOTAL_COUNT,
        TASK_DISPATCH_WIDTH,
    );
    task_dispatch[1] = flattened_dispatch(
        task_two_sided_groups,
        MAX_TASK_WORKGROUP_TOTAL_COUNT,
        TASK_DISPATCH_WIDTH,
    );
}

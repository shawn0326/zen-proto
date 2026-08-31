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

struct InstanceClassification {
    meshlet_count: u32,
    meshlet_offset: u32,
    selected_lod: u32,
    _pad: u32,
};

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
};

@group(0) @binding(0) var<storage, read_write> classifications: array<InstanceClassification>;
@group(0) @binding(1) var<storage, read_write> counters: Counters;
@group(0) @binding(2) var<uniform> frame: FrameUniform;
@group(0) @binding(3) var<storage, read_write> candidate_dispatch: DispatchIndirectArgs;
@group(0) @binding(4) var<storage, read_write> scan_blocks: array<u32>;

const SCAN_WORKGROUP_SIZE: u32 = 256u;

var<workgroup> scan_values: array<u32, 256>;

fn saturating_add(left: u32, right: u32) -> u32 {
    if (left > 0xffffffffu - right) {
        atomicOr(&counters.overflow, 1u);
        return 0xffffffffu;
    }
    return left + right;
}

fn instance_block_count() -> u32 {
    return frame.counts.x / SCAN_WORKGROUP_SIZE
        + select(0u, 1u, (frame.counts.x % SCAN_WORKGROUP_SIZE) != 0u);
}

fn linear_workgroup_id(group_id: vec3<u32>) -> u32 {
    return group_id.y * max(frame.limits.z, 1u) + group_id.x;
}

// Every invocation must call this function together. The two barriers per step prevent an
// invocation from overwriting a value before all other lanes have consumed the previous step.
fn inclusive_workgroup_scan(lane: u32) {
    for (var offset = 1u; offset < SCAN_WORKGROUP_SIZE; offset *= 2u) {
        var addend = 0u;
        if (lane >= offset) {
            addend = scan_values[lane - offset];
        }
        workgroupBarrier();
        scan_values[lane] = saturating_add(scan_values[lane], addend);
        workgroupBarrier();
    }
}

// Stage 1: independently scan 256 instance counts per workgroup. The offsets written here are
// relative to the block; stage 3 adds the prefix of all preceding blocks.
@compute @workgroup_size(256)
fn scan_blocks_local(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let block_id = linear_workgroup_id(group_id);
    if (block_id >= instance_block_count()) {
        return;
    }

    let instance_id = block_id * SCAN_WORKGROUP_SIZE + lane;
    let is_active = instance_id < frame.counts.x;
    var count = 0u;
    if (is_active) {
        count = classifications[instance_id].meshlet_count;
    }
    scan_values[lane] = count;
    workgroupBarrier();
    inclusive_workgroup_scan(lane);

    if (is_active) {
        var local_offset = 0u;
        if (lane != 0u) {
            local_offset = scan_values[lane - 1u];
        }
        classifications[instance_id].meshlet_offset = min(local_offset, frame.counts.z);
    }
    if (lane == SCAN_WORKGROUP_SIZE - 1u) {
        scan_blocks[block_id] = scan_values[lane];
    }
}

// Stage 2: scan the much smaller block-sum array. Its contiguous ranges are distributed evenly
// across all 256 lanes. Each lane scans its range, the workgroup scans the 256 range totals, and
// every lane adds its range prefix. At the default 262144-instance capacity each lane handles four
// block sums, rather than one invocation serially visiting every instance.
@compute @workgroup_size(256)
fn scan_block_sums(@builtin(local_invocation_index) lane: u32) {
    let block_count = instance_block_count();
    let base_length = block_count / SCAN_WORKGROUP_SIZE;
    let extra = block_count % SCAN_WORKGROUP_SIZE;
    let lane_length = base_length + select(0u, 1u, lane < extra);
    let lane_begin = lane * base_length + min(lane, extra);
    let lane_end = lane_begin + lane_length;

    var lane_total = 0u;
    for (var block_id = lane_begin; block_id < lane_end; block_id += 1u) {
        let block_sum = scan_blocks[block_id];
        scan_blocks[block_id] = lane_total;
        lane_total = saturating_add(lane_total, block_sum);
    }

    scan_values[lane] = lane_total;
    workgroupBarrier();
    inclusive_workgroup_scan(lane);

    var lane_offset = 0u;
    if (lane != 0u) {
        lane_offset = scan_values[lane - 1u];
    }
    storageBarrier();
    for (var block_id = lane_begin; block_id < lane_end; block_id += 1u) {
        scan_blocks[block_id] = saturating_add(lane_offset, scan_blocks[block_id]);
    }

    if (lane == 0u) {
        let total = scan_values[SCAN_WORKGROUP_SIZE - 1u];
        if (total > frame.counts.z) {
            atomicOr(&counters.overflow, 1u);
        }
        let clamped = min(total, frame.counts.z);
        atomicStore(&counters.candidate_count, clamped);

        // Avoid `clamped + 63` overflow when a device exposes a capacity near u32::MAX.
        let groups = clamped / 64u + select(0u, 1u, (clamped % 64u) != 0u);
        let max_dimension = max(frame.limits.z, 1u);
        candidate_dispatch.x = min(groups, max_dimension);
        let complete_rows = groups / max_dimension;
        let rounded_rows = complete_rows + select(0u, 1u, (groups % max_dimension) != 0u);
        candidate_dispatch.y = select(1u, rounded_rows, groups > max_dimension);
        candidate_dispatch.z = 1u;
        if (candidate_dispatch.y > max_dimension) {
            candidate_dispatch.y = max_dimension;
            atomicOr(&counters.overflow, 16u);
        }
    }
}

// Stage 3: turn each block-relative exclusive prefix into the global exclusive prefix expected by
// packet/work scatter. All arithmetic is saturated before the configured capacity clamp.
@compute @workgroup_size(256)
fn add_block_offsets(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let block_id = linear_workgroup_id(group_id);
    if (block_id >= instance_block_count()) {
        return;
    }

    let instance_id = block_id * SCAN_WORKGROUP_SIZE + lane;
    if (instance_id >= frame.counts.x) {
        return;
    }
    let global_offset = saturating_add(
        scan_blocks[block_id],
        classifications[instance_id].meshlet_offset,
    );
    classifications[instance_id].meshlet_offset = min(global_offset, frame.counts.z);
}

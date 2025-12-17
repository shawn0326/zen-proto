struct Counters {
    visible_count: atomic<u32>,
};

struct DispatchArgs {
    x: u32;
    y: u32;
    z: u32;
};

const WG_SIZE: u32 = 64u;

@group(0) @binding(0)
var<storage, read> counters: Counters;

@group(0) @binding(1)
var<storage, read_write> dispatch_args: DispatchArgs;

@compute @workgroup_size(1)
fn main() {
    let vc = atomicLoad(&counters.visible_count);
    let groups = (vc + WG_SIZE - 1u) / WG_SIZE;
    dispatch_args.x = groups;
    dispatch_args.y = 1u;
    dispatch_args.z = 1u;
}
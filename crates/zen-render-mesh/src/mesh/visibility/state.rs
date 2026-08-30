use super::{VisibilityHistory, VisibilityList};

/// Persistent GPU state shared by the Mesh visibility passes.
pub(crate) struct MeshVisibilityState {
    pub list_a: VisibilityList,
    pub list_b: VisibilityList,
    pub history: VisibilityHistory,
}

impl MeshVisibilityState {
    pub fn new(device: &wgpu::Device, max_instance_count: u32) -> Self {
        Self {
            list_a: VisibilityList::new(device, "list_a", max_instance_count),
            list_b: VisibilityList::new(device, "list_b", max_instance_count),
            history: VisibilityHistory::new(device, max_instance_count),
        }
    }
}

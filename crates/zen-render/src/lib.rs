//! Domain-independent FrameGraph render orchestration.
//!
//! [`RenderHost`] owns execution infrastructure while a concrete
//! [`FrameComposer`] owns render-domain state and records the frame recipe.

mod compose;
mod host;

pub use compose::{FrameComposeContext, FrameComposer, PresentTarget, RenderFrameInput};
pub use host::RenderHost;

#[cfg(feature = "snapshot")]
pub use zen_frame_graph::snapshot::{
    FrameGraphSnapshotV1, SnapshotExportError, SnapshotJsonError,
    to_json as frame_graph_snapshot_to_json, to_json_pretty as frame_graph_snapshot_to_json_pretty,
};
pub use zen_frame_graph::{
    FrameGraphError, GpuTimingNodeKind, GpuTimingNodeReport, GpuTimingReport,
    GpuTimingUnavailableReason,
};

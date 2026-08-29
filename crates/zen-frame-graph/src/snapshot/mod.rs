mod error;
mod export;
mod json;
mod types;

pub use error::{SnapshotExportError, SnapshotJsonError};
pub use export::{CreateFrameGraphSnapshotOptions, create_frame_graph_snapshot};
pub use json::{to_json, to_json_pretty};
pub use types::*;

pub const FRAME_GRAPH_SNAPSHOT_FORMAT: &str = "t3d.frame-graph-snapshot";
pub const FRAME_GRAPH_SNAPSHOT_VERSION: SnapshotVersion = SnapshotVersion { major: 1, minor: 0 };

#[cfg(test)]
mod tests;

use super::{FrameGraphSnapshotV1, SnapshotJsonError};

pub fn to_json(snapshot: &FrameGraphSnapshotV1) -> Result<String, SnapshotJsonError> {
    serde_json::to_string(snapshot).map_err(Into::into)
}

pub fn to_json_pretty(snapshot: &FrameGraphSnapshotV1) -> Result<String, SnapshotJsonError> {
    serde_json::to_string_pretty(snapshot).map_err(Into::into)
}

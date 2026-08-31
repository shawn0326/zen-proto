use std::{ffi::OsString, path::PathBuf, process::ExitCode};

use zen_demo::meshlet_benchmark::{
    MeshletAutoProfileFile, MeshletBenchmarkFileError, MeshletBenchmarkReport,
    indexed_regression_is_acceptable,
};

#[derive(Debug, thiserror::Error)]
enum ProfileCliError {
    #[error(
        "usage: meshlet-profile <indexed.json> <task-mesh.json> [--legacy legacy.json] -o <auto-profile.json>"
    )]
    Usage,
    #[error("missing value after -o/--output")]
    MissingOutput,
    #[error("unexpected argument {0:?}")]
    Unexpected(OsString),
    #[error("IndexedIndirect GPU p95 regresses by more than 10% against the legacy report")]
    IndexedRegression,
    #[error(transparent)]
    File(#[from] MeshletBenchmarkFileError),
    #[error(transparent)]
    Contract(#[from] zen_demo::meshlet_benchmark::MeshletBenchmarkError),
}

#[derive(Debug, Eq, PartialEq)]
struct ProfileArgs {
    indexed: PathBuf,
    task_mesh: PathBuf,
    legacy: Option<PathBuf>,
    output: PathBuf,
}

fn parse_args(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<ProfileArgs, ProfileCliError> {
    let mut arguments = arguments.into_iter();
    let indexed = arguments.next().ok_or(ProfileCliError::Usage)?;
    let task_mesh = arguments.next().ok_or(ProfileCliError::Usage)?;
    if indexed.to_string_lossy().starts_with('-') || task_mesh.to_string_lossy().starts_with('-') {
        return Err(ProfileCliError::Usage);
    }
    let mut output = None;
    let mut legacy = None;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("-o" | "--output") if output.is_none() => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or(ProfileCliError::MissingOutput)?,
                ));
            }
            Some("--legacy") if legacy.is_none() => {
                legacy = Some(PathBuf::from(
                    arguments.next().ok_or(ProfileCliError::Usage)?,
                ));
            }
            _ => return Err(ProfileCliError::Unexpected(argument)),
        }
    }
    Ok(ProfileArgs {
        indexed: indexed.into(),
        task_mesh: task_mesh.into(),
        legacy,
        output: output.ok_or(ProfileCliError::MissingOutput)?,
    })
}

fn run() -> Result<(), ProfileCliError> {
    let arguments = parse_args(std::env::args_os().skip(1))?;
    let indexed = MeshletBenchmarkReport::read_json_file(&arguments.indexed)?;
    let task_mesh = MeshletBenchmarkReport::read_json_file(&arguments.task_mesh)?;
    if let Some(path) = &arguments.legacy {
        let legacy = MeshletBenchmarkReport::read_json_file(path)?;
        if !indexed_regression_is_acceptable(&legacy, &indexed)? {
            return Err(ProfileCliError::IndexedRegression);
        }
        println!("legacy regression gate passed: {}", path.display());
    }
    let profile = MeshletAutoProfileFile::from_reports(&indexed, &task_mesh)?;
    profile.write_json_file(&arguments.output)?;
    println!(
        "wrote qualifying Vulkan Auto profile: {} (indexed p95={}ns, task-mesh p95={}ns)",
        arguments.output.display(),
        profile.profile.indexed_gpu_p95_ns,
        profile.profile.task_mesh_gpu_p95_ns,
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("meshlet-profile: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_reports_and_required_output() {
        assert_eq!(
            parse_args(
                [
                    "indexed.json",
                    "task.json",
                    "--legacy",
                    "legacy.json",
                    "--output",
                    "auto.json",
                ]
                .map(OsString::from),
            )
            .unwrap(),
            ProfileArgs {
                indexed: "indexed.json".into(),
                task_mesh: "task.json".into(),
                legacy: Some("legacy.json".into()),
                output: "auto.json".into(),
            }
        );
    }

    #[test]
    fn output_is_required() {
        assert!(matches!(
            parse_args(["indexed.json", "task.json"].map(OsString::from)),
            Err(ProfileCliError::MissingOutput)
        ));
    }
}

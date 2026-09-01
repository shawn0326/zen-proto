use std::{ffi::OsString, path::PathBuf, process::ExitCode};

use zen_demo::{
    gltf_loader::{LoadGltfOptions, load_gltf},
    meshlet_support::{MeshletCacheStatus, load_or_build_zenmesh, raw_static_meshes},
};
use zen_render_mesh::MeshletBuildConfig;

#[derive(Debug, thiserror::Error)]
enum BuildCliError {
    #[error(
        "usage: build-zenmesh <input.gltf|input.glb> [-o output.zenmesh] [--max-vertices N] [--max-triangles N] [--task-packet N] [--max-lods N] [--lod-ratio F] [--min-lod-triangles N] [--cone-weight F] [--flip-v] [--bake-node-transform]"
    )]
    Usage,
    #[error("missing value after {0}")]
    MissingValue(&'static str),
    #[error("invalid value {value:?} for {option}")]
    InvalidValue {
        option: &'static str,
        value: OsString,
    },
    #[error("unexpected argument {0:?}")]
    Unexpected(OsString),
    #[error(transparent)]
    Asset(#[from] zen_demo::meshlet_support::MeshletAssetCacheError),
    #[error(transparent)]
    BuildConfig(#[from] zen_render_mesh::meshlet::MeshletAssetError),
    #[error(transparent)]
    Gltf(#[from] zen_demo::gltf_loader::GltfLoadError),
}

struct BuildArgs {
    input: PathBuf,
    output: PathBuf,
    config: MeshletBuildConfig,
    loader: LoadGltfOptions,
}

fn take_value(
    arguments: &mut impl Iterator<Item = OsString>,
    option: &'static str,
) -> Result<OsString, BuildCliError> {
    arguments.next().ok_or(BuildCliError::MissingValue(option))
}

fn parse_value<T: std::str::FromStr>(
    value: OsString,
    option: &'static str,
) -> Result<T, BuildCliError> {
    value
        .to_str()
        .and_then(|value| value.parse().ok())
        .ok_or(BuildCliError::InvalidValue { option, value })
}

fn parse_args(arguments: impl IntoIterator<Item = OsString>) -> Result<BuildArgs, BuildCliError> {
    let mut arguments = arguments.into_iter();
    let input = arguments.next().ok_or(BuildCliError::Usage)?;
    if input.to_string_lossy().starts_with('-') {
        return Err(BuildCliError::Usage);
    }
    let input = PathBuf::from(input);
    let mut output = None;
    let mut config = MeshletBuildConfig::default();
    let mut loader = LoadGltfOptions {
        global_scale: 1.0,
        flip_v: false,
        bake_node_transform: false,
    };

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("-o" | "--output") => {
                output = Some(PathBuf::from(take_value(&mut arguments, "--output")?));
            }
            Some("--max-vertices") => {
                config.max_meshlet_vertices = parse_value(
                    take_value(&mut arguments, "--max-vertices")?,
                    "--max-vertices",
                )?;
            }
            Some("--max-triangles") => {
                config.max_meshlet_triangles = parse_value(
                    take_value(&mut arguments, "--max-triangles")?,
                    "--max-triangles",
                )?;
            }
            Some("--task-packet") => {
                config.task_packet_meshlets = parse_value(
                    take_value(&mut arguments, "--task-packet")?,
                    "--task-packet",
                )?;
            }
            Some("--max-lods") => {
                config.max_lods =
                    parse_value(take_value(&mut arguments, "--max-lods")?, "--max-lods")?;
            }
            Some("--lod-ratio") => {
                config.lod_target_ratio =
                    parse_value(take_value(&mut arguments, "--lod-ratio")?, "--lod-ratio")?;
            }
            Some("--min-lod-triangles") => {
                config.min_lod_triangles = parse_value(
                    take_value(&mut arguments, "--min-lod-triangles")?,
                    "--min-lod-triangles",
                )?;
            }
            Some("--cone-weight") => {
                config.meshlet_cone_weight = parse_value(
                    take_value(&mut arguments, "--cone-weight")?,
                    "--cone-weight",
                )?;
            }
            Some("--flip-v") => loader.flip_v = true,
            Some("--bake-node-transform") => loader.bake_node_transform = true,
            _ => return Err(BuildCliError::Unexpected(argument)),
        }
    }

    let output = output.unwrap_or_else(|| input.with_extension("zenmesh"));
    Ok(BuildArgs {
        input,
        output,
        config,
        loader,
    })
}

fn run() -> Result<(), BuildCliError> {
    let arguments = parse_args(std::env::args_os().skip(1))?;
    arguments.config.validate()?;
    let model = load_gltf(&arguments.input, arguments.loader)?;
    let meshes = raw_static_meshes(&model)?;
    let result = load_or_build_zenmesh(&arguments.output, &meshes, arguments.config)?;
    println!(
        "{} {}: meshes={}, lods={}, meshlets={}, vertices={}, fallback-indices={}, source={}, build={}",
        match result.status {
            MeshletCacheStatus::Hit => "reused",
            MeshletCacheStatus::RebuiltMissing | MeshletCacheStatus::RebuiltStale => "built",
        },
        arguments.output.display(),
        result.asset.meshes().len(),
        result.asset.lods().len(),
        result.asset.meshlets().len(),
        result.asset.positions().len(),
        result.asset.fallback_indices().len(),
        result.asset.cache_key().source_hash,
        result.asset.cache_key().build_hash,
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("build-zenmesh: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_agreed_asset_contract() {
        let args = parse_args([OsString::from("scene.gltf")]).unwrap();
        assert_eq!(args.output, PathBuf::from("scene.zenmesh"));
        assert_eq!(args.config.max_meshlet_vertices, 64);
        assert_eq!(args.config.max_meshlet_triangles, 64);
        assert_eq!(args.config.task_packet_meshlets, 32);
        assert_eq!(args.config.lod_target_ratio, 0.5);
        assert_eq!(args.config.min_lod_triangles, 128);
    }

    #[test]
    fn parses_all_build_controls() {
        let args = parse_args(
            [
                "scene.glb",
                "-o",
                "cache/custom.zenmesh",
                "--max-vertices",
                "32",
                "--max-triangles",
                "48",
                "--task-packet",
                "16",
                "--max-lods",
                "4",
                "--lod-ratio",
                "0.4",
                "--min-lod-triangles",
                "64",
                "--cone-weight",
                "0.75",
                "--flip-v",
                "--bake-node-transform",
            ]
            .map(OsString::from),
        )
        .unwrap();
        assert_eq!(args.output, PathBuf::from("cache/custom.zenmesh"));
        assert_eq!(args.config.max_meshlet_vertices, 32);
        assert_eq!(args.config.max_meshlet_triangles, 48);
        assert_eq!(args.config.task_packet_meshlets, 16);
        assert_eq!(args.config.max_lods, 4);
        assert_eq!(args.config.lod_target_ratio, 0.4);
        assert_eq!(args.config.min_lod_triangles, 64);
        assert_eq!(args.config.meshlet_cone_weight, 0.75);
        assert!(args.loader.flip_v);
        assert!(args.loader.bake_node_transform);
    }
}

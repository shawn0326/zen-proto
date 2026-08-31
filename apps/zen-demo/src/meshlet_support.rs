use std::{
    ffi::OsString,
    fmt, fs, io,
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
};

use zen_render_mesh::{
    MeshletBackend, MeshletBuildConfig, MeshletRendererConfig, MeshletSceneAsset,
    meshlet::{MeshletAssetError, MeshletCacheKey, MeshletPsoClass, RawStaticMesh},
};

use crate::gltf_loader::{LoadedAlphaMode, LoadedGltfModel};
use crate::meshlet_benchmark::MeshletBenchmarkRenderer;

/// Renderer names exposed by the Vulkan meshlet demo.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DemoRenderer {
    Legacy,
    IndexedIndirect,
    MeshOnly,
    TaskMesh,
    #[default]
    Auto,
}

impl DemoRenderer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::IndexedIndirect => "indexed",
            Self::MeshOnly => "mesh",
            Self::TaskMesh => "task-mesh",
            Self::Auto => "auto",
        }
    }

    pub const fn meshlet_backend(self) -> Option<MeshletBackend> {
        match self {
            Self::Legacy => None,
            Self::IndexedIndirect => Some(MeshletBackend::IndexedIndirect),
            Self::MeshOnly => Some(MeshletBackend::MeshOnly),
            Self::TaskMesh => Some(MeshletBackend::TaskMesh),
            Self::Auto => Some(MeshletBackend::Auto),
        }
    }

    pub fn meshlet_config(self) -> Option<MeshletRendererConfig> {
        self.meshlet_backend().map(|backend| MeshletRendererConfig {
            backend,
            ..Default::default()
        })
    }

    pub const fn benchmark_renderer(self) -> Option<MeshletBenchmarkRenderer> {
        match self {
            Self::Legacy => Some(MeshletBenchmarkRenderer::Legacy),
            Self::IndexedIndirect => Some(MeshletBenchmarkRenderer::Indexed),
            Self::MeshOnly => Some(MeshletBenchmarkRenderer::MeshOnly),
            Self::TaskMesh => Some(MeshletBenchmarkRenderer::TaskMesh),
            Self::Auto => None,
        }
    }
}

impl fmt::Display for DemoRenderer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for DemoRenderer {
    type Err = ParseDemoRendererError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "indexed" | "indexed-indirect" => Ok(Self::IndexedIndirect),
            "mesh" | "mesh-only" => Ok(Self::MeshOnly),
            "task-mesh" | "task_mesh" => Ok(Self::TaskMesh),
            "auto" => Ok(Self::Auto),
            _ => Err(ParseDemoRendererError(value.to_owned())),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("unknown renderer {0:?}; expected legacy, indexed, mesh, task-mesh, or auto")]
pub struct ParseDemoRendererError(String);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshletDemoArgs {
    pub renderer: DemoRenderer,
    pub model: Option<PathBuf>,
    pub cache: Option<PathBuf>,
    /// Enables the fixed 1920x1080 executable benchmark and writes its JSON report here.
    pub benchmark_out: Option<PathBuf>,
    /// Identity-bound profile consumed only by `--renderer auto`.
    pub auto_profile: Option<PathBuf>,
    /// Explicit acknowledgement that the selected benchmark scene is geometry-bound.
    pub geometry_bound: bool,
}

impl Default for MeshletDemoArgs {
    fn default() -> Self {
        Self {
            renderer: DemoRenderer::Auto,
            model: None,
            cache: None,
            benchmark_out: None,
            auto_profile: None,
            geometry_bound: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParseMeshletDemoArgsError {
    #[error("missing value after {0}")]
    MissingValue(&'static str),
    #[error("unexpected argument {0:?}")]
    Unexpected(OsString),
    #[error(transparent)]
    Renderer(#[from] ParseDemoRendererError),
    #[error("--benchmark-out requires a concrete renderer; auto is selected from a profile")]
    BenchmarkAuto,
    #[error("--auto-profile is only valid with --renderer auto")]
    ProfileWithoutAuto,
}

/// Parses `--renderer legacy|indexed|mesh|task-mesh|auto`, an optional model path, and an optional
/// explicit cache path. It deliberately has no process-global side effects so tests and alternate
/// launchers can reuse it.
pub fn parse_meshlet_demo_args(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<MeshletDemoArgs, ParseMeshletDemoArgsError> {
    let mut result = MeshletDemoArgs::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        if argument == "--renderer" {
            let value = arguments
                .next()
                .ok_or(ParseMeshletDemoArgsError::MissingValue("--renderer"))?;
            result.renderer = value
                .to_str()
                .ok_or_else(|| ParseMeshletDemoArgsError::Unexpected(value.clone()))?
                .parse()?;
        } else if argument == "--cache" {
            result.cache = Some(PathBuf::from(
                arguments
                    .next()
                    .ok_or(ParseMeshletDemoArgsError::MissingValue("--cache"))?,
            ));
        } else if argument == "--benchmark-out" {
            result.benchmark_out =
                Some(PathBuf::from(arguments.next().ok_or(
                    ParseMeshletDemoArgsError::MissingValue("--benchmark-out"),
                )?));
        } else if argument == "--auto-profile" {
            result.auto_profile =
                Some(PathBuf::from(arguments.next().ok_or(
                    ParseMeshletDemoArgsError::MissingValue("--auto-profile"),
                )?));
        } else if argument == "--geometry-bound" {
            result.geometry_bound = true;
        } else if argument.to_string_lossy().starts_with('-') || result.model.is_some() {
            return Err(ParseMeshletDemoArgsError::Unexpected(argument));
        } else {
            result.model = Some(PathBuf::from(argument));
        }
    }
    if result.benchmark_out.is_some() && result.renderer == DemoRenderer::Auto {
        return Err(ParseMeshletDemoArgsError::BenchmarkAuto);
    }
    if result.auto_profile.is_some() && result.renderer != DemoRenderer::Auto {
        return Err(ParseMeshletDemoArgsError::ProfileWithoutAuto);
    }
    Ok(result)
}

#[derive(Debug, thiserror::Error)]
pub enum MeshletAssetCacheError {
    #[error(transparent)]
    Asset(#[from] MeshletAssetError),
    #[error("failed to read meshlet cache {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write meshlet cache {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("mesh {mesh_id} is referenced with conflicting materials {first} and {second}")]
    ConflictingMaterial {
        mesh_id: u32,
        first: u32,
        second: u32,
    },
    #[error(
        "mesh {mesh_id} uses unsupported glTF alpha mode {alpha_mode:?}; meshlets currently require opaque geometry"
    )]
    UnsupportedAlphaMode {
        mesh_id: u32,
        alpha_mode: LoadedAlphaMode,
    },
    #[error("mesh {mesh_id} has no matching glTF surface metadata")]
    MissingSurfaceMetadata { mesh_id: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshletCacheStatus {
    Hit,
    RebuiltMissing,
    RebuiltStale,
}

pub struct CachedMeshletAsset {
    pub asset: MeshletSceneAsset,
    pub status: MeshletCacheStatus,
}

/// Converts the existing glTF loader's legacy mesh records to the shared static meshlet source
/// format. This preserves vertex attributes and the global material assignment without coupling
/// the asset builder to the renderer implementation.
pub fn raw_static_meshes(
    model: &LoadedGltfModel,
) -> Result<Vec<RawStaticMesh>, MeshletAssetCacheError> {
    if model.mesh_surfaces.len() != model.meshes.len() {
        let mesh_id = model.mesh_surfaces.len().min(model.meshes.len()) as u32;
        return Err(MeshletAssetCacheError::MissingSurfaceMetadata { mesh_id });
    }
    for (mesh_id, surface) in model.mesh_surfaces.iter().enumerate() {
        if surface.alpha_mode != LoadedAlphaMode::Opaque {
            return Err(MeshletAssetCacheError::UnsupportedAlphaMode {
                mesh_id: mesh_id as u32,
                alpha_mode: surface.alpha_mode,
            });
        }
    }
    let mut materials = vec![None; model.meshes.len()];
    for instance in &model.instances {
        let Some(slot) = materials.get_mut(instance.mesh_id as usize) else {
            continue;
        };
        match *slot {
            Some(first) if first != instance.material_id => {
                return Err(MeshletAssetCacheError::ConflictingMaterial {
                    mesh_id: instance.mesh_id,
                    first,
                    second: instance.material_id,
                });
            }
            Some(_) => {}
            None => *slot = Some(instance.material_id),
        }
    }

    Ok(model
        .meshes
        .iter()
        .enumerate()
        .map(|(mesh_id, mesh)| RawStaticMesh {
            positions: mesh
                .vertices
                .iter()
                .map(|vertex| vertex.position.truncate().to_array())
                .collect(),
            normals: mesh
                .vertices
                .iter()
                .map(|vertex| vertex.normal.truncate().to_array())
                .collect(),
            tex_coords: mesh
                .vertices
                .iter()
                .map(|vertex| vertex.uv.to_array())
                .collect(),
            colors: mesh
                .vertices
                .iter()
                .map(|vertex| vertex.color.to_array())
                .collect(),
            indices: mesh.indices.iter().map(|&index| u32::from(index)).collect(),
            material_slot: materials[mesh_id].unwrap_or(0),
            pso_class: if model.mesh_surfaces[mesh_id].double_sided {
                MeshletPsoClass::OpaqueTwoSided
            } else {
                MeshletPsoClass::OpaqueBackface
            },
        })
        .collect())
}

pub fn meshlet_cache_key(meshes: &[RawStaticMesh], config: MeshletBuildConfig) -> MeshletCacheKey {
    MeshletCacheKey {
        source_hash: MeshletSceneAsset::source_hash(meshes),
        build_hash: config.build_hash(),
    }
}

/// Loads a matching `.zenmesh` cache, or deterministically rebuilds and replaces a missing/stale
/// entry. Corrupt data is treated as stale; unrelated filesystem errors remain visible.
pub fn load_or_build_zenmesh(
    cache_path: impl AsRef<Path>,
    meshes: &[RawStaticMesh],
    config: MeshletBuildConfig,
) -> Result<CachedMeshletAsset, MeshletAssetCacheError> {
    let cache_path = cache_path.as_ref();
    let expected = meshlet_cache_key(meshes, config);
    let status = match fs::read(cache_path) {
        Ok(bytes) => match MeshletSceneAsset::decode_cached(&bytes, expected) {
            Ok(asset) => {
                return Ok(CachedMeshletAsset {
                    asset,
                    status: MeshletCacheStatus::Hit,
                });
            }
            Err(_) => MeshletCacheStatus::RebuiltStale,
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => MeshletCacheStatus::RebuiltMissing,
        Err(source) => {
            return Err(MeshletAssetCacheError::Read {
                path: cache_path.to_owned(),
                source,
            });
        }
    };

    let asset = MeshletSceneAsset::build(meshes, config)?;
    let encoded = asset.encode_zenmesh()?;
    if let Some(parent) = cache_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| MeshletAssetCacheError::Write {
            path: cache_path.to_owned(),
            source,
        })?;
    }
    let parent = cache_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|source| {
        MeshletAssetCacheError::Write {
            path: cache_path.to_owned(),
            source,
        }
    })?;
    temporary
        .write_all(&encoded)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|source| MeshletAssetCacheError::Write {
            path: cache_path.to_owned(),
            source,
        })?;
    temporary
        .persist(cache_path)
        .map_err(|error| MeshletAssetCacheError::Write {
            path: cache_path.to_owned(),
            source: error.error,
        })?;

    Ok(CachedMeshletAsset { asset, status })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec4};
    use zen_render_mesh::{Instance, Mesh, Vertex};

    fn triangle_model() -> LoadedGltfModel {
        let vertices = [
            ([0.0, 0.0, 0.0], [0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 1.0]),
        ]
        .map(|(position, uv)| Vertex {
            position: Vec4::new(position[0], position[1], position[2], 1.0),
            normal: Vec4::new(0.0, 0.0, 1.0, 0.0),
            color: Vec4::ONE,
            uv: Vec2::from_array(uv),
        });
        LoadedGltfModel {
            meshes: vec![Mesh {
                vertices: vertices.into(),
                indices: vec![0, 1, 2],
            }],
            mesh_surfaces: vec![crate::gltf_loader::LoadedMeshSurface {
                alpha_mode: LoadedAlphaMode::Opaque,
                double_sided: false,
            }],
            materials: Vec::new(),
            instances: vec![Instance {
                transform: glam::Mat4::IDENTITY,
                mesh_id: 0,
                material_id: 7,
                _pad: [0; 2],
            }],
            textures: Vec::new(),
        }
    }

    #[test]
    fn renderer_cli_names_are_stable() {
        for (name, expected) in [
            ("legacy", DemoRenderer::Legacy),
            ("indexed", DemoRenderer::IndexedIndirect),
            ("mesh", DemoRenderer::MeshOnly),
            ("task-mesh", DemoRenderer::TaskMesh),
            ("auto", DemoRenderer::Auto),
        ] {
            assert_eq!(name.parse(), Ok(expected));
            assert_eq!(expected.as_str(), name);
        }
    }

    #[test]
    fn demo_arguments_accept_options_in_any_order() {
        let parsed = parse_meshlet_demo_args([
            OsString::from("scene.gltf"),
            OsString::from("--renderer"),
            OsString::from("task-mesh"),
            OsString::from("--cache"),
            OsString::from("scene.zenmesh"),
            OsString::from("--benchmark-out"),
            OsString::from("task.json"),
            OsString::from("--geometry-bound"),
        ])
        .unwrap();
        assert_eq!(parsed.renderer, DemoRenderer::TaskMesh);
        assert_eq!(parsed.model, Some(PathBuf::from("scene.gltf")));
        assert_eq!(parsed.cache, Some(PathBuf::from("scene.zenmesh")));
        assert_eq!(parsed.benchmark_out, Some(PathBuf::from("task.json")));
        assert!(parsed.geometry_bound);
    }

    #[test]
    fn auto_profile_and_benchmark_modes_are_mutually_well_scoped() {
        let auto = parse_meshlet_demo_args([
            OsString::from("--renderer"),
            OsString::from("auto"),
            OsString::from("--auto-profile"),
            OsString::from("auto.json"),
        ])
        .unwrap();
        assert_eq!(auto.auto_profile, Some(PathBuf::from("auto.json")));

        assert_eq!(
            parse_meshlet_demo_args([
                OsString::from("--renderer"),
                OsString::from("auto"),
                OsString::from("--benchmark-out"),
                OsString::from("bad.json"),
            ]),
            Err(ParseMeshletDemoArgsError::BenchmarkAuto)
        );
        assert_eq!(
            parse_meshlet_demo_args([
                OsString::from("--renderer"),
                OsString::from("indexed"),
                OsString::from("--auto-profile"),
                OsString::from("bad.json"),
            ]),
            Err(ParseMeshletDemoArgsError::ProfileWithoutAuto)
        );
    }

    #[test]
    fn legacy_mesh_conversion_preserves_material_and_attributes() {
        let raw = raw_static_meshes(&triangle_model()).unwrap();
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].material_slot, 7);
        assert_eq!(raw[0].indices, [0, 1, 2]);
        assert_eq!(raw[0].tex_coords[1], [1.0, 0.0]);
    }

    #[test]
    fn conversion_maps_double_sided_and_rejects_non_opaque_surfaces() {
        let mut model = triangle_model();
        model.mesh_surfaces[0].double_sided = true;
        let raw = raw_static_meshes(&model).unwrap();
        assert_eq!(raw[0].pso_class, MeshletPsoClass::OpaqueTwoSided);

        model.mesh_surfaces[0].alpha_mode = LoadedAlphaMode::Mask;
        assert!(matches!(
            raw_static_meshes(&model),
            Err(MeshletAssetCacheError::UnsupportedAlphaMode {
                mesh_id: 0,
                alpha_mode: LoadedAlphaMode::Mask,
            })
        ));
    }

    #[test]
    fn cache_rebuilds_then_hits_and_rejects_changed_build_parameters() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("triangle.zenmesh");
        let raw = raw_static_meshes(&triangle_model()).unwrap();

        let first = load_or_build_zenmesh(&cache, &raw, MeshletBuildConfig::default()).unwrap();
        assert_eq!(first.status, MeshletCacheStatus::RebuiltMissing);
        let second = load_or_build_zenmesh(&cache, &raw, MeshletBuildConfig::default()).unwrap();
        assert_eq!(second.status, MeshletCacheStatus::Hit);

        let changed = MeshletBuildConfig {
            max_lods: 1,
            ..Default::default()
        };
        let third = load_or_build_zenmesh(&cache, &raw, changed).unwrap();
        assert_eq!(third.status, MeshletCacheStatus::RebuiltStale);
        assert_eq!(third.asset.config(), changed);
    }

    #[test]
    fn cache_rebuilds_corrupt_bytes_and_changed_source() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("triangle.zenmesh");
        let raw = raw_static_meshes(&triangle_model()).unwrap();
        load_or_build_zenmesh(&cache, &raw, MeshletBuildConfig::default()).unwrap();

        fs::write(&cache, b"not-a-zenmesh").unwrap();
        let corrupt = load_or_build_zenmesh(&cache, &raw, MeshletBuildConfig::default()).unwrap();
        assert_eq!(corrupt.status, MeshletCacheStatus::RebuiltStale);

        let mut changed = raw;
        changed[0].positions[0][0] = 0.25;
        let source =
            load_or_build_zenmesh(&cache, &changed, MeshletBuildConfig::default()).unwrap();
        assert_eq!(source.status, MeshletCacheStatus::RebuiltStale);
        assert_eq!(
            source.asset.cache_key().source_hash,
            MeshletSceneAsset::source_hash(&changed)
        );
    }
}

use std::path::Path;

use zen_render_mesh::{
    Instance, Material, MaterialTextureBinding, Mesh, Texture, TextureAddressMode,
    TextureMagFilter, TextureMinFilter, TextureSampler, Vertex,
};

pub struct LoadedGltfModel {
    pub meshes: Vec<Mesh>,
    /// Surface semantics aligned one-to-one with `meshes`.
    pub mesh_surfaces: Vec<LoadedMeshSurface>,
    pub materials: Vec<Material>,
    pub instances: Vec<Instance>,
    pub textures: Vec<Texture>,
    pub samplers: Vec<TextureSampler>,
}

#[derive(Debug, thiserror::Error)]
pub enum GltfLoadError {
    #[error("failed to import glTF {path}: {source}")]
    Import {
        path: String,
        #[source]
        source: gltf::Error,
    },
    #[error("material {material} {slot} uses TEXCOORD_{tex_coord}; only TEXCOORD_0 is supported")]
    UnsupportedTexCoord {
        material: usize,
        slot: &'static str,
        tex_coord: u32,
    },
    #[error(
        "KHR_texture_transform is not supported; refusing to render transformed UVs incorrectly"
    )]
    UnsupportedTextureTransform,
}

#[derive(Clone, Copy)]
pub struct LoadGltfOptions {
    pub global_scale: f32,
    pub flip_v: bool,
    /// Bake node transforms into vertex positions/normals and emit identity instances.
    pub bake_node_transform: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadedAlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadedMeshSurface {
    pub alpha_mode: LoadedAlphaMode,
    pub double_sided: bool,
}

impl Default for LoadGltfOptions {
    fn default() -> Self {
        Self {
            global_scale: 1.0,
            flip_v: true,
            bake_node_transform: true,
        }
    }
}

pub fn load_gltf(
    path: impl AsRef<Path>,
    options: LoadGltfOptions,
) -> Result<LoadedGltfModel, GltfLoadError> {
    let path = path.as_ref();
    let (document, buffers, images) =
        gltf::import(path).map_err(|source| GltfLoadError::Import {
            path: path.display().to_string(),
            source,
        })?;
    validate_supported_extensions(&document)?;

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .unwrap_or_else(|| panic!("glTF has no scenes: {}", path.display()));

    // Textures: reserve 0 for white fallback.
    let mut textures: Vec<Texture> = Vec::with_capacity(images.len() + 1);
    textures.push(Texture::white_1x1());
    let texture_id_for_image: Vec<u32> = (0..images.len()).map(|i| (i as u32) + 1).collect();

    for (i, img) in images.iter().enumerate() {
        let (rgba, width, height) = convert_gltf_image_to_rgba8(img);
        textures.push(Texture {
            width,
            height,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            pixels: rgba,
        });
        let _ = i;
    }

    // Reserve sampler 0 for the glTF default sampler. Explicit sampler objects remain a
    // separate table, so two glTF textures can share an image but use different sampling.
    let mut samplers = Vec::with_capacity(document.samplers().len() + 1);
    samplers.push(TextureSampler::default());
    samplers.extend(document.samplers().map(texture_sampler_from_gltf));
    let texture_bindings = build_texture_bindings(&document, &texture_id_for_image);

    // Materials: baseColorFactor * baseColorTexture (if present).
    let mut materials: Vec<Material> = Vec::with_capacity(document.materials().len() + 1);
    for m in document.materials() {
        let material_index = m.index().unwrap_or(materials.len());
        let pbr = m.pbr_metallic_roughness();
        let factor = pbr.base_color_factor();
        let albedo = texture_binding(
            pbr.base_color_texture(),
            &texture_bindings,
            material_index,
            "baseColorTexture",
        )?;

        let emissive_factor = m.emissive_factor();
        let emissive_rgb =
            glam::Vec3::new(emissive_factor[0], emissive_factor[1], emissive_factor[2]);

        // If emissiveTexture is missing, fall back to white so emissive_factor can still work.
        let emissive = texture_binding(
            m.emissive_texture(),
            &texture_bindings,
            material_index,
            "emissiveTexture",
        )?;

        // AO (occlusion) support
        let mut occlusion = MaterialTextureBinding::default();
        let mut ao_strength = 1.0f32;
        if let Some(occ) = m.occlusion_texture() {
            if occ.tex_coord() != 0 {
                return Err(GltfLoadError::UnsupportedTexCoord {
                    material: material_index,
                    slot: "occlusionTexture",
                    tex_coord: occ.tex_coord(),
                });
            }
            occlusion = texture_bindings[occ.texture().index()];
            ao_strength = occ.strength();
        }

        let emissive_ao =
            glam::Vec4::new(emissive_rgb.x, emissive_rgb.y, emissive_rgb.z, ao_strength);

        materials.push(Material {
            albedo_factor: glam::Vec4::new(factor[0], factor[1], factor[2], factor[3]),
            emissive_ao,
            albedo,
            emissive,
            occlusion,
            _padding: [0; 2],
        });
    }

    // Ensure we have a valid fallback material.
    let default_material_id = if materials.is_empty() {
        materials.push(Material {
            albedo_factor: glam::Vec4::ONE,
            emissive_ao: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            albedo: MaterialTextureBinding::default(),
            emissive: MaterialTextureBinding::default(),
            occlusion: MaterialTextureBinding::default(),
            _padding: [0; 2],
        });
        0u32
    } else {
        // Append a dedicated default (white) so primitives with no material index are stable.
        let id = materials.len() as u32;
        materials.push(Material {
            albedo_factor: glam::Vec4::ONE,
            emissive_ao: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            albedo: MaterialTextureBinding::default(),
            emissive: MaterialTextureBinding::default(),
            occlusion: MaterialTextureBinding::default(),
            _padding: [0; 2],
        });
        id
    };

    let mut loaded = LoadedSceneOutput::default();

    let scale_m = glam::Mat4::from_scale(glam::Vec3::splat(options.global_scale));
    for node in scene.nodes() {
        load_node_recursive(
            &node,
            &scale_m,
            options,
            &buffers,
            default_material_id,
            &mut loaded,
        );
    }

    Ok(LoadedGltfModel {
        meshes: loaded.meshes,
        mesh_surfaces: loaded.mesh_surfaces,
        materials,
        instances: loaded.instances,
        textures,
        samplers,
    })
}

fn validate_supported_extensions(document: &gltf::Document) -> Result<(), GltfLoadError> {
    if document
        .extensions_used()
        .any(|extension| extension == "KHR_texture_transform")
    {
        return Err(GltfLoadError::UnsupportedTextureTransform);
    }
    Ok(())
}

fn build_texture_bindings(
    document: &gltf::Document,
    texture_id_for_image: &[u32],
) -> Vec<MaterialTextureBinding> {
    document
        .textures()
        .map(|texture| MaterialTextureBinding {
            texture_id: texture_id_for_image[texture.source().index()],
            sampler_id: texture
                .sampler()
                .index()
                .map_or(0, |index| index as u32 + 1),
        })
        .collect()
}

fn texture_binding(
    info: Option<gltf::texture::Info<'_>>,
    texture_bindings: &[MaterialTextureBinding],
    material: usize,
    slot: &'static str,
) -> Result<MaterialTextureBinding, GltfLoadError> {
    let Some(info) = info else {
        return Ok(MaterialTextureBinding::default());
    };
    if info.tex_coord() != 0 {
        return Err(GltfLoadError::UnsupportedTexCoord {
            material,
            slot,
            tex_coord: info.tex_coord(),
        });
    }
    Ok(texture_bindings[info.texture().index()])
}

fn texture_sampler_from_gltf(sampler: gltf::texture::Sampler<'_>) -> TextureSampler {
    let address = |mode| match mode {
        gltf::texture::WrappingMode::ClampToEdge => TextureAddressMode::ClampToEdge,
        gltf::texture::WrappingMode::Repeat => TextureAddressMode::Repeat,
        gltf::texture::WrappingMode::MirroredRepeat => TextureAddressMode::MirroredRepeat,
    };
    let mag_filter = match sampler
        .mag_filter()
        .unwrap_or(gltf::texture::MagFilter::Linear)
    {
        gltf::texture::MagFilter::Nearest => TextureMagFilter::Nearest,
        gltf::texture::MagFilter::Linear => TextureMagFilter::Linear,
    };
    let min_filter = match sampler
        .min_filter()
        .unwrap_or(gltf::texture::MinFilter::LinearMipmapLinear)
    {
        gltf::texture::MinFilter::Nearest => TextureMinFilter::Nearest,
        gltf::texture::MinFilter::Linear => TextureMinFilter::Linear,
        gltf::texture::MinFilter::NearestMipmapNearest => TextureMinFilter::NearestMipmapNearest,
        gltf::texture::MinFilter::LinearMipmapNearest => TextureMinFilter::LinearMipmapNearest,
        gltf::texture::MinFilter::NearestMipmapLinear => TextureMinFilter::NearestMipmapLinear,
        gltf::texture::MinFilter::LinearMipmapLinear => TextureMinFilter::LinearMipmapLinear,
    };
    TextureSampler {
        address_mode_u: address(sampler.wrap_s()),
        address_mode_v: address(sampler.wrap_t()),
        mag_filter,
        min_filter,
    }
}

#[derive(Default)]
struct LoadedSceneOutput {
    meshes: Vec<Mesh>,
    mesh_surfaces: Vec<LoadedMeshSurface>,
    instances: Vec<Instance>,
}

fn load_node_recursive(
    node: &gltf::Node,
    parent_world: &glam::Mat4,
    options: LoadGltfOptions,
    buffers: &[gltf::buffer::Data],
    default_material_id: u32,
    loaded: &mut LoadedSceneOutput,
) {
    let local = mat4_from_gltf(node.transform().matrix());
    let world = *parent_world * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                panic!(
                    "Unsupported primitive mode {:?} (only Triangles supported)",
                    primitive.mode()
                );
            }

            let material_id = primitive
                .material()
                .index()
                .map(|i| i as u32)
                .unwrap_or(default_material_id);
            let material = primitive.material();
            let alpha_mode = match material.alpha_mode() {
                gltf::material::AlphaMode::Opaque => LoadedAlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => LoadedAlphaMode::Mask,
                gltf::material::AlphaMode::Blend => LoadedAlphaMode::Blend,
            };

            let engine_mesh = build_engine_mesh_from_primitive(&primitive, buffers, world, options);
            let mesh_id = loaded.meshes.len() as u32;
            loaded.meshes.push(engine_mesh);
            loaded.mesh_surfaces.push(LoadedMeshSurface {
                alpha_mode,
                double_sided: material.double_sided(),
            });

            // Emit identity instance if we baked, otherwise keep node transform.
            let transform = if options.bake_node_transform {
                glam::Mat4::IDENTITY
            } else {
                world
            };
            loaded.instances.push(Instance {
                transform,
                mesh_id,
                material_id,
                _pad: [0; 2],
            });
        }
    }

    for child in node.children() {
        load_node_recursive(
            &child,
            &world,
            options,
            buffers,
            default_material_id,
            loaded,
        );
    }
}

fn build_engine_mesh_from_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    world: glam::Mat4,
    options: LoadGltfOptions,
) -> Mesh {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));

    let positions: Vec<glam::Vec3> = reader
        .read_positions()
        .unwrap_or_else(|| panic!("glTF primitive missing POSITION"))
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();

    let vertex_count = positions.len();
    assert!(vertex_count > 0, "Primitive has zero vertices");
    assert!(
        vertex_count <= (u16::MAX as usize),
        "Primitive vertex_count {} exceeds u16 limit",
        vertex_count
    );

    let mut indices_u32: Vec<u32> = if let Some(read_indices) = reader.read_indices() {
        read_indices.into_u32().collect()
    } else {
        (0..vertex_count as u32).collect()
    };

    assert!(
        indices_u32.len().is_multiple_of(3),
        "Triangle primitive has non-multiple-of-3 indices"
    );
    let max_index = indices_u32.iter().copied().max().unwrap_or(0) as usize;
    assert!(
        max_index < vertex_count,
        "Index out of bounds: max_index={} vertex_count={}",
        max_index,
        vertex_count
    );
    assert!(max_index <= (u16::MAX as usize), "Indices exceed u16 limit");
    let indices: Vec<u16> = indices_u32.drain(..).map(|i| i as u16).collect();

    let normals_opt: Option<Vec<glam::Vec3>> = reader.read_normals().map(|it| {
        it.map(|n| glam::Vec3::new(n[0], n[1], n[2]))
            .collect::<Vec<_>>()
    });

    let normals: Vec<glam::Vec3> = if let Some(n) = normals_opt {
        assert!(
            n.len() == vertex_count,
            "NORMAL count mismatch: {} vs {}",
            n.len(),
            vertex_count
        );
        n
    } else {
        compute_smooth_normals(&positions, &indices)
    };

    let uvs: Vec<glam::Vec2> = if let Some(tc) = reader.read_tex_coords(0) {
        tc.into_f32()
            .map(|uv| {
                let u = uv[0];
                let mut v = uv[1];
                if options.flip_v {
                    v = 1.0 - v;
                }
                glam::Vec2::new(u, v)
            })
            .collect()
    } else {
        vec![glam::Vec2::ZERO; vertex_count]
    };

    assert!(
        uvs.len() == vertex_count,
        "TEXCOORD_0 count mismatch: {} vs {}",
        uvs.len(),
        vertex_count
    );

    // 打印一下原始uv的最大和最小值
    // println!(
    //     "Original UVs: u [{}, {}], v [{}, {}]",
    //     uvs.iter().map(|uv| uv.x).fold(f32::INFINITY, f32::min),
    //     uvs.iter().map(|uv| uv.x).fold(f32::NEG_INFINITY, f32::max),
    //     uvs.iter().map(|uv| uv.y).fold(f32::INFINITY, f32::min),
    //     uvs.iter().map(|uv| uv.y).fold(f32::NEG_INFINITY, f32::max),
    // );

    let colors: Vec<glam::Vec4> = if let Some(c) = reader.read_colors(0) {
        c.into_rgba_f32()
            .map(|c| glam::Vec4::new(c[0], c[1], c[2], c[3]))
            .collect()
    } else {
        vec![glam::Vec4::ONE; vertex_count]
    };
    assert!(
        colors.len() == vertex_count,
        "COLOR_0 count mismatch: {} vs {}",
        colors.len(),
        vertex_count
    );

    // Bake world transform if requested.
    let (pos_mat, nrm_mat3) = if options.bake_node_transform {
        let pos_mat = world;
        let inv_t = world.inverse().transpose();
        let nrm_mat3 = glam::Mat3::from_mat4(inv_t);
        (pos_mat, nrm_mat3)
    } else {
        (glam::Mat4::IDENTITY, glam::Mat3::IDENTITY)
    };

    let mut vertices: Vec<Vertex> = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let p = positions[i];
        let mut n = normals[i];
        let uv = uvs[i];
        let c = colors[i];

        let p4 = pos_mat * glam::Vec4::new(p.x, p.y, p.z, 1.0);
        if options.bake_node_transform {
            n = (nrm_mat3 * n).normalize_or_zero();
        }

        vertices.push(Vertex {
            position: glam::Vec4::new(p4.x, p4.y, p4.z, 1.0),
            normal: glam::Vec4::new(n.x, n.y, n.z, 0.0),
            color: c,
            uv,
        });
    }

    Mesh { vertices, indices }
}

fn compute_smooth_normals(positions: &[glam::Vec3], indices: &[u16]) -> Vec<glam::Vec3> {
    let mut normals = vec![glam::Vec3::ZERO; positions.len()];

    for tri in indices.as_chunks::<3>().0 {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2);
        if n.length_squared() > 0.0 {
            normals[i0] += n;
            normals[i1] += n;
            normals[i2] += n;
        }
    }

    for n in &mut normals {
        *n = n.normalize_or_zero();
    }

    normals
}

fn mat4_from_gltf(m: [[f32; 4]; 4]) -> glam::Mat4 {
    // glTF matrices are column-major; glam expects column-major in from_cols_array_2d.
    glam::Mat4::from_cols_array_2d(&m)
}

fn convert_gltf_image_to_rgba8(img: &gltf::image::Data) -> (Vec<u8>, u32, u32) {
    let width = img.width;
    let height = img.height;
    let pixels = &img.pixels;

    let rgba: Vec<u8> = match img.format {
        gltf::image::Format::R8G8B8A8 => pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for p in pixels.as_chunks::<3>().0 {
                out.push(p[0]);
                out.push(p[1]);
                out.push(p[2]);
                out.push(255);
            }
            out
        }
        gltf::image::Format::R8G8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for p in pixels.as_chunks::<2>().0 {
                let r = p[0];
                let g = p[1];
                out.push(r);
                out.push(g);
                out.push(0);
                out.push(255);
            }
            out
        }
        gltf::image::Format::R8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for &r in pixels.iter() {
                out.push(r);
                out.push(r);
                out.push(r);
                out.push(255);
            }
            out
        }
        other => {
            panic!(
                "Unsupported glTF image format {:?}; only 8-bit formats are supported",
                other
            );
        }
    };

    assert!(
        rgba.len() == (width * height * 4) as usize,
        "RGBA8 conversion size mismatch"
    );

    (rgba, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_document(json: &str) -> gltf::Document {
        gltf::Gltf::from_slice(json.as_bytes()).unwrap().document
    }

    #[test]
    fn sampler_wrap_and_all_min_filters_map_exactly() {
        let document = parse_document(
            r#"{
                "asset":{"version":"2.0"},
                "samplers":[
                    {"wrapS":33071,"wrapT":10497,"magFilter":9728,"minFilter":9728},
                    {"wrapS":33648,"wrapT":33071,"magFilter":9729,"minFilter":9729},
                    {"minFilter":9984},{"minFilter":9985},{"minFilter":9986},{"minFilter":9987}
                ]
            }"#,
        );
        let actual = document
            .samplers()
            .map(texture_sampler_from_gltf)
            .collect::<Vec<_>>();
        assert_eq!(actual[0].address_mode_u, TextureAddressMode::ClampToEdge);
        assert_eq!(actual[0].address_mode_v, TextureAddressMode::Repeat);
        assert_eq!(actual[0].mag_filter, TextureMagFilter::Nearest);
        assert_eq!(actual[1].address_mode_u, TextureAddressMode::MirroredRepeat);
        assert_eq!(actual[1].address_mode_v, TextureAddressMode::ClampToEdge);
        assert_eq!(actual[1].mag_filter, TextureMagFilter::Linear);
        assert_eq!(
            actual
                .iter()
                .map(|sampler| sampler.min_filter)
                .collect::<Vec<_>>(),
            [
                TextureMinFilter::Nearest,
                TextureMinFilter::Linear,
                TextureMinFilter::NearestMipmapNearest,
                TextureMinFilter::LinearMipmapNearest,
                TextureMinFilter::NearestMipmapLinear,
                TextureMinFilter::LinearMipmapLinear,
            ]
        );
    }

    #[test]
    fn omitted_and_empty_samplers_use_gltf_defaults() {
        let document = parse_document(
            r#"{
                "asset":{"version":"2.0"},
                "images":[{"uri":"unused.png"}],
                "samplers":[{}],
                "textures":[{"source":0},{"source":0,"sampler":0}]
            }"#,
        );
        let bindings = build_texture_bindings(&document, &[9]);
        assert_eq!(
            bindings[0],
            MaterialTextureBinding {
                texture_id: 9,
                sampler_id: 0
            }
        );
        assert_eq!(
            bindings[1],
            MaterialTextureBinding {
                texture_id: 9,
                sampler_id: 1
            }
        );
        assert_eq!(
            texture_sampler_from_gltf(document.samplers().next().unwrap()),
            TextureSampler::default()
        );
    }

    #[test]
    fn one_image_can_be_reused_with_distinct_samplers() {
        let document = parse_document(
            r#"{
                "asset":{"version":"2.0"},
                "images":[{"uri":"unused.png"}],
                "samplers":[{"wrapS":10497},{"wrapS":33071}],
                "textures":[{"source":0,"sampler":0},{"source":0,"sampler":1}]
            }"#,
        );
        let bindings = build_texture_bindings(&document, &[3]);
        assert_eq!(bindings[0].texture_id, bindings[1].texture_id);
        assert_ne!(bindings[0].sampler_id, bindings[1].sampler_id);
    }

    #[test]
    fn nonzero_texcoord_and_texture_transform_are_rejected() {
        let document = parse_document(
            r#"{
                "asset":{"version":"2.0"},
                "images":[{"uri":"unused.png"}],
                "textures":[{"source":0}],
                "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0,"texCoord":1}}}]
            }"#,
        );
        let info = document
            .materials()
            .next()
            .unwrap()
            .pbr_metallic_roughness()
            .base_color_texture();
        assert!(matches!(
            texture_binding(
                info,
                &[MaterialTextureBinding::default()],
                0,
                "baseColorTexture"
            ),
            Err(GltfLoadError::UnsupportedTexCoord { tex_coord: 1, .. })
        ));

        let document = parse_document(
            r#"{
                "asset":{"version":"2.0"},
                "extensionsUsed":["KHR_texture_transform"]
            }"#,
        );
        assert!(matches!(
            validate_supported_extensions(&document),
            Err(GltfLoadError::UnsupportedTextureTransform)
        ));
    }

    #[test]
    fn damaged_helmet_keeps_repeat_sampling_for_its_out_of_range_uvs() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("DamagedHelmet")
            .join("glTF")
            .join("DamagedHelmet.gltf");
        let model = load_gltf(
            path,
            LoadGltfOptions {
                global_scale: 1.0,
                flip_v: false,
                bake_node_transform: false,
            },
        )
        .unwrap();
        let albedo = model.materials[0].albedo;
        let sampler = model.samplers[albedo.sampler_id as usize];
        assert_eq!(sampler.address_mode_u, TextureAddressMode::Repeat);
        assert_eq!(sampler.address_mode_v, TextureAddressMode::Repeat);
        assert!(
            model
                .meshes
                .iter()
                .flat_map(|mesh| &mesh.vertices)
                .any(|vertex| vertex.uv.y > 1.0),
            "fixture must retain the UV range that exposed clamp-induced atlas streaking"
        );
    }
}

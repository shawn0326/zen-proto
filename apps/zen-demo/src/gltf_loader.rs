use std::path::Path;

use zen_renderer::mesh::{Instance, Material, Mesh, Texture, Vertex};

pub struct LoadedGltfModel {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub instances: Vec<Instance>,
    pub textures: Vec<Texture>,
}

#[derive(Clone, Copy)]
pub struct LoadGltfOptions {
    pub global_scale: f32,
    pub flip_v: bool,
    /// Bake node transforms into vertex positions/normals and emit identity instances.
    pub bake_node_transform: bool,
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

pub fn load_gltf(path: impl AsRef<Path>, options: LoadGltfOptions) -> LoadedGltfModel {
    let path = path.as_ref();
    let (document, buffers, images) = gltf::import(path)
        .unwrap_or_else(|e| panic!("Failed to import glTF {}: {e}", path.display()));

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

    // Materials: baseColorFactor * baseColorTexture (if present).
    let mut materials: Vec<Material> = Vec::with_capacity(document.materials().len() + 1);
    for m in document.materials() {
        let pbr = m.pbr_metallic_roughness();
        let factor = pbr.base_color_factor();
        let mut texture_id = 0u32;
        if let Some(tex) = pbr.base_color_texture() {
            let image_index = tex.texture().source().index();
            texture_id = texture_id_for_image.get(image_index).copied().unwrap_or(0);
        }

        let emissive_factor = m.emissive_factor();
        let emissive_rgb =
            glam::Vec3::new(emissive_factor[0], emissive_factor[1], emissive_factor[2]);

        // If emissiveTexture is missing, fall back to white so emissive_factor can still work.
        let mut emissive_texture_id = 0u32;
        if let Some(info) = m.emissive_texture() {
            let image_index = info.texture().source().index();
            emissive_texture_id = texture_id_for_image.get(image_index).copied().unwrap_or(0);
        }

        // AO (occlusion) support
        let mut ao_texture_id = 0u32;
        let mut ao_strength = 1.0f32;
        if let Some(occ) = m.occlusion_texture() {
            let image_index = occ.texture().source().index();
            ao_texture_id = texture_id_for_image.get(image_index).copied().unwrap_or(0);
            ao_strength = occ.strength();
        }

        let emissive_ao =
            glam::Vec4::new(emissive_rgb.x, emissive_rgb.y, emissive_rgb.z, ao_strength);

        materials.push(Material {
            albedo_factor: glam::Vec4::new(factor[0], factor[1], factor[2], factor[3]),
            emissive_ao,
            tex_ids: [texture_id, emissive_texture_id, ao_texture_id, 0],
        });
    }

    // Ensure we have a valid fallback material.
    let default_material_id = if materials.is_empty() {
        materials.push(Material {
            albedo_factor: glam::Vec4::ONE,
            emissive_ao: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            tex_ids: [0, 0, 0, 0],
        });
        0u32
    } else {
        // Append a dedicated default (white) so primitives with no material index are stable.
        let id = materials.len() as u32;
        materials.push(Material {
            albedo_factor: glam::Vec4::ONE,
            emissive_ao: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            tex_ids: [0, 0, 0, 0],
        });
        id
    };

    let mut meshes: Vec<Mesh> = Vec::new();
    let mut instances: Vec<Instance> = Vec::new();

    let scale_m = glam::Mat4::from_scale(glam::Vec3::splat(options.global_scale));
    for node in scene.nodes() {
        load_node_recursive(
            &node,
            &scale_m,
            options,
            &buffers,
            default_material_id,
            &mut meshes,
            &mut instances,
        );
    }

    LoadedGltfModel {
        meshes,
        materials,
        instances,
        textures,
    }
}

fn load_node_recursive(
    node: &gltf::Node,
    parent_world: &glam::Mat4,
    options: LoadGltfOptions,
    buffers: &[gltf::buffer::Data],
    default_material_id: u32,
    meshes_out: &mut Vec<Mesh>,
    instances_out: &mut Vec<Instance>,
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

            let engine_mesh = build_engine_mesh_from_primitive(&primitive, buffers, world, options);
            let mesh_id = meshes_out.len() as u32;
            meshes_out.push(engine_mesh);

            // Emit identity instance if we baked, otherwise keep node transform.
            let transform = if options.bake_node_transform {
                glam::Mat4::IDENTITY
            } else {
                world
            };
            instances_out.push(Instance {
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
            meshes_out,
            instances_out,
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

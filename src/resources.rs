use crate::material::{Material, MaterialStorage};
use crate::mesh::{Mesh, MeshStorage};
use crate::primitive::{Primitive, PrimitiveStorage};
use crate::texture::{Texture, TextureStorage};

pub struct Resources {
    pub meshes: MeshStorage,
    pub materials: MaterialStorage,
    pub primitives: PrimitiveStorage,
    pub textures: TextureStorage,
}

impl Resources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[Mesh],
        materials: &[Material],
        primitives: &[Primitive],
        textures: &[Texture],
    ) -> Self {
        let meshes = MeshStorage::from_meshes(device, meshes);
        let materials = MaterialStorage::from_materials(device, materials);
        let primitives = PrimitiveStorage::from_primitives(device, primitives);
        let textures = TextureStorage::from_textures(device, queue, textures);

        Self {
            meshes,
            materials,
            primitives,
            textures,
        }
    }
}

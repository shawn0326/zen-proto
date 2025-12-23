use crate::material::{Material, MaterialStorage};
use crate::mesh::{Mesh, MeshStorage};
use crate::primitive::{Primitive, PrimitiveStorage};

pub struct Resources {
    pub meshes: MeshStorage,
    pub materials: MaterialStorage,
    pub primitives: PrimitiveStorage,
}

impl Resources {
    pub fn new(
        device: &wgpu::Device,
        meshes: &[Mesh],
        materials: &[Material],
        primitives: &[Primitive],
    ) -> Self {
        let meshes = MeshStorage::from_meshes(device, meshes);
        let materials = MaterialStorage::from_materials(device, materials);
        let primitives = PrimitiveStorage::from_primitives(device, primitives);

        Self {
            meshes,
            materials,
            primitives,
        }
    }
}

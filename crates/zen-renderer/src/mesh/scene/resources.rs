use super::{
    Instance, InstanceStorage, Material, MaterialStorage, Mesh, MeshStorage, Texture,
    TextureStorage,
};

pub(crate) struct MeshSceneResources {
    meshes: MeshStorage,
    materials: MaterialStorage,
    instances: InstanceStorage,
    textures: TextureStorage,
}

impl MeshSceneResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[Mesh],
        materials: &[Material],
        instances: &[Instance],
        textures: &[Texture],
    ) -> Self {
        let meshes = MeshStorage::from_meshes(device, meshes);
        let materials = MaterialStorage::from_materials(device, materials);
        let instances = InstanceStorage::from_instances(device, instances);
        let textures = TextureStorage::from_textures(device, queue, textures);

        Self {
            meshes,
            materials,
            instances,
            textures,
        }
    }

    pub(crate) fn meshes(&self) -> &MeshStorage {
        &self.meshes
    }

    pub(crate) fn materials(&self) -> &MaterialStorage {
        &self.materials
    }

    pub(crate) fn instances(&self) -> &InstanceStorage {
        &self.instances
    }

    pub(crate) fn textures(&self) -> &TextureStorage {
        &self.textures
    }
}

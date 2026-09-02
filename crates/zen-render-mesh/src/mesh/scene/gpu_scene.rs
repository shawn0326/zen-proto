use super::{
    Instance, InstanceStorage, Material, MaterialStorage, Mesh, MeshStorage, Texture,
    TextureResourceError, TextureSampler, TextureSamplingConfig, TextureStorage,
};

pub(crate) struct MeshGpuScene {
    meshes: MeshStorage,
    materials: MaterialStorage,
    instances: InstanceStorage,
    textures: TextureStorage,
}

impl MeshGpuScene {
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor keeps the legacy scene resource inputs explicit"
    )]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[Mesh],
        materials: &[Material],
        instances: &[Instance],
        textures: &[Texture],
        samplers: &[TextureSampler],
        sampling: TextureSamplingConfig,
    ) -> Result<Self, TextureResourceError> {
        let meshes = MeshStorage::from_meshes(device, meshes);
        let materials = MaterialStorage::from_materials(device, materials);
        let instances = InstanceStorage::from_instances(device, instances);
        let textures = TextureStorage::from_resources(device, queue, textures, samplers, sampling)?;

        Ok(Self {
            meshes,
            materials,
            instances,
            textures,
        })
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

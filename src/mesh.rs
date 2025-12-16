#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: glam::Vec4,
    pub color: glam::Vec4,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshTableEntry {
    pub index_count: u32,   // number of indices
    pub first_index: u32,   // offset in the global index buffer (in indices)
    pub base_vertex: i32,   // offset in the global vertex buffer (in vertices)
    pub _pad: u32,          // pad to 16 bytes for WGSL/storage friendliness
    pub sphere: glam::Vec4, // bounding sphere (xyz: center, w: radius)
}

pub struct MeshesContext {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub mesh_table_buffer: wgpu::Buffer,
}

impl MeshesContext {
    pub fn from_meshes(device: &wgpu::Device, meshes: &[Mesh]) -> Self {
        use wgpu::util::DeviceExt;

        let total_vertices: usize = meshes.iter().map(|m| m.vertices.len()).sum();
        let total_indices: usize = meshes.iter().map(|m| m.indices.len()).sum();

        let mut all_vertices: Vec<Vertex> = Vec::with_capacity(total_vertices);
        let mut all_indices: Vec<u16> = Vec::with_capacity(total_indices);
        let mut mesh_table: Vec<MeshTableEntry> = Vec::with_capacity(meshes.len());

        for mesh in meshes {
            let base_vertex = all_vertices.len() as i32;
            let first_index = all_indices.len() as u32;
            let index_count = mesh.indices.len() as u32;

            // 不改写 u16 索引值；用 base_vertex 来做顶点偏移（更安全，避免 u16 溢出）
            all_vertices.extend_from_slice(&mesh.vertices);
            all_indices.extend_from_slice(&mesh.indices);

            mesh_table.push(MeshTableEntry {
                index_count,
                first_index,
                base_vertex,
                _pad: 0,
                sphere: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            });
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshes.vertex_buffer"),
            contents: bytemuck::cast_slice(&all_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshes.index_buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mesh_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshes.mesh_table_buffer"),
            contents: bytemuck::cast_slice(&mesh_table),
            usage: wgpu::BufferUsages::STORAGE,
        });

        MeshesContext {
            vertex_buffer,
            index_buffer,
            mesh_table_buffer,
        }
    }
}

pub fn create_triangle_mesh() -> Mesh {
    let vertices: [Vertex; 3] = [
        Vertex {
            position: glam::Vec4::new(-0.5, -0.5, 0.0, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, -0.5, 0.0, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.0, 0.5, 0.0, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
    ];

    let indices: [u16; 3] = [0, 1, 2];

    Mesh {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
    }
}

pub fn create_box_mesh() -> Mesh {
    let vertices: [Vertex; 8] = [
        Vertex {
            position: glam::Vec4::new(-0.5, -0.5, -0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, -0.5, -0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, 0.5, -0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(-0.5, 0.5, -0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(-0.5, -0.5, 0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, -0.5, 0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, 0.5, 0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(-0.5, 0.5, 0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
    ];

    let indices: [u16; 36] = [
        0, 1, 2, 2, 3, 0, // back face
        4, 5, 6, 6, 7, 4, // front face
        0, 4, 7, 7, 3, 0, // left face
        1, 5, 6, 6, 2, 1, // right face
        3, 2, 6, 6, 7, 3, // top face
        0, 1, 5, 5, 4, 0, // bottom face
    ];

    Mesh {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
    }
}

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

pub fn create_triangle_mesh() -> Mesh {
    let vertices: [Vertex; 3] = [
        Vertex {
            position: glam::Vec4::new(-0.5, -0.5, 0.0, 1.0),
            color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, -0.5, 0.0, 1.0),
            color: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.0, 0.5, 0.0, 1.0),
            color: glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
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
            color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, -0.5, -0.5, 1.0),
            color: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, 0.5, -0.5, 1.0),
            color: glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(-0.5, 0.5, -0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 0.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(-0.5, -0.5, 0.5, 1.0),
            color: glam::Vec4::new(1.0, 0.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, -0.5, 0.5, 1.0),
            color: glam::Vec4::new(0.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, 0.5, 0.5, 1.0),
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(-0.5, 0.5, 0.5, 1.0),
            color: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
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

#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: glam::Vec4,
    pub normal: glam::Vec4,
    pub color: glam::Vec4,
    pub uv: glam::Vec2,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexPacked {
    pub px: f32,
    pub py: f32,
    pub pz: f32,
    pub n_oct: u32, // 法线oct编码（snorm16x2打包到u32）
    pub uv01: u32,  // UV：unorm16x2打包到u32
    pub c: u32,     // RGBA8颜色打包到u32
    pub _pad0: u32, // 显式补齐到32B
    pub _pad1: u32,
}

impl VertexPacked {
    #[inline]
    fn pack_rgba8(color: glam::Vec4) -> u32 {
        fn q(x: f32) -> u32 {
            let x = x.clamp(0.0, 1.0);
            (x * 255.0 + 0.5) as u32
        }

        let r = q(color.x);
        let g = q(color.y);
        let b = q(color.z);
        let a = q(color.w);
        (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | ((a & 0xFF) << 24)
    }

    #[inline]
    fn unpack_rgba8(c: u32) -> glam::Vec4 {
        let r = ((c >> 0) & 0xFF) as f32 / 255.0;
        let g = ((c >> 8) & 0xFF) as f32 / 255.0;
        let b = ((c >> 16) & 0xFF) as f32 / 255.0;
        let a = ((c >> 24) & 0xFF) as f32 / 255.0;
        glam::Vec4::new(r, g, b, a)
    }

    #[inline]
    fn pack_unorm16x2(uv: glam::Vec2) -> u32 {
        fn q(x: f32) -> u32 {
            let x = x.clamp(0.0, 1.0);
            (x * 65535.0 + 0.5) as u32
        }

        let u = q(uv.x);
        let v = q(uv.y);
        (u & 0xFFFF) | ((v & 0xFFFF) << 16)
    }

    #[inline]
    fn unpack_unorm16x2(p: u32) -> glam::Vec2 {
        let u = (p & 0xFFFF) as f32 / 65535.0;
        let v = ((p >> 16) & 0xFFFF) as f32 / 65535.0;
        glam::Vec2::new(u, v)
    }

    #[inline]
    fn pack_snorm16(x: f32) -> i16 {
        let x = x.clamp(-1.0, 1.0);
        let q = (x * 32767.0).round() as i32;
        q.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }

    #[inline]
    fn unpack_snorm16(x: i16) -> f32 {
        // For -32768, this yields slightly less than -1.0; clamp for safety.
        (x as f32 / 32767.0).clamp(-1.0, 1.0)
    }

    #[inline]
    fn oct_encode(n: glam::Vec3) -> glam::Vec2 {
        let mut n = if n.length_squared() > 0.0 {
            n.normalize()
        } else {
            glam::Vec3::Z
        };
        let inv_l1 = 1.0 / (n.x.abs() + n.y.abs() + n.z.abs());
        n *= inv_l1;

        let mut e = glam::Vec2::new(n.x, n.y);
        if n.z < 0.0 {
            let ex = (1.0 - e.y.abs()) * e.x.signum();
            let ey = (1.0 - e.x.abs()) * e.y.signum();
            e = glam::Vec2::new(ex, ey);
        }
        e
    }

    #[inline]
    fn oct_decode(e: glam::Vec2) -> glam::Vec3 {
        let mut n = glam::Vec3::new(e.x, e.y, 1.0 - e.x.abs() - e.y.abs());
        if n.z < 0.0 {
            let ox = (1.0 - n.y.abs()) * n.x.signum();
            let oy = (1.0 - n.x.abs()) * n.y.signum();
            n.x = ox;
            n.y = oy;
        }
        if n.length_squared() > 0.0 {
            n.normalize()
        } else {
            glam::Vec3::Z
        }
    }

    #[inline]
    fn pack_normal_oct_snorm16x2(normal_xyz: glam::Vec3) -> u32 {
        let e = Self::oct_encode(normal_xyz);
        let sx = Self::pack_snorm16(e.x) as u16 as u32;
        let sy = Self::pack_snorm16(e.y) as u16 as u32;
        sx | (sy << 16)
    }

    #[inline]
    fn unpack_normal_oct_snorm16x2(packed: u32) -> glam::Vec3 {
        let sx = (packed & 0xFFFF) as u16 as i16;
        let sy = ((packed >> 16) & 0xFFFF) as u16 as i16;
        let e = glam::Vec2::new(Self::unpack_snorm16(sx), Self::unpack_snorm16(sy));
        Self::oct_decode(e)
    }
}

impl From<&Vertex> for VertexPacked {
    fn from(v: &Vertex) -> Self {
        let normal_xyz = glam::Vec3::new(v.normal.x, v.normal.y, v.normal.z);
        Self {
            px: v.position.x,
            py: v.position.y,
            pz: v.position.z,
            n_oct: Self::pack_normal_oct_snorm16x2(normal_xyz),
            uv01: Self::pack_unorm16x2(v.uv),
            c: Self::pack_rgba8(v.color),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

impl From<&VertexPacked> for Vertex {
    fn from(v: &VertexPacked) -> Self {
        let n = VertexPacked::unpack_normal_oct_snorm16x2(v.n_oct);
        Vertex {
            position: glam::Vec4::new(v.px, v.py, v.pz, 1.0),
            normal: glam::Vec4::new(n.x, n.y, n.z, 0.0),
            color: VertexPacked::unpack_rgba8(v.c),
            uv: VertexPacked::unpack_unorm16x2(v.uv01),
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl Mesh {
    pub fn bounding_sphere(&self) -> glam::Vec4 {
        // 空网格：返回半径 0 的球
        if self.vertices.is_empty() {
            return glam::Vec4::new(0.0, 0.0, 0.0, 0.0);
        }

        // 1) 计算质心作为球心（简单且快速）
        let mut sum = glam::Vec3::new(0.0, 0.0, 0.0);
        for v in &self.vertices {
            sum += glam::Vec3::new(v.position.x, v.position.y, v.position.z);
        }
        let inv_n = 1.0 / (self.vertices.len() as f32);
        let center3 = sum * inv_n;

        // 2) 半径为到质心的最大距离
        let mut max_dsq = 0.0f32;
        for v in &self.vertices {
            let p = glam::Vec3::new(v.position.x, v.position.y, v.position.z);
            let dsq = (p - center3).length_squared();
            if dsq > max_dsq {
                max_dsq = dsq;
            }
        }
        let radius = max_dsq.sqrt();

        glam::Vec4::new(center3.x, center3.y, center3.z, radius)
    }
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

pub struct MeshStorage {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    mesh_table_buffer: wgpu::Buffer,
    mesh_count: u32,
}

impl MeshStorage {
    pub fn from_meshes(device: &wgpu::Device, meshes: &[Mesh]) -> Self {
        use wgpu::util::DeviceExt;

        let total_vertices: usize = meshes.iter().map(|m| m.vertices.len()).sum();
        let total_indices: usize = meshes.iter().map(|m| m.indices.len()).sum();

        let mut all_vertices: Vec<VertexPacked> = Vec::with_capacity(total_vertices);
        let mut all_indices: Vec<u16> = Vec::with_capacity(total_indices);
        let mut mesh_table: Vec<MeshTableEntry> = Vec::with_capacity(meshes.len());

        for mesh in meshes {
            let base_vertex = all_vertices.len() as i32;
            let first_index = all_indices.len() as u32;
            let index_count = mesh.indices.len() as u32;

            // 不改写 u16 索引值；用 base_vertex 来做顶点偏移（更安全，避免 u16 溢出）
            all_vertices.extend(mesh.vertices.iter().map(VertexPacked::from));
            all_indices.extend_from_slice(&mesh.indices);

            mesh_table.push(MeshTableEntry {
                index_count,
                first_index,
                base_vertex,
                _pad: 0,
                sphere: mesh.bounding_sphere(),
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

        Self {
            vertex_buffer,
            index_buffer,
            mesh_table_buffer,
            mesh_count: meshes.len() as u32,
        }
    }

    pub(crate) fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    pub(crate) fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    pub(crate) fn mesh_table_buffer(&self) -> &wgpu::Buffer {
        &self.mesh_table_buffer
    }

    pub(crate) fn mesh_count(&self) -> u32 {
        self.mesh_count
    }
}

impl Mesh {
    pub fn create_triangle() -> Self {
        // 三角形面朝 +Z，法线为 (0,0,1,0)
        let normal = glam::Vec4::new(0.0, 0.0, 1.0, 0.0);
        let vertices: [Vertex; 3] = [
            Vertex {
                position: glam::Vec4::new(-0.5, -0.5, 0.0, 1.0),
                normal,
                color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
                uv: glam::Vec2::new(0.0, 0.0),
            },
            Vertex {
                position: glam::Vec4::new(0.5, -0.5, 0.0, 1.0),
                normal,
                color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
                uv: glam::Vec2::new(1.0, 0.0),
            },
            Vertex {
                position: glam::Vec4::new(0.0, 0.5, 0.0, 1.0),
                normal,
                color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
                uv: glam::Vec2::new(0.5, 1.0),
            },
        ];

        let indices: [u16; 3] = [0, 1, 2];

        Self {
            vertices: vertices.to_vec(),
            indices: indices.to_vec(),
        }
    }

    pub fn create_box() -> Self {
        let white = glam::Vec4::new(1.0, 1.0, 1.0, 1.0);

        // 每个面的数据：(法线, 4个顶点)
        let faces = [
            // back (-Z)
            (
                glam::Vec4::new(0.0, 0.0, -1.0, 0.0),
                [
                    [-0.5, -0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, -0.5, -0.5],
                ],
            ),
            // front (+Z)
            (
                glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
                [
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ],
            ),
            // left (-X)
            (
                glam::Vec4::new(-1.0, 0.0, 0.0, 0.0),
                [
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                ],
            ),
            // right (+X)
            (
                glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
                [
                    [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, -0.5, 0.5],
                ],
            ),
            // top (+Y)
            (
                glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
                [
                    [-0.5, 0.5, -0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, -0.5],
                ],
            ),
            // bottom (-Y)
            (
                glam::Vec4::new(0.0, -1.0, 0.0, 0.0),
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, -0.5, 0.5],
                ],
            ),
        ];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for (face_idx, (normal, positions)) in faces.iter().enumerate() {
            let base = (face_idx * 4) as u16;
            for (i, pos) in positions.iter().enumerate() {
                let uv = match i {
                    0 => glam::Vec2::new(0.0, 0.0),
                    1 => glam::Vec2::new(0.0, 1.0),
                    2 => glam::Vec2::new(1.0, 1.0),
                    _ => glam::Vec2::new(1.0, 0.0),
                };

                vertices.push(Vertex {
                    position: glam::Vec4::new(pos[0], pos[1], pos[2], 1.0),
                    normal: *normal,
                    color: white,
                    uv,
                });
            }
            // 两个三角形
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        Self { vertices, indices }
    }

    pub fn create_sphere(subdivisions: u32) -> Self {
        // UV sphere: stacks(纬向) 与 slices(经向)
        // subdivisions 太小会退化，给个下限
        let stacks = subdivisions.max(3) as usize;
        let slices = (subdivisions.max(3) * 2) as usize;

        let radius: f32 = 0.5;
        let white = glam::Vec4::new(1.0, 1.0, 1.0, 1.0);

        // 顶点： (stacks+1) * (slices+1) 以处理经度缝合
        let vert_count = (stacks + 1) * (slices + 1);
        assert!(
            vert_count <= u16::MAX as usize,
            "create_sphere_mesh: too many vertices ({vert_count}), increase index type or reduce subdivisions"
        );

        let mut vertices = Vec::with_capacity(vert_count);
        let mut indices = Vec::with_capacity(stacks * slices * 6);

        let pi = std::f32::consts::PI;
        let two_pi = 2.0 * pi;

        for i in 0..=stacks {
            let v = i as f32 / stacks as f32; // 0..1
            let theta = v * pi; // 0..PI (north->south)

            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for j in 0..=slices {
                let u = j as f32 / slices as f32; // 0..1
                let phi = u * two_pi; // 0..2PI

                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                // 右手系：x 右，y 上，z 前（你项目里一般这样用）
                let x = sin_theta * cos_phi;
                let y = cos_theta;
                let z = sin_theta * sin_phi;

                let normal3 = glam::Vec3::new(x, y, z).normalize();
                let normal = glam::Vec4::new(normal3.x, normal3.y, normal3.z, 0.0);

                vertices.push(Vertex {
                    position: glam::Vec4::new(radius * x, radius * y, radius * z, 1.0),
                    normal,
                    color: white,
                    uv: glam::Vec2::new(u, 1.0 - v),
                });
            }
        }

        // 索引：每个 quad -> 2 triangles
        let stride = (slices + 1) as u16;
        for i in 0..stacks {
            for j in 0..slices {
                let a = (i as u16) * stride + (j as u16);
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;

                // 外侧面：a-c-b 与 b-c-d（CCW）
                indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }

        Self { vertices, indices }
    }
}

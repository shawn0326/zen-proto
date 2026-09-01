use std::collections::{BTreeMap, BTreeSet};

use super::{
    BoundsSphere, LodTableEntry, MeshTableEntry, MeshletAssetError, MeshletAssetHash,
    MeshletBuildConfig, MeshletSceneAsset, MeshletTableEntry, NormalCone, PackedVertexAttributes,
    ZENMESH_VERSION,
};

const MAGIC: [u8; 8] = *b"ZENMESH\0";
const ENDIAN_TAG: u32 = 0x0102_0304;
const HEADER_SIZE: usize = 80;
const DIRECTORY_ENTRY_SIZE: usize = 48;
const ALIGNMENT: usize = 16;
const SECTION_COUNT: usize = 9;

const CONFIG_STRIDE: usize = 32;
const MESH_STRIDE: usize = 48;
const LOD_STRIDE: usize = 48;
const MESHLET_STRIDE: usize = 64;
const POSITION_STRIDE: usize = 12;
const ATTRIBUTE_STRIDE: usize = 16;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SectionKind {
    BuildConfig = 1,
    Meshes = 2,
    Lods = 3,
    Meshlets = 4,
    Positions = 5,
    Attributes = 6,
    MeshletVertexRefs = 7,
    MicroIndices = 8,
    FallbackIndices = 9,
}

impl TryFrom<u32> for SectionKind {
    type Error = MeshletAssetError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::BuildConfig),
            2 => Ok(Self::Meshes),
            3 => Ok(Self::Lods),
            4 => Ok(Self::Meshlets),
            5 => Ok(Self::Positions),
            6 => Ok(Self::Attributes),
            7 => Ok(Self::MeshletVertexRefs),
            8 => Ok(Self::MicroIndices),
            9 => Ok(Self::FallbackIndices),
            _ => Err(MeshletAssetError::InvalidFormat(format!(
                "unknown section kind {value}"
            ))),
        }
    }
}

struct EncodedSection {
    kind: SectionKind,
    count: u64,
    stride: u32,
    bytes: Vec<u8>,
    offset: u64,
}

#[derive(Clone, Copy)]
struct DirectoryEntry {
    kind: SectionKind,
    offset: usize,
    byte_len: usize,
    count: usize,
    stride: usize,
    checksum: u64,
}

pub(super) fn encode(asset: &MeshletSceneAsset) -> Result<Vec<u8>, MeshletAssetError> {
    asset.validate()?;
    let mut sections = vec![
        section(
            SectionKind::BuildConfig,
            1,
            CONFIG_STRIDE,
            encode_config(asset.config),
        )?,
        section(
            SectionKind::Meshes,
            asset.meshes.len(),
            MESH_STRIDE,
            encode_meshes(&asset.meshes),
        )?,
        section(
            SectionKind::Lods,
            asset.lods.len(),
            LOD_STRIDE,
            encode_lods(&asset.lods),
        )?,
        section(
            SectionKind::Meshlets,
            asset.meshlets.len(),
            MESHLET_STRIDE,
            encode_meshlets(&asset.meshlets),
        )?,
        section(
            SectionKind::Positions,
            asset.positions.len(),
            POSITION_STRIDE,
            encode_positions(&asset.positions),
        )?,
        section(
            SectionKind::Attributes,
            asset.attributes.len(),
            ATTRIBUTE_STRIDE,
            encode_attributes(&asset.attributes),
        )?,
        section(
            SectionKind::MeshletVertexRefs,
            asset.meshlet_vertex_refs.len(),
            4,
            encode_u32s(&asset.meshlet_vertex_refs),
        )?,
        section(
            SectionKind::MicroIndices,
            asset.micro_indices.len(),
            1,
            asset.micro_indices.clone(),
        )?,
        section(
            SectionKind::FallbackIndices,
            asset.fallback_indices.len(),
            4,
            encode_u32s(&asset.fallback_indices),
        )?,
    ];
    debug_assert_eq!(sections.len(), SECTION_COUNT);

    let directory_offset = HEADER_SIZE;
    let mut cursor = align_up(
        HEADER_SIZE
            .checked_add(SECTION_COUNT * DIRECTORY_ENTRY_SIZE)
            .ok_or(MeshletAssetError::SizeOverflow("section directory"))?,
        ALIGNMENT,
    )?;
    for section in &mut sections {
        section.offset =
            u64::try_from(cursor).map_err(|_| MeshletAssetError::SizeOverflow("section offset"))?;
        cursor = align_up(
            cursor
                .checked_add(section.bytes.len())
                .ok_or(MeshletAssetError::SizeOverflow("encoded file"))?,
            ALIGNMENT,
        )?;
    }
    let file_size = cursor;
    let mut output = Vec::with_capacity(file_size);
    output.extend_from_slice(&MAGIC);
    push_u32(&mut output, ZENMESH_VERSION);
    push_u32(&mut output, ENDIAN_TAG);
    push_u32(&mut output, HEADER_SIZE as u32);
    push_u32(&mut output, SECTION_COUNT as u32);
    output.extend_from_slice(&asset.source_hash.to_bytes());
    output.extend_from_slice(&asset.build_hash.to_bytes());
    push_u64(&mut output, directory_offset as u64);
    push_u64(&mut output, file_size as u64);
    let header_checksum = checksum64(&output);
    push_u64(&mut output, header_checksum);
    debug_assert_eq!(output.len(), HEADER_SIZE);

    for section in &sections {
        push_u32(&mut output, section.kind as u32);
        push_u32(&mut output, 0);
        push_u64(&mut output, section.offset);
        push_u64(&mut output, section.bytes.len() as u64);
        push_u64(&mut output, section.count);
        push_u32(&mut output, section.stride);
        push_u32(&mut output, 0);
        push_u64(&mut output, checksum64(&section.bytes));
    }
    pad_to(&mut output, ALIGNMENT);
    for section in sections {
        if output.len() != section.offset as usize {
            return Err(MeshletAssetError::InvalidAsset(
                "internal section offset calculation mismatch".into(),
            ));
        }
        output.extend_from_slice(&section.bytes);
        pad_to(&mut output, ALIGNMENT);
    }
    debug_assert_eq!(output.len(), file_size);
    Ok(output)
}

fn section(
    kind: SectionKind,
    count: usize,
    stride: usize,
    bytes: Vec<u8>,
) -> Result<EncodedSection, MeshletAssetError> {
    let expected = count
        .checked_mul(stride)
        .ok_or(MeshletAssetError::SizeOverflow("section byte length"))?;
    if bytes.len() != expected {
        return Err(MeshletAssetError::InvalidAsset(format!(
            "internal {:?} encoder produced {} bytes, expected {expected}",
            kind,
            bytes.len()
        )));
    }
    Ok(EncodedSection {
        kind,
        count: count as u64,
        stride: stride as u32,
        bytes,
        offset: 0,
    })
}

pub(super) fn decode(bytes: &[u8]) -> Result<MeshletSceneAsset, MeshletAssetError> {
    if bytes.len() < HEADER_SIZE {
        return Err(MeshletAssetError::InvalidFormat("truncated header".into()));
    }
    let mut header = Reader::new(&bytes[..HEADER_SIZE]);
    if header.read_exact(8)? != MAGIC {
        return Err(MeshletAssetError::InvalidFormat("bad magic".into()));
    }
    let version = header.read_u32()?;
    if version != ZENMESH_VERSION {
        return Err(MeshletAssetError::UnsupportedVersion(version));
    }
    if header.read_u32()? != ENDIAN_TAG {
        return Err(MeshletAssetError::InvalidFormat(
            "unsupported byte order marker".into(),
        ));
    }
    if header.read_u32()? as usize != HEADER_SIZE {
        return Err(MeshletAssetError::InvalidFormat(
            "unexpected v1 header size".into(),
        ));
    }
    let section_count = header.read_u32()? as usize;
    if section_count != SECTION_COUNT {
        return Err(MeshletAssetError::InvalidFormat(format!(
            "expected {SECTION_COUNT} sections, found {section_count}"
        )));
    }
    let source_hash = MeshletAssetHash::from_bytes(header.read_array::<16>()?);
    let build_hash = MeshletAssetHash::from_bytes(header.read_array::<16>()?);
    let directory_offset = usize_from_u64(header.read_u64()?, "directory offset")?;
    let file_size = usize_from_u64(header.read_u64()?, "file size")?;
    let header_checksum = header.read_u64()?;
    if header_checksum != checksum64(&bytes[..HEADER_SIZE - 8]) {
        return Err(MeshletAssetError::InvalidFormat(
            "header checksum mismatch".into(),
        ));
    }
    if file_size != bytes.len() || file_size % ALIGNMENT != 0 {
        return Err(MeshletAssetError::InvalidFormat(format!(
            "declared file size {file_size} does not match aligned input size {}",
            bytes.len()
        )));
    }
    if directory_offset != HEADER_SIZE {
        return Err(MeshletAssetError::InvalidFormat(
            "unexpected v1 directory offset".into(),
        ));
    }
    let directory_size = section_count
        .checked_mul(DIRECTORY_ENTRY_SIZE)
        .ok_or(MeshletAssetError::SizeOverflow("section directory"))?;
    let directory_end = directory_offset
        .checked_add(directory_size)
        .ok_or(MeshletAssetError::SizeOverflow("section directory"))?;
    if directory_end > bytes.len() {
        return Err(MeshletAssetError::InvalidFormat(
            "truncated section directory".into(),
        ));
    }

    let mut directory_reader = Reader::new(&bytes[directory_offset..directory_end]);
    let mut directory = BTreeMap::new();
    let mut occupied = BTreeSet::new();
    let minimum_section_offset = align_up(directory_end, ALIGNMENT)?;
    for _ in 0..section_count {
        let kind = SectionKind::try_from(directory_reader.read_u32()?)?;
        if directory_reader.read_u32()? != 0 {
            return Err(MeshletAssetError::InvalidFormat(format!(
                "section {kind:?} has unsupported flags"
            )));
        }
        let offset = usize_from_u64(directory_reader.read_u64()?, "section offset")?;
        let byte_len = usize_from_u64(directory_reader.read_u64()?, "section byte length")?;
        let count = usize_from_u64(directory_reader.read_u64()?, "section element count")?;
        let stride = directory_reader.read_u32()? as usize;
        if directory_reader.read_u32()? != 0 {
            return Err(MeshletAssetError::InvalidFormat(format!(
                "section {kind:?} has a non-zero reserved field"
            )));
        }
        let checksum = directory_reader.read_u64()?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(MeshletAssetError::SizeOverflow("section range"))?;
        if offset < minimum_section_offset || offset % ALIGNMENT != 0 || end > bytes.len() {
            return Err(MeshletAssetError::InvalidFormat(format!(
                "section {kind:?} has an out-of-range or unaligned byte range"
            )));
        }
        let expected_len = count
            .checked_mul(stride)
            .ok_or(MeshletAssetError::SizeOverflow("section element range"))?;
        if stride == 0 || byte_len != expected_len {
            return Err(MeshletAssetError::InvalidFormat(format!(
                "section {kind:?} byte length does not equal count * stride"
            )));
        }
        if checksum64(&bytes[offset..end]) != checksum {
            return Err(MeshletAssetError::InvalidFormat(format!(
                "section {kind:?} checksum mismatch"
            )));
        }
        // Include padding in overlap detection so another section cannot alias it.
        let aligned_end = align_up(end, ALIGNMENT)?;
        if occupied
            .iter()
            .any(|&(other_start, other_end)| offset < other_end && other_start < aligned_end)
        {
            return Err(MeshletAssetError::InvalidFormat(
                "section ranges overlap".into(),
            ));
        }
        occupied.insert((offset, aligned_end));
        if directory
            .insert(
                kind,
                DirectoryEntry {
                    kind,
                    offset,
                    byte_len,
                    count,
                    stride,
                    checksum,
                },
            )
            .is_some()
        {
            return Err(MeshletAssetError::InvalidFormat(format!(
                "duplicate section {kind:?}"
            )));
        }
    }
    if directory.len() != SECTION_COUNT {
        return Err(MeshletAssetError::InvalidFormat(
            "missing required section".into(),
        ));
    }

    // Canonical v1 files require zero-filled alignment gaps. Besides making byte-for-byte output
    // deterministic, this ensures corruption in padding cannot silently survive checksums.
    let mut data_ranges: Vec<(usize, usize)> = directory
        .values()
        .filter(|entry| entry.byte_len != 0)
        .map(|entry| (entry.offset, entry.offset + entry.byte_len))
        .collect();
    data_ranges.sort_unstable();
    let mut gap_start = directory_end;
    for (start, end) in data_ranges {
        if bytes[gap_start..start].iter().any(|&byte| byte != 0) {
            return Err(MeshletAssetError::InvalidFormat(
                "non-zero section alignment padding".into(),
            ));
        }
        gap_start = end;
    }
    if bytes[gap_start..].iter().any(|&byte| byte != 0) {
        return Err(MeshletAssetError::InvalidFormat(
            "non-zero trailing alignment padding".into(),
        ));
    }

    let config_entry = require(&directory, SectionKind::BuildConfig, CONFIG_STRIDE)?;
    if config_entry.count != 1 {
        return Err(MeshletAssetError::InvalidFormat(
            "build-config section must contain exactly one record".into(),
        ));
    }
    let config = decode_config(section_bytes(bytes, config_entry))?;
    let asset = MeshletSceneAsset {
        source_hash,
        build_hash,
        config,
        meshes: decode_meshes(section_of(
            bytes,
            &directory,
            SectionKind::Meshes,
            MESH_STRIDE,
        )?)?,
        lods: decode_lods(section_of(
            bytes,
            &directory,
            SectionKind::Lods,
            LOD_STRIDE,
        )?)?,
        meshlets: decode_meshlets(section_of(
            bytes,
            &directory,
            SectionKind::Meshlets,
            MESHLET_STRIDE,
        )?)?,
        positions: decode_positions(section_of(
            bytes,
            &directory,
            SectionKind::Positions,
            POSITION_STRIDE,
        )?)?,
        attributes: decode_attributes(section_of(
            bytes,
            &directory,
            SectionKind::Attributes,
            ATTRIBUTE_STRIDE,
        )?)?,
        meshlet_vertex_refs: decode_u32s(section_of(
            bytes,
            &directory,
            SectionKind::MeshletVertexRefs,
            4,
        )?)?,
        micro_indices: copy_section_bytes(section_of(
            bytes,
            &directory,
            SectionKind::MicroIndices,
            1,
        )?)?,
        fallback_indices: decode_u32s(section_of(
            bytes,
            &directory,
            SectionKind::FallbackIndices,
            4,
        )?)?,
    };
    asset.validate()?;
    Ok(asset)
}

fn require(
    directory: &BTreeMap<SectionKind, DirectoryEntry>,
    kind: SectionKind,
    expected_stride: usize,
) -> Result<DirectoryEntry, MeshletAssetError> {
    let entry = *directory
        .get(&kind)
        .ok_or_else(|| MeshletAssetError::InvalidFormat(format!("missing section {kind:?}")))?;
    if entry.kind != kind || entry.stride != expected_stride {
        return Err(MeshletAssetError::InvalidFormat(format!(
            "section {kind:?} has stride {}, expected {expected_stride}",
            entry.stride
        )));
    }
    let _ = entry.checksum;
    Ok(entry)
}

fn section_of<'a>(
    file: &'a [u8],
    directory: &BTreeMap<SectionKind, DirectoryEntry>,
    kind: SectionKind,
    stride: usize,
) -> Result<&'a [u8], MeshletAssetError> {
    Ok(section_bytes(file, require(directory, kind, stride)?))
}

fn section_bytes(file: &[u8], entry: DirectoryEntry) -> &[u8] {
    &file[entry.offset..entry.offset + entry.byte_len]
}

fn encode_config(config: MeshletBuildConfig) -> Vec<u8> {
    let mut output = Vec::with_capacity(CONFIG_STRIDE);
    push_u32(&mut output, config.max_meshlet_vertices);
    push_u32(&mut output, config.max_meshlet_triangles);
    push_u32(&mut output, config.task_workgroup_meshlets);
    push_u32(&mut output, config.max_lods);
    push_f32(&mut output, config.lod_target_ratio);
    push_u32(&mut output, config.min_lod_triangles);
    push_f32(&mut output, config.meshlet_cone_weight);
    push_u32(&mut output, 0);
    output
}

fn decode_config(bytes: &[u8]) -> Result<MeshletBuildConfig, MeshletAssetError> {
    let mut reader = Reader::new(bytes);
    let config = MeshletBuildConfig {
        max_meshlet_vertices: reader.read_u32()?,
        max_meshlet_triangles: reader.read_u32()?,
        task_workgroup_meshlets: reader.read_u32()?,
        max_lods: reader.read_u32()?,
        lod_target_ratio: reader.read_f32()?,
        min_lod_triangles: reader.read_u32()?,
        meshlet_cone_weight: reader.read_f32()?,
    };
    if reader.read_u32()? != 0 {
        return Err(MeshletAssetError::InvalidFormat(
            "non-zero build-config reserved field".into(),
        ));
    }
    config.validate()?;
    Ok(config)
}

fn encode_meshes(meshes: &[MeshTableEntry]) -> Vec<u8> {
    let mut output = Vec::with_capacity(meshes.len() * MESH_STRIDE);
    for mesh in meshes {
        push_u32(&mut output, mesh.first_lod);
        push_u32(&mut output, mesh.lod_count);
        push_u32(&mut output, mesh.first_vertex);
        push_u32(&mut output, mesh.vertex_count);
        push_u32(&mut output, mesh.material_slot);
        push_u32(&mut output, mesh.pso_class);
        push_u32(&mut output, 0);
        push_u32(&mut output, 0);
        push_sphere(&mut output, mesh.bounds);
    }
    output
}

fn decode_meshes(bytes: &[u8]) -> Result<Vec<MeshTableEntry>, MeshletAssetError> {
    decode_records(bytes, MESH_STRIDE, |reader| {
        let mesh = MeshTableEntry {
            first_lod: reader.read_u32()?,
            lod_count: reader.read_u32()?,
            first_vertex: reader.read_u32()?,
            vertex_count: reader.read_u32()?,
            material_slot: reader.read_u32()?,
            pso_class: reader.read_u32()?,
            bounds: {
                if reader.read_u32()? != 0 || reader.read_u32()? != 0 {
                    return Err(MeshletAssetError::InvalidFormat(
                        "non-zero mesh reserved field".into(),
                    ));
                }
                reader.read_sphere()?
            },
        };
        Ok(mesh)
    })
}

fn encode_lods(lods: &[LodTableEntry]) -> Vec<u8> {
    let mut output = Vec::with_capacity(lods.len() * LOD_STRIDE);
    for lod in lods {
        push_u32(&mut output, lod.first_meshlet);
        push_u32(&mut output, lod.meshlet_count);
        push_u32(&mut output, lod.first_index);
        push_u32(&mut output, lod.index_count);
        push_u32(&mut output, lod.first_vertex);
        push_u32(&mut output, lod.vertex_count);
        push_f32(&mut output, lod.geometric_error);
        push_u32(&mut output, 0);
        push_sphere(&mut output, lod.bounds);
    }
    output
}

fn decode_lods(bytes: &[u8]) -> Result<Vec<LodTableEntry>, MeshletAssetError> {
    decode_records(bytes, LOD_STRIDE, |reader| {
        let first_meshlet = reader.read_u32()?;
        let meshlet_count = reader.read_u32()?;
        let first_index = reader.read_u32()?;
        let index_count = reader.read_u32()?;
        let first_vertex = reader.read_u32()?;
        let vertex_count = reader.read_u32()?;
        let geometric_error = reader.read_f32()?;
        if reader.read_u32()? != 0 {
            return Err(MeshletAssetError::InvalidFormat(
                "non-zero LOD reserved field".into(),
            ));
        }
        Ok(LodTableEntry {
            first_meshlet,
            meshlet_count,
            first_index,
            index_count,
            first_vertex,
            vertex_count,
            geometric_error,
            bounds: reader.read_sphere()?,
        })
    })
}

fn encode_meshlets(meshlets: &[MeshletTableEntry]) -> Vec<u8> {
    let mut output = Vec::with_capacity(meshlets.len() * MESHLET_STRIDE);
    for meshlet in meshlets {
        push_u32(&mut output, meshlet.vertex_offset);
        push_u32(&mut output, meshlet.vertex_count);
        push_u32(&mut output, meshlet.triangle_offset);
        push_u32(&mut output, meshlet.triangle_count);
        push_u32(&mut output, meshlet.fallback_first_index);
        push_u32(&mut output, meshlet.fallback_index_count);
        push_u32(&mut output, 0);
        push_u32(&mut output, 0);
        push_sphere(&mut output, meshlet.bounds);
        push_cone(&mut output, meshlet.normal_cone);
    }
    output
}

fn decode_meshlets(bytes: &[u8]) -> Result<Vec<MeshletTableEntry>, MeshletAssetError> {
    decode_records(bytes, MESHLET_STRIDE, |reader| {
        let vertex_offset = reader.read_u32()?;
        let vertex_count = reader.read_u32()?;
        let triangle_offset = reader.read_u32()?;
        let triangle_count = reader.read_u32()?;
        let fallback_first_index = reader.read_u32()?;
        let fallback_index_count = reader.read_u32()?;
        if reader.read_u32()? != 0 || reader.read_u32()? != 0 {
            return Err(MeshletAssetError::InvalidFormat(
                "non-zero meshlet reserved field".into(),
            ));
        }
        Ok(MeshletTableEntry {
            vertex_offset,
            vertex_count,
            triangle_offset,
            triangle_count,
            fallback_first_index,
            fallback_index_count,
            bounds: reader.read_sphere()?,
            normal_cone: reader.read_cone()?,
        })
    })
}

fn encode_positions(positions: &[[f32; 3]]) -> Vec<u8> {
    let mut output = Vec::with_capacity(positions.len() * POSITION_STRIDE);
    for position in positions {
        for value in position {
            push_f32(&mut output, *value);
        }
    }
    output
}

fn decode_positions(bytes: &[u8]) -> Result<Vec<[f32; 3]>, MeshletAssetError> {
    decode_records(bytes, POSITION_STRIDE, |reader| {
        Ok([reader.read_f32()?, reader.read_f32()?, reader.read_f32()?])
    })
}

fn encode_attributes(attributes: &[PackedVertexAttributes]) -> Vec<u8> {
    let mut output = Vec::with_capacity(attributes.len() * ATTRIBUTE_STRIDE);
    for attributes in attributes {
        push_u32(&mut output, attributes.normal_oct);
        push_f32(&mut output, attributes.uv[0]);
        push_f32(&mut output, attributes.uv[1]);
        push_u32(&mut output, attributes.color_rgba8);
    }
    output
}

fn decode_attributes(bytes: &[u8]) -> Result<Vec<PackedVertexAttributes>, MeshletAssetError> {
    decode_records(bytes, ATTRIBUTE_STRIDE, |reader| {
        Ok(PackedVertexAttributes {
            normal_oct: reader.read_u32()?,
            uv: [reader.read_f32()?, reader.read_f32()?],
            color_rgba8: reader.read_u32()?,
        })
    })
}

fn encode_u32s(values: &[u32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(values.len() * 4);
    for value in values {
        push_u32(&mut output, *value);
    }
    output
}

fn decode_u32s(bytes: &[u8]) -> Result<Vec<u32>, MeshletAssetError> {
    decode_records(bytes, 4, |reader| reader.read_u32())
}

fn copy_section_bytes(bytes: &[u8]) -> Result<Vec<u8>, MeshletAssetError> {
    let mut output = Vec::new();
    output.try_reserve_exact(bytes.len()).map_err(|_| {
        MeshletAssetError::InvalidFormat(
            "decoded section exceeds the available allocation budget".into(),
        )
    })?;
    output.extend_from_slice(bytes);
    Ok(output)
}

fn decode_records<T>(
    bytes: &[u8],
    stride: usize,
    mut decode: impl FnMut(&mut Reader<'_>) -> Result<T, MeshletAssetError>,
) -> Result<Vec<T>, MeshletAssetError> {
    if !bytes.len().is_multiple_of(stride) {
        return Err(MeshletAssetError::InvalidFormat(
            "section length is not divisible by its stride".into(),
        ));
    }
    let record_count = bytes.len() / stride;
    let mut output = Vec::new();
    output.try_reserve_exact(record_count).map_err(|_| {
        MeshletAssetError::InvalidFormat(
            "decoded record section exceeds the available allocation budget".into(),
        )
    })?;
    for record in bytes.chunks_exact(stride) {
        let mut reader = Reader::new(record);
        output.push(decode(&mut reader)?);
        if reader.remaining() != 0 {
            return Err(MeshletAssetError::InvalidFormat(
                "record decoder did not consume the declared stride".into(),
            ));
        }
    }
    Ok(output)
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.cursor
    }

    fn read_exact(&mut self, length: usize) -> Result<&'a [u8], MeshletAssetError> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or(MeshletAssetError::SizeOverflow("decoder cursor"))?;
        if end > self.bytes.len() {
            return Err(MeshletAssetError::InvalidFormat("truncated record".into()));
        }
        let output = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(output)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], MeshletAssetError> {
        self.read_exact(N)?.try_into().map_err(|_| {
            MeshletAssetError::InvalidFormat("internal fixed-array decode failure".into())
        })
    }

    fn read_u32(&mut self) -> Result<u32, MeshletAssetError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, MeshletAssetError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_f32(&mut self) -> Result<f32, MeshletAssetError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_sphere(&mut self) -> Result<BoundsSphere, MeshletAssetError> {
        Ok(BoundsSphere {
            center: [self.read_f32()?, self.read_f32()?, self.read_f32()?],
            radius: self.read_f32()?,
        })
    }

    fn read_cone(&mut self) -> Result<NormalCone, MeshletAssetError> {
        Ok(NormalCone {
            axis: [self.read_f32()?, self.read_f32()?, self.read_f32()?],
            cutoff: self.read_f32()?,
        })
    }
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(output: &mut Vec<u8>, value: f32) {
    push_u32(output, value.to_bits());
}

fn push_sphere(output: &mut Vec<u8>, sphere: BoundsSphere) {
    for value in sphere.center {
        push_f32(output, value);
    }
    push_f32(output, sphere.radius);
}

fn push_cone(output: &mut Vec<u8>, cone: NormalCone) {
    for value in cone.axis {
        push_f32(output, value);
    }
    push_f32(output, cone.cutoff);
}

fn pad_to(output: &mut Vec<u8>, alignment: usize) {
    let aligned = (output.len() + alignment - 1) & !(alignment - 1);
    output.resize(aligned, 0);
}

fn align_up(value: usize, alignment: usize) -> Result<usize, MeshletAssetError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(MeshletAssetError::SizeOverflow("alignment"))
}

fn usize_from_u64(value: u64, label: &'static str) -> Result<usize, MeshletAssetError> {
    usize::try_from(value).map_err(|_| MeshletAssetError::SizeOverflow(label))
}

fn checksum64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

use std::path::{Path, PathBuf};

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        files.push(path.to_owned());
        return;
    }
    let mut entries = std::fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to scan benchmark identity path {path:?}: {error}"))
        .map(|entry| {
            entry
                .expect("failed to read benchmark identity directory entry")
                .path()
        })
        .collect::<Vec<_>>();
    entries.sort();
    for entry in entries {
        collect_files(&entry, files);
    }
}

fn hash_byte(hash: &mut u64, byte: u8) {
    *hash ^= u64::from(byte);
    *hash = hash.wrapping_mul(1_099_511_628_211);
}

fn main() {
    let roots = [
        Path::new("../../Cargo.toml"),
        Path::new("../../Cargo.lock"),
        Path::new("../../crates/zen-render-mesh/src/meshlet"),
        Path::new("../../crates/zen-render-mesh/shaders/meshlet"),
        Path::new("../../crates/zen-frame-graph/src/gpu_timing.rs"),
        Path::new("../../crates/zen-frame-graph/src/execution.rs"),
        Path::new("../../crates/zen-render/src/host.rs"),
        Path::new("../../third_party/wgpu-hal-30.0.1/src/vulkan/command.rs"),
        Path::new("../../third_party/wgpu-hal-30.0.1/src/vulkan/conv.rs"),
        Path::new("src/device.rs"),
        Path::new("src/meshlet_benchmark.rs"),
        Path::new("src/bin/meshlet-gltf.rs"),
    ];
    let mut files = Vec::new();
    for root in roots {
        collect_files(root, &mut files);
    }
    files.sort();

    let mut hash = 14_695_981_039_346_656_037_u64;
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        for byte in path.to_string_lossy().replace('\\', "/").bytes() {
            hash_byte(&mut hash, byte);
        }
        hash_byte(&mut hash, 0);
        let contents = std::fs::read(&path).unwrap_or_else(|error| {
            panic!("failed to hash benchmark identity file {path:?}: {error}")
        });
        for byte in contents {
            hash_byte(&mut hash, byte);
        }
        hash_byte(&mut hash, 0xff);
    }
    println!("cargo:rustc-env=ZEN_MESHLET_SOURCE_FINGERPRINT={hash:016x}");
}

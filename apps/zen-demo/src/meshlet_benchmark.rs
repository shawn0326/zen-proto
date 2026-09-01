//! Vulkan meshlet benchmark identity, persistence, and statistics.
//!
//! The demo owns benchmark identity and persistence because a renderer profile is only valid for
//! one adapter, driver, graphics backend, scene, resolution, and deterministic camera track. This
//! module accepts already-measured nanoseconds; its only GPU-facing input is the immutable adapter
//! identity used to prevent stale Auto profiles from crossing devices or driver builds.

use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use zen_render_mesh::{MeshletBenchmarkProfile, MeshletGpuPassTimings};

/// JSON schema emitted by [`MeshletBenchmarkReport`].
pub const MESHLET_BENCHMARK_SCHEMA_VERSION: u32 = 6;
/// Required benchmark width.
pub const MESHLET_BENCHMARK_WIDTH: u32 = 1_920;
/// Required benchmark height.
pub const MESHLET_BENCHMARK_HEIGHT: u32 = 1_080;
/// Number of frames discarded before collecting samples.
pub const MESHLET_BENCHMARK_WARMUP_FRAMES: u32 = 120;
/// Minimum number of measured frames in a report.
pub const MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES: u32 = 600;
/// Maximum accepted IndexedIndirect p95 regression against the legacy baseline (10%).
pub const MESHLET_INDEXED_MAX_REGRESSION_BPS: u32 = 1_000;
/// Versioned renderer/build identity embedded in reports and auto-selection profiles.
///
/// Bump the suffix whenever benchmark comparability is intentionally broken by a geometry-pipeline
/// change. A source-control revision may be appended by packaging through `ZEN_BUILD_REVISION`.
pub const MESHLET_BENCHMARK_RENDERER_BUILD: &str = concat!(
    "zen-demo-",
    env!("CARGO_PKG_VERSION"),
    ":meshlet-gpu-driven-v1"
);
/// Stable deterministic camera track used by the executable benchmark.
pub const MESHLET_BENCHMARK_CAMERA_TRACK: &str = "meshlet-orbit-v1";

/// Renderer path measured by a benchmark run.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeshletBenchmarkRenderer {
    Legacy,
    Indexed,
    MeshOnly,
    TaskMesh,
}

impl MeshletBenchmarkRenderer {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Indexed => "indexed",
            Self::MeshOnly => "mesh",
            Self::TaskMesh => "task-mesh",
        }
    }
}

impl fmt::Display for MeshletBenchmarkRenderer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Resolution stored in a benchmark report.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshletBenchmarkResolution {
    pub width: u32,
    pub height: u32,
}

impl MeshletBenchmarkResolution {
    pub const FIXED: Self = Self {
        width: MESHLET_BENCHMARK_WIDTH,
        height: MESHLET_BENCHMARK_HEIGHT,
    };
}

/// Identity shared by the indexed and task-mesh runs of one comparison.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshletBenchmarkContext {
    /// Graphics API/backend name, for example `vulkan` or `dx12`.
    pub backend: String,
    /// Adapter name or another stable application-provided adapter identifier.
    pub adapter: String,
    /// Backend vendor/device IDs. These are Vulkan IDs for the Vulkan-only benchmark.
    pub adapter_vendor: u32,
    pub adapter_device: u32,
    pub adapter_device_type: String,
    /// Driver identity, including a version when the platform exposes one.
    pub driver: String,
    pub driver_info: String,
    /// Versioned identity for renderer/shader/benchmark compatibility.
    pub renderer_build: String,
    /// Stable scene identifier, including the asset revision when relevant.
    pub scene: String,
    /// Stable name/version of the deterministic camera path.
    pub camera_track: String,
}

impl MeshletBenchmarkContext {
    #[must_use]
    pub fn new(
        backend: impl Into<String>,
        adapter: impl Into<String>,
        driver: impl Into<String>,
        scene: impl Into<String>,
        camera_track: impl Into<String>,
    ) -> Self {
        Self {
            backend: backend.into(),
            adapter: adapter.into(),
            adapter_vendor: 0,
            adapter_device: 0,
            adapter_device_type: "unknown".into(),
            driver: driver.into(),
            driver_info: "unknown".into(),
            renderer_build: benchmark_renderer_build_id(),
            scene: scene.into(),
            camera_track: camera_track.into(),
        }
    }

    /// Builds the complete Vulkan identity used by executable reports and auto profiles.
    #[must_use]
    pub fn from_vulkan_adapter(adapter: &wgpu::AdapterInfo, scene: impl Into<String>) -> Self {
        Self {
            backend: "vulkan".into(),
            adapter: identity_text(&adapter.name),
            adapter_vendor: adapter.vendor,
            adapter_device: adapter.device,
            adapter_device_type: format!("{:?}", adapter.device_type),
            driver: identity_text(&adapter.driver),
            driver_info: identity_text(&adapter.driver_info),
            renderer_build: benchmark_renderer_build_id(),
            scene: scene.into(),
            camera_track: MESHLET_BENCHMARK_CAMERA_TRACK.into(),
        }
    }

    fn validate(&self) -> Result<(), MeshletBenchmarkError> {
        for (field, value) in [
            ("backend", self.backend.as_str()),
            ("adapter", self.adapter.as_str()),
            ("adapter_device_type", self.adapter_device_type.as_str()),
            ("driver", self.driver.as_str()),
            ("driver_info", self.driver_info.as_str()),
            ("renderer_build", self.renderer_build.as_str()),
            ("scene", self.scene.as_str()),
            ("camera_track", self.camera_track.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(MeshletBenchmarkError::MissingIdentity { field });
            }
        }
        Ok(())
    }
}

/// Build identity with a deterministic renderer/shader source fingerprint and optional package
/// revision. Local shader edits therefore invalidate persisted Auto profiles even when the crate
/// version has not changed.
#[must_use]
pub fn benchmark_renderer_build_id() -> String {
    let base = format!(
        "{MESHLET_BENCHMARK_RENDERER_BUILD}:src-{}",
        env!("ZEN_MESHLET_SOURCE_FINGERPRINT")
    );
    let revision = option_env!("ZEN_BUILD_REVISION")
        .or(option_env!("GIT_COMMIT_HASH"))
        .or(option_env!("VERGEN_GIT_SHA"));
    match revision {
        Some(revision) if !revision.trim().is_empty() => {
            format!("{base}:{revision}")
        }
        _ => base,
    }
}

fn identity_text(value: &str) -> String {
    if value.trim().is_empty() {
        "unknown".into()
    } else {
        value.to_owned()
    }
}

/// Nanosecond timings supplied by the benchmark harness for one frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletFrameSampleNs {
    /// Whole CPU `RenderHost::render_frame` duration. This is always available.
    pub cpu_frame_ns: u64,
    /// Exact split timings supplied by `RenderHost` when the execution completed successfully.
    /// They remain optional so imported or older reports cannot misrepresent unavailable data.
    pub cpu_encode_ns: Option<u64>,
    pub cpu_submit_ns: Option<u64>,
    pub gpu_frame_ns: u64,
    /// Fixed-topology meshlet pass timings. Legacy and unavailable paths remain `None`.
    pub gpu_passes: MeshletGpuPassTimings,
}

impl MeshletFrameSampleNs {
    #[must_use]
    pub fn new(cpu_encode_ns: u64, cpu_submit_ns: u64, gpu_frame_ns: u64) -> Self {
        Self {
            cpu_frame_ns: cpu_encode_ns.saturating_add(cpu_submit_ns),
            cpu_encode_ns: Some(cpu_encode_ns),
            cpu_submit_ns: Some(cpu_submit_ns),
            gpu_frame_ns,
            gpu_passes: MeshletGpuPassTimings::default(),
        }
    }

    /// Creates an honest combined CPU sample when encode/submit cannot be separated by the host.
    #[must_use]
    pub fn combined(cpu_frame_ns: u64, gpu_frame_ns: u64) -> Self {
        Self {
            cpu_frame_ns,
            cpu_encode_ns: None,
            cpu_submit_ns: None,
            gpu_frame_ns,
            gpu_passes: MeshletGpuPassTimings::default(),
        }
    }

    #[must_use]
    pub fn split(
        cpu_frame_ns: u64,
        cpu_encode_ns: u64,
        cpu_submit_ns: u64,
        gpu_frame_ns: u64,
    ) -> Self {
        Self {
            cpu_frame_ns,
            cpu_encode_ns: Some(cpu_encode_ns),
            cpu_submit_ns: Some(cpu_submit_ns),
            gpu_frame_ns,
            gpu_passes: MeshletGpuPassTimings::default(),
        }
    }

    #[must_use]
    pub const fn with_gpu_passes(mut self, gpu_passes: MeshletGpuPassTimings) -> Self {
        self.gpu_passes = gpu_passes;
        self
    }
}

/// Integer nanosecond summary. Even-sized medians are the midpoint rounded down; p95 uses the
/// nearest-rank definition.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshletBenchmarkStatisticsNs {
    pub median: u64,
    pub p95: u64,
}

impl MeshletBenchmarkStatisticsNs {
    #[must_use]
    pub fn from_samples(samples: &[u64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let midpoint = sorted.len() / 2;
        let median = if sorted.len().is_multiple_of(2) {
            let lower = sorted[midpoint - 1];
            let upper = sorted[midpoint];
            lower + (upper - lower) / 2
        } else {
            sorted[midpoint]
        };
        let p95_rank = ((sorted.len() as u128 * 95).div_ceil(100)) as usize;
        let p95 = sorted[p95_rank - 1];

        Some(Self { median, p95 })
    }
}

/// Per-pass GPU summaries. A path is `None` unless every sampled frame contained that pass.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshletBenchmarkGpuPassStatisticsNs {
    pub clear_frame_counters: Option<MeshletBenchmarkStatisticsNs>,
    pub instance_classify_lod_count: Option<MeshletBenchmarkStatisticsNs>,
    pub prefix_scan: Option<MeshletBenchmarkStatisticsNs>,
    pub candidate_scatter: Option<MeshletBenchmarkStatisticsNs>,
    pub coarse_cull: Option<MeshletBenchmarkStatisticsNs>,
    pub occluder_depth: Option<MeshletBenchmarkStatisticsNs>,
    pub hiz_build: Option<MeshletBenchmarkStatisticsNs>,
    pub clear_coarse_results: Option<MeshletBenchmarkStatisticsNs>,
    pub final_cull: Option<MeshletBenchmarkStatisticsNs>,
    pub indirect_prepare: Option<MeshletBenchmarkStatisticsNs>,
    pub backend_raster: Option<MeshletBenchmarkStatisticsNs>,
    pub stats_copy: Option<MeshletBenchmarkStatisticsNs>,
}

impl MeshletBenchmarkGpuPassStatisticsNs {
    fn from_samples(samples: &[MeshletFrameSampleNs]) -> Self {
        let summarize = |select: fn(MeshletGpuPassTimings) -> Option<u64>| {
            let values = samples
                .iter()
                .map(|sample| select(sample.gpu_passes))
                .collect::<Option<Vec<_>>>()?;
            MeshletBenchmarkStatisticsNs::from_samples(&values)
        };
        Self {
            clear_frame_counters: summarize(|timing| timing.clear_frame_counters_ns),
            instance_classify_lod_count: summarize(|timing| timing.instance_classify_lod_count_ns),
            prefix_scan: summarize(|timing| timing.prefix_scan_ns),
            candidate_scatter: summarize(|timing| timing.candidate_scatter_ns),
            coarse_cull: summarize(|timing| timing.coarse_cull_ns),
            occluder_depth: summarize(|timing| timing.occluder_depth_ns),
            hiz_build: summarize(|timing| timing.hiz_build_ns),
            clear_coarse_results: summarize(|timing| timing.clear_coarse_results_ns),
            final_cull: summarize(|timing| timing.final_cull_ns),
            indirect_prepare: summarize(|timing| timing.indirect_prepare_ns),
            backend_raster: summarize(|timing| timing.backend_raster_ns),
            stats_copy: summarize(|timing| timing.stats_copy_ns),
        }
    }

    fn named_statistics(&self) -> [(&'static str, Option<MeshletBenchmarkStatisticsNs>); 12] {
        [
            (
                "gpu_passes_ns.clear_frame_counters",
                self.clear_frame_counters,
            ),
            (
                "gpu_passes_ns.instance_classify_lod_count",
                self.instance_classify_lod_count,
            ),
            ("gpu_passes_ns.prefix_scan", self.prefix_scan),
            ("gpu_passes_ns.candidate_scatter", self.candidate_scatter),
            ("gpu_passes_ns.coarse_cull", self.coarse_cull),
            ("gpu_passes_ns.occluder_depth", self.occluder_depth),
            ("gpu_passes_ns.hiz_build", self.hiz_build),
            (
                "gpu_passes_ns.clear_coarse_results",
                self.clear_coarse_results,
            ),
            ("gpu_passes_ns.final_cull", self.final_cull),
            ("gpu_passes_ns.indirect_prepare", self.indirect_prepare),
            ("gpu_passes_ns.backend_raster", self.backend_raster),
            ("gpu_passes_ns.stats_copy", self.stats_copy),
        ]
    }
}

/// Persistable result for one renderer path.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshletBenchmarkReport {
    pub schema_version: u32,
    pub renderer: MeshletBenchmarkRenderer,
    #[serde(flatten)]
    pub context: MeshletBenchmarkContext,
    pub resolution: MeshletBenchmarkResolution,
    pub geometry_bound: bool,
    pub warmup_frames: u32,
    pub sample_frames: u32,
    /// Must remain false. Meshlet runs capture counters for every timestamped frame and reject a
    /// report if any fixed-capacity GPU arena clamped work.
    pub capacity_overflow_observed: bool,
    pub cpu_frame_ns: MeshletBenchmarkStatisticsNs,
    pub cpu_encode_ns: Option<MeshletBenchmarkStatisticsNs>,
    pub cpu_submit_ns: Option<MeshletBenchmarkStatisticsNs>,
    pub gpu_frame_ns: MeshletBenchmarkStatisticsNs,
    pub gpu_passes_ns: MeshletBenchmarkGpuPassStatisticsNs,
}

impl MeshletBenchmarkReport {
    /// Encodes a stable, human-readable JSON representation.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Decodes a report. Call [`Self::validate_contract`] before trusting externally supplied data.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Checks the fixed benchmark contract and internal summary consistency.
    pub fn validate_contract(&self) -> Result<(), MeshletBenchmarkError> {
        if self.schema_version != MESHLET_BENCHMARK_SCHEMA_VERSION {
            return Err(MeshletBenchmarkError::UnsupportedSchemaVersion {
                actual: self.schema_version,
            });
        }
        self.context.validate()?;
        if self.resolution != MeshletBenchmarkResolution::FIXED {
            return Err(MeshletBenchmarkError::InvalidResolution {
                actual: self.resolution,
            });
        }
        if self.warmup_frames != MESHLET_BENCHMARK_WARMUP_FRAMES {
            return Err(MeshletBenchmarkError::InvalidWarmupFrames {
                actual: self.warmup_frames,
            });
        }
        if self.sample_frames < MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES {
            return Err(MeshletBenchmarkError::InsufficientSamples {
                actual: self.sample_frames,
            });
        }
        if self.capacity_overflow_observed {
            return Err(MeshletBenchmarkError::CapacityOverflowObserved);
        }
        self.validate_gpu_timing_coverage()?;
        let mut statistics = vec![
            ("cpu_frame_ns", self.cpu_frame_ns),
            ("gpu_frame_ns", self.gpu_frame_ns),
        ];
        if let Some(value) = self.cpu_encode_ns {
            statistics.push(("cpu_encode_ns", value));
        }
        if let Some(value) = self.cpu_submit_ns {
            statistics.push(("cpu_submit_ns", value));
        }
        if self.cpu_encode_ns.is_some() != self.cpu_submit_ns.is_some() {
            return Err(MeshletBenchmarkError::IncompleteCpuTimingSplit);
        }
        statistics.extend(
            self.gpu_passes_ns
                .named_statistics()
                .into_iter()
                .filter_map(|(name, value)| value.map(|value| (name, value))),
        );
        for (metric, statistics) in statistics {
            if statistics.p95 < statistics.median {
                return Err(MeshletBenchmarkError::InvalidStatistics { metric });
            }
        }
        Ok(())
    }

    fn validate_gpu_timing_coverage(&self) -> Result<(), MeshletBenchmarkError> {
        let passes = &self.gpu_passes_ns;
        let named = passes.named_statistics();
        if self.renderer == MeshletBenchmarkRenderer::Legacy {
            if let Some((pass, _)) = named.into_iter().find(|(_, timing)| timing.is_some()) {
                return Err(MeshletBenchmarkError::UnexpectedGpuPassTiming {
                    renderer: self.renderer,
                    pass,
                });
            }
            return Ok(());
        }

        for (pass, timing) in [
            (
                "gpu_passes_ns.clear_frame_counters",
                passes.clear_frame_counters,
            ),
            (
                "gpu_passes_ns.instance_classify_lod_count",
                passes.instance_classify_lod_count,
            ),
            ("gpu_passes_ns.prefix_scan", passes.prefix_scan),
            ("gpu_passes_ns.candidate_scatter", passes.candidate_scatter),
            ("gpu_passes_ns.coarse_cull", passes.coarse_cull),
            ("gpu_passes_ns.occluder_depth", passes.occluder_depth),
            ("gpu_passes_ns.hiz_build", passes.hiz_build),
            (
                "gpu_passes_ns.clear_coarse_results",
                passes.clear_coarse_results,
            ),
            ("gpu_passes_ns.final_cull", passes.final_cull),
            ("gpu_passes_ns.indirect_prepare", passes.indirect_prepare),
            ("gpu_passes_ns.backend_raster", passes.backend_raster),
            ("gpu_passes_ns.stats_copy", passes.stats_copy),
        ] {
            if timing.is_none() {
                return Err(MeshletBenchmarkError::MissingGpuPassTiming {
                    renderer: self.renderer,
                    pass,
                });
            }
        }
        Ok(())
    }

    pub fn read_json_file(path: impl AsRef<Path>) -> Result<Self, MeshletBenchmarkFileError> {
        let path = path.as_ref();
        let json = fs::read_to_string(path).map_err(|source| MeshletBenchmarkFileError::Read {
            path: path.to_owned(),
            source,
        })?;
        let report = Self::from_json(&json).map_err(|source| MeshletBenchmarkFileError::Json {
            path: path.to_owned(),
            source,
        })?;
        report.validate_contract()?;
        Ok(report)
    }

    pub fn write_json_file(&self, path: impl AsRef<Path>) -> Result<(), MeshletBenchmarkFileError> {
        self.validate_contract()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|source| MeshletBenchmarkFileError::Write {
                path: path.to_owned(),
                source,
            })?;
        }
        let json = self
            .to_json_pretty()
            .map_err(|source| MeshletBenchmarkFileError::Json {
                path: path.to_owned(),
                source,
            })?;
        fs::write(path, format!("{json}\n")).map_err(|source| MeshletBenchmarkFileError::Write {
            path: path.to_owned(),
            source,
        })
    }
}

/// Collects measured timings after discarding exactly the fixed warm-up interval.
#[derive(Debug)]
pub struct MeshletBenchmarkCollector {
    renderer: MeshletBenchmarkRenderer,
    context: MeshletBenchmarkContext,
    geometry_bound: bool,
    warmup_frames: u32,
    capacity_overflow_observed: bool,
    samples: Vec<MeshletFrameSampleNs>,
}

impl MeshletBenchmarkCollector {
    #[must_use]
    pub fn new(
        renderer: MeshletBenchmarkRenderer,
        context: MeshletBenchmarkContext,
        geometry_bound: bool,
    ) -> Self {
        Self {
            renderer,
            context,
            geometry_bound,
            warmup_frames: 0,
            capacity_overflow_observed: false,
            samples: Vec::with_capacity(MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES as usize),
        }
    }

    /// Records one presented frame. Warm-up timings are intentionally discarded.
    pub fn record_frame(&mut self, sample: MeshletFrameSampleNs) {
        if self.warmup_frames < MESHLET_BENCHMARK_WARMUP_FRAMES {
            self.warmup_frames += 1;
        } else {
            self.samples.push(sample);
        }
    }

    #[must_use]
    pub const fn warmup_frames_recorded(&self) -> u32 {
        self.warmup_frames
    }

    #[must_use]
    pub fn sample_frames_recorded(&self) -> usize {
        self.samples.len()
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.warmup_frames == MESHLET_BENCHMARK_WARMUP_FRAMES
            && self.samples.len() >= MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES as usize
    }

    /// Finishes the report once all warm-up and minimum sample frames have been observed.
    pub fn finish(self) -> Result<MeshletBenchmarkReport, MeshletBenchmarkError> {
        if self.warmup_frames != MESHLET_BENCHMARK_WARMUP_FRAMES {
            return Err(MeshletBenchmarkError::IncompleteWarmup {
                actual: self.warmup_frames,
            });
        }
        self.context.validate()?;
        let sample_frames = u32::try_from(self.samples.len())
            .map_err(|_| MeshletBenchmarkError::SampleCountOverflow)?;
        if sample_frames < MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES {
            return Err(MeshletBenchmarkError::InsufficientSamples {
                actual: sample_frames,
            });
        }

        let gpu_passes_ns = MeshletBenchmarkGpuPassStatisticsNs::from_samples(&self.samples);
        let mut cpu_frame_samples = Vec::with_capacity(self.samples.len());
        let mut cpu_encode_samples = Vec::with_capacity(self.samples.len());
        let mut cpu_submit_samples = Vec::with_capacity(self.samples.len());
        let mut has_cpu_split = true;
        let mut gpu_frame_samples = Vec::with_capacity(self.samples.len());
        for sample in self.samples {
            cpu_frame_samples.push(sample.cpu_frame_ns);
            match (sample.cpu_encode_ns, sample.cpu_submit_ns) {
                (Some(encode), Some(submit)) if has_cpu_split => {
                    cpu_encode_samples.push(encode);
                    cpu_submit_samples.push(submit);
                }
                _ => {
                    has_cpu_split = false;
                    cpu_encode_samples.clear();
                    cpu_submit_samples.clear();
                }
            }
            gpu_frame_samples.push(sample.gpu_frame_ns);
        }

        Ok(MeshletBenchmarkReport {
            schema_version: MESHLET_BENCHMARK_SCHEMA_VERSION,
            renderer: self.renderer,
            context: self.context,
            resolution: MeshletBenchmarkResolution::FIXED,
            geometry_bound: self.geometry_bound,
            warmup_frames: self.warmup_frames,
            sample_frames,
            capacity_overflow_observed: self.capacity_overflow_observed,
            cpu_frame_ns: MeshletBenchmarkStatisticsNs::from_samples(&cpu_frame_samples)
                .expect("the minimum sample count is non-zero"),
            cpu_encode_ns: has_cpu_split
                .then(|| MeshletBenchmarkStatisticsNs::from_samples(&cpu_encode_samples).unwrap()),
            cpu_submit_ns: has_cpu_split
                .then(|| MeshletBenchmarkStatisticsNs::from_samples(&cpu_submit_samples).unwrap()),
            gpu_frame_ns: MeshletBenchmarkStatisticsNs::from_samples(&gpu_frame_samples)
                .expect("the minimum sample count is non-zero"),
            gpu_passes_ns,
        })
    }
}

/// Builds the renderer's compact promotion profile from two comparable persisted reports.
///
/// This function deliberately does not reimplement the 10% promotion calculation. Call
/// [`MeshletBenchmarkProfile::approves_task_mesh`] on the returned profile so the renderer remains
/// the single source of truth for the speedup gate.
pub fn meshlet_benchmark_profile_from_reports(
    indexed: &MeshletBenchmarkReport,
    task_mesh: &MeshletBenchmarkReport,
) -> Result<MeshletBenchmarkProfile, MeshletBenchmarkError> {
    indexed.validate_contract()?;
    task_mesh.validate_contract()?;
    expect_renderer(indexed, "indexed report", MeshletBenchmarkRenderer::Indexed)?;
    expect_renderer(
        task_mesh,
        "task-mesh report",
        MeshletBenchmarkRenderer::TaskMesh,
    )?;

    if !indexed.geometry_bound {
        return Err(MeshletBenchmarkError::NotGeometryBound {
            renderer: indexed.renderer,
        });
    }
    if !task_mesh.geometry_bound {
        return Err(MeshletBenchmarkError::NotGeometryBound {
            renderer: task_mesh.renderer,
        });
    }

    ensure_same(
        "backend",
        &indexed.context.backend,
        &task_mesh.context.backend,
    )?;
    if !indexed.context.backend.eq_ignore_ascii_case("vulkan") {
        return Err(MeshletBenchmarkError::UnsupportedBackend {
            actual: indexed.context.backend.clone(),
        });
    }
    ensure_same(
        "adapter",
        &indexed.context.adapter,
        &task_mesh.context.adapter,
    )?;
    if indexed.context.adapter_vendor != task_mesh.context.adapter_vendor {
        return Err(MeshletBenchmarkError::MismatchedContext {
            field: "adapter_vendor",
        });
    }
    if indexed.context.adapter_device != task_mesh.context.adapter_device {
        return Err(MeshletBenchmarkError::MismatchedContext {
            field: "adapter_device",
        });
    }
    ensure_same(
        "adapter_device_type",
        &indexed.context.adapter_device_type,
        &task_mesh.context.adapter_device_type,
    )?;
    ensure_same("driver", &indexed.context.driver, &task_mesh.context.driver)?;
    ensure_same(
        "driver_info",
        &indexed.context.driver_info,
        &task_mesh.context.driver_info,
    )?;
    ensure_same(
        "renderer_build",
        &indexed.context.renderer_build,
        &task_mesh.context.renderer_build,
    )?;
    ensure_same("scene", &indexed.context.scene, &task_mesh.context.scene)?;
    ensure_same(
        "camera_track",
        &indexed.context.camera_track,
        &task_mesh.context.camera_track,
    )?;
    if indexed.resolution != task_mesh.resolution {
        return Err(MeshletBenchmarkError::MismatchedContext {
            field: "resolution",
        });
    }

    Ok(MeshletBenchmarkProfile {
        geometry_bound: true,
        warmup_frames: indexed.warmup_frames.min(task_mesh.warmup_frames),
        sample_frames: indexed.sample_frames.min(task_mesh.sample_frames),
        indexed_gpu_p95_ns: indexed.gpu_frame_ns.p95,
        task_mesh_gpu_p95_ns: task_mesh.gpu_frame_ns.p95,
    })
}

/// Checks the acceptance gate for the new IndexedIndirect path against the legacy Vulkan baseline.
pub fn indexed_regression_is_acceptable(
    legacy: &MeshletBenchmarkReport,
    indexed: &MeshletBenchmarkReport,
) -> Result<bool, MeshletBenchmarkError> {
    legacy.validate_contract()?;
    indexed.validate_contract()?;
    expect_renderer(legacy, "legacy report", MeshletBenchmarkRenderer::Legacy)?;
    expect_renderer(indexed, "indexed report", MeshletBenchmarkRenderer::Indexed)?;
    if legacy.context != indexed.context {
        return Err(MeshletBenchmarkError::MismatchedContext { field: "context" });
    }
    if legacy.resolution != indexed.resolution {
        return Err(MeshletBenchmarkError::MismatchedContext {
            field: "resolution",
        });
    }
    let allowed_bps = 10_000u128 + u128::from(MESHLET_INDEXED_MAX_REGRESSION_BPS);
    Ok(u128::from(indexed.gpu_frame_ns.p95) * 10_000
        <= u128::from(legacy.gpu_frame_ns.p95) * allowed_bps)
}

/// Persisted, identity-bound profile consumed by `--renderer auto`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshletAutoProfileFile {
    pub schema_version: u32,
    pub context: MeshletBenchmarkContext,
    pub resolution: MeshletBenchmarkResolution,
    pub profile: MeshletAutoProfileMeasurements,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshletAutoProfileMeasurements {
    pub geometry_bound: bool,
    pub warmup_frames: u32,
    pub sample_frames: u32,
    pub indexed_gpu_p95_ns: u64,
    pub task_mesh_gpu_p95_ns: u64,
}

impl From<MeshletBenchmarkProfile> for MeshletAutoProfileMeasurements {
    fn from(value: MeshletBenchmarkProfile) -> Self {
        Self {
            geometry_bound: value.geometry_bound,
            warmup_frames: value.warmup_frames,
            sample_frames: value.sample_frames,
            indexed_gpu_p95_ns: value.indexed_gpu_p95_ns,
            task_mesh_gpu_p95_ns: value.task_mesh_gpu_p95_ns,
        }
    }
}

impl From<MeshletAutoProfileMeasurements> for MeshletBenchmarkProfile {
    fn from(value: MeshletAutoProfileMeasurements) -> Self {
        Self {
            geometry_bound: value.geometry_bound,
            warmup_frames: value.warmup_frames,
            sample_frames: value.sample_frames,
            indexed_gpu_p95_ns: value.indexed_gpu_p95_ns,
            task_mesh_gpu_p95_ns: value.task_mesh_gpu_p95_ns,
        }
    }
}

impl MeshletAutoProfileFile {
    pub fn from_reports(
        indexed: &MeshletBenchmarkReport,
        task_mesh: &MeshletBenchmarkReport,
    ) -> Result<Self, MeshletBenchmarkError> {
        let profile = meshlet_benchmark_profile_from_reports(indexed, task_mesh)?;
        if !profile.approves_task_mesh() {
            return Err(MeshletBenchmarkError::ProfileDoesNotApproveTaskMesh);
        }
        Ok(Self {
            schema_version: MESHLET_BENCHMARK_SCHEMA_VERSION,
            context: indexed.context.clone(),
            resolution: indexed.resolution,
            profile: profile.into(),
        })
    }

    pub fn validate_for(
        &self,
        adapter: &wgpu::AdapterInfo,
        scene: &str,
    ) -> Result<(), MeshletBenchmarkError> {
        if self.schema_version != MESHLET_BENCHMARK_SCHEMA_VERSION {
            return Err(MeshletBenchmarkError::UnsupportedSchemaVersion {
                actual: self.schema_version,
            });
        }
        self.context.validate()?;
        if self.resolution != MeshletBenchmarkResolution::FIXED {
            return Err(MeshletBenchmarkError::InvalidResolution {
                actual: self.resolution,
            });
        }
        if adapter.backend != wgpu::Backend::Vulkan
            || !self.context.backend.eq_ignore_ascii_case("vulkan")
        {
            return Err(MeshletBenchmarkError::UnsupportedBackend {
                actual: self.context.backend.clone(),
            });
        }
        for (field, matches) in [
            (
                "adapter",
                self.context.adapter == identity_text(&adapter.name),
            ),
            (
                "adapter_vendor",
                self.context.adapter_vendor == adapter.vendor,
            ),
            (
                "adapter_device",
                self.context.adapter_device == adapter.device,
            ),
            (
                "driver",
                self.context.driver == identity_text(&adapter.driver),
            ),
            (
                "driver_info",
                self.context.driver_info == identity_text(&adapter.driver_info),
            ),
            (
                "adapter_device_type",
                self.context.adapter_device_type == format!("{:?}", adapter.device_type),
            ),
            (
                "renderer_build",
                self.context.renderer_build == benchmark_renderer_build_id(),
            ),
            ("scene", self.context.scene == scene),
            (
                "camera_track",
                self.context.camera_track == MESHLET_BENCHMARK_CAMERA_TRACK,
            ),
        ] {
            if !matches {
                return Err(MeshletBenchmarkError::ProfileIdentityMismatch { field });
            }
        }
        if !MeshletBenchmarkProfile::from(self.profile).approves_task_mesh() {
            return Err(MeshletBenchmarkError::ProfileDoesNotApproveTaskMesh);
        }
        Ok(())
    }

    #[must_use]
    pub fn renderer_profile(&self) -> MeshletBenchmarkProfile {
        self.profile.into()
    }

    pub fn read_json_file(path: impl AsRef<Path>) -> Result<Self, MeshletBenchmarkFileError> {
        let path = path.as_ref();
        let json = fs::read_to_string(path).map_err(|source| MeshletBenchmarkFileError::Read {
            path: path.to_owned(),
            source,
        })?;
        serde_json::from_str(&json).map_err(|source| MeshletBenchmarkFileError::Json {
            path: path.to_owned(),
            source,
        })
    }

    pub fn write_json_file(&self, path: impl AsRef<Path>) -> Result<(), MeshletBenchmarkFileError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|source| MeshletBenchmarkFileError::Write {
                path: path.to_owned(),
                source,
            })?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|source| {
            MeshletBenchmarkFileError::Json {
                path: path.to_owned(),
                source,
            }
        })?;
        fs::write(path, format!("{json}\n")).map_err(|source| MeshletBenchmarkFileError::Write {
            path: path.to_owned(),
            source,
        })
    }
}

fn expect_renderer(
    report: &MeshletBenchmarkReport,
    report_name: &'static str,
    expected: MeshletBenchmarkRenderer,
) -> Result<(), MeshletBenchmarkError> {
    if report.renderer != expected {
        return Err(MeshletBenchmarkError::UnexpectedRenderer {
            report: report_name,
            expected,
            actual: report.renderer,
        });
    }
    Ok(())
}

fn ensure_same(
    field: &'static str,
    indexed: &str,
    task_mesh: &str,
) -> Result<(), MeshletBenchmarkError> {
    if indexed != task_mesh {
        return Err(MeshletBenchmarkError::MismatchedContext { field });
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MeshletBenchmarkError {
    #[error("meshlet benchmark identity field {field} is empty")]
    MissingIdentity { field: &'static str },
    #[error("unsupported meshlet benchmark schema version {actual}")]
    UnsupportedSchemaVersion { actual: u32 },
    #[error(
        "meshlet benchmark resolution must be 1920x1080, got {actual_width}x{actual_height}",
        actual_width = .actual.width,
        actual_height = .actual.height
    )]
    InvalidResolution { actual: MeshletBenchmarkResolution },
    #[error("meshlet benchmark requires exactly 120 warm-up frames, got {actual}")]
    InvalidWarmupFrames { actual: u32 },
    #[error("meshlet benchmark warm-up is incomplete: recorded {actual} of 120 frames")]
    IncompleteWarmup { actual: u32 },
    #[error("meshlet benchmark requires at least 600 samples, got {actual}")]
    InsufficientSamples { actual: u32 },
    #[error("meshlet benchmark sample count does not fit in u32")]
    SampleCountOverflow,
    #[error("meshlet benchmark observed a fixed-capacity GPU overflow")]
    CapacityOverflowObserved,
    #[error("{renderer} benchmark is missing required timing {pass}")]
    MissingGpuPassTiming {
        renderer: MeshletBenchmarkRenderer,
        pass: &'static str,
    },
    #[error("{renderer} benchmark unexpectedly contains timing {pass}")]
    UnexpectedGpuPassTiming {
        renderer: MeshletBenchmarkRenderer,
        pass: &'static str,
    },
    #[error("meshlet benchmark statistics for {metric} have p95 below median")]
    InvalidStatistics { metric: &'static str },
    #[error("meshlet benchmark must provide both CPU encode and submit timings, or neither")]
    IncompleteCpuTimingSplit,
    #[error("{report} must use {expected}, got {actual}")]
    UnexpectedRenderer {
        report: &'static str,
        expected: MeshletBenchmarkRenderer,
        actual: MeshletBenchmarkRenderer,
    },
    #[error("{renderer} benchmark is not geometry-bound")]
    NotGeometryBound { renderer: MeshletBenchmarkRenderer },
    #[error("indexed and task-mesh reports differ in {field}")]
    MismatchedContext { field: &'static str },
    #[error("meshlet benchmark profiles require the Vulkan backend, got {actual:?}")]
    UnsupportedBackend { actual: String },
    #[error("meshlet auto profile does not match current {field}")]
    ProfileIdentityMismatch { field: &'static str },
    #[error("meshlet auto profile does not satisfy the TaskMesh promotion gate")]
    ProfileDoesNotApproveTaskMesh,
}

#[derive(Debug, thiserror::Error)]
pub enum MeshletBenchmarkFileError {
    #[error("failed to read meshlet benchmark JSON {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write meshlet benchmark JSON {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid meshlet benchmark JSON {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Contract(#[from] MeshletBenchmarkError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> MeshletBenchmarkContext {
        MeshletBenchmarkContext::new(
            "vulkan",
            "Example Adapter",
            "Example Driver 1.2.3",
            "bistro-v1",
            "orbit-v1",
        )
    }

    fn report(renderer: MeshletBenchmarkRenderer, gpu_p95_ns: u64) -> MeshletBenchmarkReport {
        let pass = Some(MeshletBenchmarkStatisticsNs { median: 1, p95: 2 });
        let gpu_passes_ns = match renderer {
            MeshletBenchmarkRenderer::Legacy => MeshletBenchmarkGpuPassStatisticsNs::default(),
            MeshletBenchmarkRenderer::Indexed
            | MeshletBenchmarkRenderer::MeshOnly
            | MeshletBenchmarkRenderer::TaskMesh => MeshletBenchmarkGpuPassStatisticsNs {
                clear_frame_counters: pass,
                instance_classify_lod_count: pass,
                prefix_scan: pass,
                candidate_scatter: pass,
                coarse_cull: pass,
                occluder_depth: pass,
                hiz_build: pass,
                clear_coarse_results: pass,
                final_cull: pass,
                indirect_prepare: pass,
                backend_raster: pass,
                stats_copy: pass,
            },
        };
        MeshletBenchmarkReport {
            schema_version: MESHLET_BENCHMARK_SCHEMA_VERSION,
            renderer,
            context: context(),
            resolution: MeshletBenchmarkResolution::FIXED,
            geometry_bound: true,
            warmup_frames: MESHLET_BENCHMARK_WARMUP_FRAMES,
            sample_frames: MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES,
            capacity_overflow_observed: false,
            cpu_frame_ns: MeshletBenchmarkStatisticsNs {
                median: 15,
                p95: 28,
            },
            cpu_encode_ns: Some(MeshletBenchmarkStatisticsNs {
                median: 10,
                p95: 20,
            }),
            cpu_submit_ns: Some(MeshletBenchmarkStatisticsNs { median: 5, p95: 8 }),
            gpu_frame_ns: MeshletBenchmarkStatisticsNs {
                median: gpu_p95_ns.saturating_sub(10),
                p95: gpu_p95_ns,
            },
            gpu_passes_ns,
        }
    }

    #[test]
    fn collector_discards_warmup_and_computes_median_and_nearest_rank_p95() {
        let mut collector =
            MeshletBenchmarkCollector::new(MeshletBenchmarkRenderer::Indexed, context(), true);
        for _ in 0..MESHLET_BENCHMARK_WARMUP_FRAMES {
            collector.record_frame(MeshletFrameSampleNs::new(u64::MAX, u64::MAX, u64::MAX));
        }
        for value in 1..=MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES as u64 {
            collector.record_frame(
                MeshletFrameSampleNs::new(value, value + 1_000, value + 2_000).with_gpu_passes(
                    MeshletGpuPassTimings {
                        prefix_scan_ns: Some(value * 3),
                        backend_raster_ns: Some(value * 4),
                        ..Default::default()
                    },
                ),
            );
        }

        assert!(collector.is_ready());
        let report = collector.finish().unwrap();
        assert_eq!(report.resolution, MeshletBenchmarkResolution::FIXED);
        assert_eq!(report.warmup_frames, 120);
        assert_eq!(report.sample_frames, 600);
        assert_eq!(
            report.cpu_encode_ns,
            Some(MeshletBenchmarkStatisticsNs {
                median: 300,
                p95: 570
            })
        );
        assert_eq!(report.cpu_submit_ns.unwrap().median, 1_300);
        assert_eq!(report.cpu_submit_ns.unwrap().p95, 1_570);
        assert_eq!(report.cpu_frame_ns.median, 1_601);
        assert_eq!(report.cpu_frame_ns.p95, 2_140);
        assert_eq!(report.gpu_frame_ns.median, 2_300);
        assert_eq!(report.gpu_frame_ns.p95, 2_570);
        assert_eq!(report.gpu_passes_ns.prefix_scan.unwrap().median, 901);
        assert_eq!(report.gpu_passes_ns.prefix_scan.unwrap().p95, 1_710);
        assert_eq!(report.gpu_passes_ns.backend_raster.unwrap().median, 1_202);
        assert!(report.gpu_passes_ns.hiz_build.is_none());
    }

    #[test]
    fn p95_uses_nearest_rank_at_integer_and_fractional_boundaries() {
        let integer_boundary: Vec<_> = (1..=20).collect();
        let fractional_boundary: Vec<_> = (1..=21).collect();

        assert_eq!(
            MeshletBenchmarkStatisticsNs::from_samples(&integer_boundary)
                .unwrap()
                .p95,
            19
        );
        assert_eq!(
            MeshletBenchmarkStatisticsNs::from_samples(&fractional_boundary)
                .unwrap()
                .p95,
            20
        );
    }

    #[test]
    fn collector_rejects_incomplete_runs() {
        let collector =
            MeshletBenchmarkCollector::new(MeshletBenchmarkRenderer::Indexed, context(), true);
        assert_eq!(
            collector.finish(),
            Err(MeshletBenchmarkError::IncompleteWarmup { actual: 0 })
        );

        let mut collector =
            MeshletBenchmarkCollector::new(MeshletBenchmarkRenderer::Indexed, context(), true);
        for _ in 0..MESHLET_BENCHMARK_WARMUP_FRAMES {
            collector.record_frame(MeshletFrameSampleNs::new(1, 1, 1));
        }
        assert_eq!(
            collector.finish(),
            Err(MeshletBenchmarkError::InsufficientSamples { actual: 0 })
        );
    }

    #[test]
    fn report_json_contains_identity_counts_and_all_timing_statistics() {
        let report = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        let json = report.to_json_pretty().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["renderer"], "indexed");
        assert_eq!(value["backend"], "vulkan");
        assert_eq!(value["adapter"], "Example Adapter");
        assert_eq!(value["driver"], "Example Driver 1.2.3");
        assert_eq!(value["scene"], "bistro-v1");
        assert_eq!(value["camera_track"], "orbit-v1");
        assert_eq!(value["resolution"]["width"], 1_920);
        assert_eq!(value["resolution"]["height"], 1_080);
        assert_eq!(value["warmup_frames"], 120);
        assert_eq!(value["sample_frames"], 600);
        assert_eq!(value["capacity_overflow_observed"], false);
        assert_eq!(value["cpu_frame_ns"]["median"], 15);
        assert_eq!(value["cpu_encode_ns"]["median"], 10);
        assert_eq!(value["cpu_submit_ns"]["p95"], 8);
        assert_eq!(value["gpu_frame_ns"]["p95"], 1_000);
        assert_eq!(value["gpu_passes_ns"]["backend_raster"]["p95"], 2);
        assert_eq!(MeshletBenchmarkReport::from_json(&json).unwrap(), report);
    }

    #[test]
    fn profile_requires_geometry_bound_comparable_runs() {
        let indexed = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        let mut task_mesh = report(MeshletBenchmarkRenderer::TaskMesh, 900);
        task_mesh.context.camera_track = "different-track".into();
        assert_eq!(
            meshlet_benchmark_profile_from_reports(&indexed, &task_mesh),
            Err(MeshletBenchmarkError::MismatchedContext {
                field: "camera_track"
            })
        );

        task_mesh.context.camera_track = indexed.context.camera_track.clone();
        task_mesh.geometry_bound = false;
        assert_eq!(
            meshlet_benchmark_profile_from_reports(&indexed, &task_mesh),
            Err(MeshletBenchmarkError::NotGeometryBound {
                renderer: MeshletBenchmarkRenderer::TaskMesh
            })
        );
    }

    #[test]
    fn profile_requires_matching_vulkan_runs() {
        let indexed = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        let mut task_mesh = report(MeshletBenchmarkRenderer::TaskMesh, 900);
        task_mesh.context.backend = "dx12".into();
        assert_eq!(
            meshlet_benchmark_profile_from_reports(&indexed, &task_mesh),
            Err(MeshletBenchmarkError::MismatchedContext { field: "backend" })
        );

        let mut indexed = indexed;
        indexed.context.backend = "dx12".into();
        assert_eq!(
            meshlet_benchmark_profile_from_reports(&indexed, &task_mesh),
            Err(MeshletBenchmarkError::UnsupportedBackend {
                actual: "dx12".into()
            })
        );
    }

    #[test]
    fn profile_owns_the_exact_ten_percent_speedup_gate() {
        let indexed = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        let task_mesh = report(MeshletBenchmarkRenderer::TaskMesh, 900);
        let profile = meshlet_benchmark_profile_from_reports(&indexed, &task_mesh).unwrap();
        assert_eq!(profile.indexed_gpu_p95_ns, 1_000);
        assert_eq!(profile.task_mesh_gpu_p95_ns, 900);
        assert!(profile.approves_task_mesh());

        let task_mesh = report(MeshletBenchmarkRenderer::TaskMesh, 901);
        let profile = meshlet_benchmark_profile_from_reports(&indexed, &task_mesh).unwrap();
        assert!(!profile.approves_task_mesh());
    }

    #[test]
    fn indexed_path_owns_the_exact_ten_percent_legacy_regression_gate() {
        let legacy = report(MeshletBenchmarkRenderer::Legacy, 1_000);
        let indexed = report(MeshletBenchmarkRenderer::Indexed, 1_100);
        assert!(indexed_regression_is_acceptable(&legacy, &indexed).unwrap());

        let indexed = report(MeshletBenchmarkRenderer::Indexed, 1_101);
        assert!(!indexed_regression_is_acceptable(&legacy, &indexed).unwrap());
    }

    #[test]
    fn deserialized_reports_must_still_satisfy_the_fixed_contract() {
        let mut invalid = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        invalid.resolution.width = 1_280;
        assert!(matches!(
            invalid.validate_contract(),
            Err(MeshletBenchmarkError::InvalidResolution { .. })
        ));

        invalid.resolution = MeshletBenchmarkResolution::FIXED;
        invalid.sample_frames = 599;
        assert_eq!(
            invalid.validate_contract(),
            Err(MeshletBenchmarkError::InsufficientSamples { actual: 599 })
        );

        let mut invalid = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        invalid.gpu_passes_ns.stats_copy = None;
        assert_eq!(
            invalid.validate_contract(),
            Err(MeshletBenchmarkError::MissingGpuPassTiming {
                renderer: MeshletBenchmarkRenderer::Indexed,
                pass: "gpu_passes_ns.stats_copy",
            })
        );

        let mut invalid = report(MeshletBenchmarkRenderer::TaskMesh, 900);
        invalid.gpu_passes_ns.final_cull = None;
        assert_eq!(
            invalid.validate_contract(),
            Err(MeshletBenchmarkError::MissingGpuPassTiming {
                renderer: MeshletBenchmarkRenderer::TaskMesh,
                pass: "gpu_passes_ns.final_cull",
            })
        );
    }

    #[test]
    fn combined_cpu_samples_are_explicitly_unavailable_as_a_split() {
        let mut collector =
            MeshletBenchmarkCollector::new(MeshletBenchmarkRenderer::Indexed, context(), true);
        for _ in 0..MESHLET_BENCHMARK_WARMUP_FRAMES {
            collector.record_frame(MeshletFrameSampleNs::combined(0, 0));
        }
        for value in 1..=MESHLET_BENCHMARK_MIN_SAMPLE_FRAMES as u64 {
            collector.record_frame(MeshletFrameSampleNs::combined(value, value * 2));
        }
        let report = collector.finish().unwrap();
        assert!(report.cpu_encode_ns.is_none());
        assert!(report.cpu_submit_ns.is_none());
        assert_eq!(report.cpu_frame_ns.median, 300);
    }

    #[test]
    fn report_and_qualifying_auto_profile_persist_as_json() {
        let directory = tempfile::tempdir().unwrap();
        let indexed_path = directory.path().join("indexed.json");
        let profile_path = directory.path().join("auto.json");
        let indexed = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        let task = report(MeshletBenchmarkRenderer::TaskMesh, 900);
        indexed.write_json_file(&indexed_path).unwrap();
        assert_eq!(
            MeshletBenchmarkReport::read_json_file(&indexed_path).unwrap(),
            indexed
        );

        let profile = MeshletAutoProfileFile::from_reports(&indexed, &task).unwrap();
        profile.write_json_file(&profile_path).unwrap();
        assert_eq!(
            MeshletAutoProfileFile::read_json_file(&profile_path).unwrap(),
            profile
        );
    }

    #[test]
    fn auto_profile_rejects_stale_adapter_driver_build_and_scene_identity() {
        let indexed = report(MeshletBenchmarkRenderer::Indexed, 1_000);
        let task = report(MeshletBenchmarkRenderer::TaskMesh, 900);
        let mut profile = MeshletAutoProfileFile::from_reports(&indexed, &task).unwrap();
        let mut adapter =
            wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan);
        adapter.name = profile.context.adapter.clone();
        adapter.vendor = profile.context.adapter_vendor;
        adapter.device = profile.context.adapter_device;
        adapter.driver = profile.context.driver.clone();
        adapter.driver_info = profile.context.driver_info.clone();
        profile.context.adapter_device_type = format!("{:?}", adapter.device_type);
        profile.context.renderer_build = benchmark_renderer_build_id();
        profile.context.camera_track = MESHLET_BENCHMARK_CAMERA_TRACK.into();
        assert!(
            profile
                .validate_for(&adapter, &profile.context.scene)
                .is_ok()
        );

        adapter.driver_info.push_str("-updated");
        assert_eq!(
            profile.validate_for(&adapter, &profile.context.scene),
            Err(MeshletBenchmarkError::ProfileIdentityMismatch {
                field: "driver_info"
            })
        );
    }
}

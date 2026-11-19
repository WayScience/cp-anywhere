//! Native throughput benchmark on the four representative workloads from
//! `spec/benchmark-contract.yaml`, timing `cp_anywhere::measure`
//! directly (no Python/FFI boundary) with the `object_intensity` +
//! `neighbors` families only, matching the exact scope the pinned
//! `cp_measure` baseline computed
//! (`benchmarks/baseline/initial-cp-measure-timing.json`) so the resulting
//! throughput ratio is a fair comparison. `scripts/benchmarks/report_native_rust.py`
//! parses Criterion's `estimates.json` output into the project's
//! benchmark-report shape.

use cp_anywhere::{measure, Families, Input};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::{Path, PathBuf};

#[path = "support/npy.rs"]
mod npy;

const WORKLOADS: &[&str] = &[
    "touching_gradient",
    "real_cell_cc0",
    "grid_medium",
    "grid_large",
];

fn fixture_dir(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../");
    let family = if name == "real_cell_cc0" {
        "real"
    } else {
        "synthetic"
    };
    root.join("fixtures").join(family).join(name)
}

fn bench_measure(c: &mut Criterion) {
    let mut group = c.benchmark_group("object_intensity_and_neighbors");
    for name in WORKLOADS {
        let dir = fixture_dir(name);
        let (shape, dna) = npy::read_f64(&dir.join("DNA.npy"));
        let (_, actin) = npy::read_f64(&dir.join("Actin.npy"));
        let (_, labels) = npy::read_u32(&dir.join("labels.npy"));
        let (_, nuclei) = npy::read_u32(&dir.join("nuclei_labels.npy"));
        let images: Vec<(&str, &[f64])> = vec![("DNA", &dna), ("Actin", &actin)];
        let objects: Vec<(&str, &[u32])> = vec![("Cells", &labels), ("Nuclei", &nuclei)];
        let input = Input {
            shape: &shape,
            images: &images,
            objects: &objects,
            mask: None,
            families: Families {
                image_intensity: false,
                object_intensity: true,
                neighbors: true,
            },
            neighbors_scopes: &[],
            neighbors_expand: &[],
            size_shape: None,
            neighbors_full_objects: &[],
        };
        group.bench_with_input(BenchmarkId::from_parameter(name), &input, |b, input| {
            b.iter(|| measure(input).unwrap());
        });
    }
    group.finish();

    // Split by family to localize any scaling bottleneck precisely, rather
    // than guess from the combined time (plan 5.4: "profile first").
    let mut split = c.benchmark_group("by_family");
    for name in WORKLOADS {
        let dir = fixture_dir(name);
        let (shape, dna) = npy::read_f64(&dir.join("DNA.npy"));
        let (_, actin) = npy::read_f64(&dir.join("Actin.npy"));
        let (_, labels) = npy::read_u32(&dir.join("labels.npy"));
        let (_, nuclei) = npy::read_u32(&dir.join("nuclei_labels.npy"));
        let images: Vec<(&str, &[f64])> = vec![("DNA", &dna), ("Actin", &actin)];
        let objects: Vec<(&str, &[u32])> = vec![("Cells", &labels), ("Nuclei", &nuclei)];
        let intensity_only = Input {
            shape: &shape,
            images: &images,
            objects: &objects,
            mask: None,
            families: Families {
                image_intensity: false,
                object_intensity: true,
                neighbors: false,
            },
            neighbors_scopes: &[],
            neighbors_expand: &[],
            size_shape: None,
            neighbors_full_objects: &[],
        };
        split.bench_with_input(
            BenchmarkId::new("object_intensity", name),
            &intensity_only,
            |b, input| b.iter(|| measure(input).unwrap()),
        );
        let neighbors_only = Input {
            shape: &shape,
            images: &images,
            objects: &objects,
            mask: None,
            families: Families {
                image_intensity: false,
                object_intensity: false,
                neighbors: true,
            },
            neighbors_scopes: &[],
            neighbors_expand: &[],
            size_shape: None,
            neighbors_full_objects: &[],
        };
        split.bench_with_input(
            BenchmarkId::new("neighbors", name),
            &neighbors_only,
            |b, input| b.iter(|| measure(input).unwrap()),
        );
    }
    split.finish();
}

criterion_group!(benches, bench_measure);
criterion_main!(benches);

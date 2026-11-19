//! CellProfiler 4.2.8 `MeasureObjectIntensityDistribution`, oracle-verified
//! against a pinned reference CellProfiler/centrosome install (centrosome
//! 1.2.3, scipy 1.9.3, numpy 1.23.5).
//!
//! This is a from-scratch Rust port of
//! `measureobjectintensitydistribution.py`'s `do_measurements` and
//! `calculate_zernikes`, restricted to the module's DEFAULT center choice
//! ("These objects", i.e. `center_object_name is None` in the real source --
//! the radial center is each object's own point farthest from its edge).
//! The two other center choices ("Centers of other objects" and "Edges of
//! other objects") require a second, separately-selected object set and are
//! an explicit, documented gap -- see [`measure`]'s doc comment -- consistent
//! with this project's established house style for out-of-scope real-module
//! behavior (see `spec/cppipe-compatibility.md`).
//!
//! ### Algorithm notes (traced from `centrosome`, not guessed)
//!
//! * `centrosome.cpmorphology.distance_to_edge` is `scipy.ndimage.
//!   distance_transform_edt` restricted to `centrosome.cpmorphology.
//!   color_labels` groups (so that non-touching objects can share one EDT
//!   call without leaking distances across objects). This port computes the
//!   exact Euclidean distance transform **per object** (`mask = labels ==
//!   this_id` only) instead of per color group. The two are mathematically
//!   identical for every object whose own complement already contains a true
//!   background pixel nearer than any other same-colored object -- i.e.
//!   every case except one object being fully encircled by another,
//!   same-colored object at 1-pixel separation. That pathological nesting is
//!   not reproduced here and is a known, explicit gap (see
//!   `DistanceToEdgeMethod` note below); ordinary touching, border, and
//!   concave/non-convex single-object shapes are unaffected and are exactly
//!   correct.
//! * The per-object radial center is `centrosome.cpmorphology.
//!   maximum_position_of_labels(distance_to_edge(labels), labels, indices)`,
//!   i.e. `scipy.ndimage.maximum_position`. Reading `scipy/ndimage/
//!   _measurements.py::_select` directly (not guessed): ties are **not**
//!   broken by row-major scan order. The real implementation does one
//!   ascending `argsort` of the *entire* image's distance-to-edge values
//!   (NumPy's own quicksort, ties included), then walks that order
//!   assigning each label's farthest-point position, so later (larger, or
//!   tied-and-later-in-sort-order) values overwrite earlier ones -- "last
//!   assignment wins". This port reproduces that exactly (reusing this
//!   crate's own `numpy_argsort`, already oracle-verified for sort-order
//!   fidelity elsewhere) because symmetric or non-convex shapes (rings,
//!   concave arcs) routinely have many pixels tied for farthest-from-edge,
//!   and a naive row-major-first tie-break measurably picks the wrong
//!   center pixel for them.
//! * `d_from_center` is `centrosome.propagate.propagate(zeros_image, seed,
//!   mask, weight=1)`'s distance output. Reading `centrosome/_propagate.pyx`
//!   directly (not guessed): with a flat all-zero height image, each
//!   `propagate` graph edge costs `sqrt(manhattan_distance)` where
//!   `manhattan_distance` is 1 for an orthogonal 8-connected step and 2 for a
//!   diagonal one -- i.e. plain Dijkstra over an 8-connected grid with edge
//!   weights `1` (orthogonal) and `sqrt(2)` (diagonal), confined to the
//!   object's own mask. Pixels the object's own connected component can't
//!   reach (a genuinely disconnected object piece) are excluded from every
//!   measurement, exactly like real CellProfiler's `good_mask`.
//! * Zernike magnitudes/phases reuse the same radial-polynomial and
//!   complex-power recursion already oracle-verified in
//!   `size_shape_moments.rs`'s `AreaShape_Zernike_*` port (confirmed
//!   algebraically identical against `centrosome.zernike.
//!   construct_zernike_polynomials`'s Horner-scheme lookup table and `z**m`
//!   complex power). It cannot be imported from that sibling module without
//!   modifying it, so the shared pieces (`Point`, convex hull, minimum
//!   enclosing circle, factorial, radial polynomial) are duplicated here;
//!   see the license notices below, carried over for the same reason they
//!   are present there. Unlike `AreaShape_Zernike_*` (which divides by the
//!   enclosing circle's analytic area `radius^2 * pi`), this module divides
//!   by the real pixel count of the object (`vr, vi` are summed per-pixel
//!   intensities dotted with the Zernike basis, then divided by `areas =
//!   scipy.ndimage.sum(ones, labels=l_, index=...)`), and the phase is
//!   `arctan2(vr, vi)` (arguments in that order, confirmed from source).

/*
centrosome: Copyright © 2003 - 2019 Broad Institute, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the Broad Institute, Inc. nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED “AS IS.” BROAD MAKES NO EXPRESS OR IMPLIED
REPRESENTATIONS OR WARRANTIES OF ANY KIND REGARDING THE SOFTWARE AND
COPYRIGHT, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, CONFORMITY WITH ANY DOCUMENTATION,
NON-INFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER OR NOT
DISCOVERABLE. IN NO EVENT SHALL BROAD, THE COPYRIGHT HOLDERS, OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF, HAVE REASON TO KNOW, OR IN
FACT SHALL KNOW OF THE POSSIBILITY OF SUCH DAMAGE.
*/

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

/// Which Zernike columns (if any) to compute, mirroring the real module's
/// "Calculate intensity Zernikes?" choice and "Maximum zernike moment".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZernikeOption {
    /// `Z_NONE`.
    None,
    /// `Z_MAGNITUDES`: `ZernikeMagnitude` columns only, up to and including
    /// radial degree `degree`.
    Magnitudes { degree: u32 },
    /// `Z_MAGNITUDES_AND_PHASE`: both `ZernikeMagnitude` and `ZernikePhase`
    /// columns, up to and including radial degree `degree`.
    MagnitudesAndPhase { degree: u32 },
}

/// Storage precision of the prepared CellProfiler image, before widening for
/// transport into the core. This controls radial intensity arithmetic only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelPrecision {
    Float32,
    Float64,
}

/// Settings mirroring one `images_list` x one object group x one
/// `bin_counts` group of `MeasureObjectIntensityDistribution`, for the
/// default ("These objects") center choice.
#[derive(Clone, Debug)]
pub struct Config {
    /// The image name baked into every output column
    /// (`RadialDistribution_<feature>_<image_name>_...`), matching the real
    /// module's `FF_GENERIC` column-naming convention.
    pub image_name: String,
    /// "Number of bins": ring count `N`.
    pub bin_count: usize,
    /// "Scale the bins?". `true` normalizes each object's own
    /// center-to-edge span to `[0, 1)`; `false` uses `maximum_radius`
    /// pixels per bin and reports an `Overflow` bin beyond it.
    pub wants_scaled: bool,
    /// "Maximum radius", in pixels. Only used when `wants_scaled` is false.
    pub maximum_radius: f64,
    pub zernikes: ZernikeOption,
}

/// One real `RadialDistribution_*` column per entry, keyed by its exact
/// CellProfiler feature name (e.g. `"RadialDistribution_FracAtD_DNA_1of4"`,
/// `"RadialDistribution_ZernikeMagnitude_DNA_3_1"`). Each `Vec<f64>` holds
/// one value per object, ordered by ascending 1-based object id; an object
/// id absent from `labels` (a "gap" in an otherwise-consecutive label range)
/// still gets a slot, filled by whatever the real module's own division-by-
/// zero arithmetic produces for a zero-pixel object (see [`measure`]'s doc
/// comment). A label image with no objects at all reports every selected
/// column as an empty `Vec`, matching `numpy.zeros(0)`.
#[derive(Debug, Default)]
pub struct IntensityDistributionResult {
    pub columns: BTreeMap<String, Vec<f64>>,
}

const EPS: f64 = f64::EPSILON;
/// Stands in for the real algorithm's effective infinity (`scipy.ndimage.
/// distance_transform_edt` on an all-foreground column/row) without risking
/// `inf - inf = NaN` in the Felzenszwalt-Huttenlocher envelope arithmetic.
/// Any test fixture has real background somewhere reachable, so this
/// sentinel is never the chosen minimum; it only has to be larger than any
/// genuine in-image distance.
const UNREACHABLE: f64 = 1e18;

pub fn measure(
    pixels: &[f64],
    labels: &[u32],
    rows: usize,
    cols: usize,
    config: &Config,
) -> Result<IntensityDistributionResult, String> {
    measure_with_overflow(pixels, labels, rows, cols, config, !config.wants_scaled)
}

/// Measure one group with overflow columns controlled separately from distance
/// normalization, as in CellProfiler's multi-group pipeline module.
pub fn measure_with_overflow(
    pixels: &[f64],
    labels: &[u32],
    rows: usize,
    cols: usize,
    config: &Config,
    emit_overflow: bool,
) -> Result<IntensityDistributionResult, String> {
    measure_with_precision(
        pixels,
        labels,
        rows,
        cols,
        config,
        emit_overflow,
        PixelPrecision::Float64,
    )
}

/// Measure using the source image's storage precision for CellProfiler's
/// dtype-sensitive radial operations. The pixel buffer may be widened to f64
/// for transport; callers must supply the precision before widening.
pub fn measure_with_precision(
    pixels: &[f64],
    labels: &[u32],
    rows: usize,
    cols: usize,
    config: &Config,
    emit_overflow: bool,
    precision: PixelPrecision,
) -> Result<IntensityDistributionResult, String> {
    if rows == 0 || cols == 0 {
        return Err("intensity distribution requires two nonzero dimensions".into());
    }
    let size = rows.checked_mul(cols).ok_or("shape product overflow")?;
    if pixels.len() != size || labels.len() != size {
        return Err("pixels/labels length does not match rows*cols".into());
    }
    if config.bin_count == 0 {
        return Err("bin_count must be at least 1".into());
    }
    if !config.wants_scaled && (config.maximum_radius.is_nan() || config.maximum_radius <= 0.0) {
        return Err("maximum_radius must be positive when wants_scaled is false".into());
    }
    let degree = match config.zernikes {
        ZernikeOption::None => None,
        ZernikeOption::Magnitudes { degree } | ZernikeOption::MagnitudesAndPhase { degree } => {
            if degree == 0 {
                return Err("zernike degree must be at least 1".into());
            }
            Some(degree)
        }
    };
    for &id in labels {
        if id > crate::MAX_LABEL {
            return Err(format!("label exceeds {}", crate::MAX_LABEL));
        }
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut out = BTreeMap::new();

    if max_label == 0 {
        for name in radial_column_names(config, emit_overflow) {
            out.insert(name, Vec::new());
        }
        if let Some(degree) = degree {
            for (name, _) in zernike_column_names(config, degree) {
                out.insert(name, Vec::new());
            }
        }
        return Ok(IntensityDistributionResult { columns: out });
    }

    let n = max_label;
    let nbins = config.bin_count + 1; // regular bins + overflow slot
    let mut hist_sum = vec![0.0_f64; n * nbins];
    let mut hist_count = vec![0.0_f64; n * nbins];
    let mut wedge_sum = vec![0.0_f64; n * nbins * 8];
    let mut wedge_count = vec![0.0_f64; n * nbins * 8];

    // Pass 1: distance-to-edge for every object, merged into one image-wide
    // buffer (objects never overlap, so this is a plain union of the
    // per-object EDTs -- see the module doc comment for the one documented
    // divergence from real CP's color-grouped version).
    let mut dist_to_edge_full = vec![0.0_f64; size];
    let mut object_masks: Vec<Vec<bool>> = vec![Vec::new(); n];
    for label in 1..=n as u32 {
        let mask: Vec<bool> = labels.iter().map(|&id| id == label).collect();
        if !mask.iter().any(|&m| m) {
            continue; // a gap id: zero pixels, every column stays at its 0 default
        }
        let dist = distance_to_edge(&mask, rows, cols);
        for idx in 0..size {
            if mask[idx] {
                dist_to_edge_full[idx] = dist[idx];
            }
        }
        object_masks[label as usize - 1] = mask;
    }

    // Pass 2: seed (center) position per object, replicating `scipy.ndimage.
    // maximum_position`'s own tie-break exactly: a single ascending argsort
    // of the WHOLE image's distance-to-edge values (ties resolved by
    // NumPy's own quicksort, not row-major scan order -- confirmed by
    // reading `scipy/ndimage/_measurements.py::_select`), then "last
    // assignment wins" per label walking that order. A symmetric or
    // non-convex object (e.g. a ring) commonly has many tied maxima, so
    // this exact tie-break genuinely changes which pixel is the center.
    let order = crate::numpy_argsort(&dist_to_edge_full);
    let mut seed_of: Vec<Option<usize>> = vec![None; n];
    for &idx in &order {
        let lbl = labels[idx];
        if lbl > 0 {
            seed_of[lbl as usize - 1] = Some(idx);
        }
    }

    // Pass 3: geodesic propagation and binning, per object.
    for label in 1..=n as u32 {
        let obj = label as usize - 1;
        let mask = &object_masks[obj];
        if mask.is_empty() {
            continue; // a gap id: zero pixels
        }
        let Some(seed_idx) = seed_of[obj] else {
            continue;
        };
        let (seed_r, seed_c) = (seed_idx / cols, seed_idx % cols);
        let (geo_dist, reached) = geodesic_distances(mask, rows, cols, seed_r, seed_c);

        for idx in 0..size {
            if !reached[idx] {
                continue;
            }
            let d_from_center = geo_dist[idx];
            let d_to_edge_px = dist_to_edge_full[idx];
            let normalized = if config.wants_scaled {
                d_from_center / (d_from_center + d_to_edge_px + 0.001)
            } else {
                d_from_center / config.maximum_radius
            };
            let mut bin = (normalized * config.bin_count as f64).floor() as isize;
            if bin < 0 {
                bin = 0;
            }
            let bin = (bin as usize).min(config.bin_count);

            let r = idx / cols;
            let c = idx % cols;
            let imask = r > seed_r;
            let jmask = c > seed_c;
            let absmask =
                (r as isize - seed_r as isize).abs() > (c as isize - seed_c as isize).abs();
            let octant = imask as usize + (jmask as usize) * 2 + (absmask as usize) * 4;

            let value = pixels[idx];
            let ring = obj * nbins + bin;
            let wedge = ring * 8 + octant;
            // SciPy densifies each coordinate-format sparse histogram in pixel order,
            // in the input dtype. Counts and downstream CV are f64.
            if precision == PixelPrecision::Float32 {
                hist_sum[ring] = (hist_sum[ring] as f32 + value as f32) as f64;
                wedge_sum[wedge] = (wedge_sum[wedge] as f32 + value as f32) as f64;
            } else {
                hist_sum[ring] += value;
                wedge_sum[wedge] += value;
            }
            hist_count[ring] += 1.0;
            wedge_count[wedge] += 1.0;
        }
    }

    let mut sum_by_object = vec![0.0_f64; n];
    let mut count_by_object = vec![0.0_f64; n];
    for obj in 0..n {
        let row = &hist_sum[obj * nbins..(obj + 1) * nbins];
        sum_by_object[obj] = match precision {
            PixelPrecision::Float32 => numpy_sum_f32(row) as f64,
            PixelPrecision::Float64 => row.iter().sum(),
        };
        for bin in 0..nbins {
            count_by_object[obj] += hist_count[obj * nbins + bin];
        }
    }

    let mut frac_at_d = vec![0.0_f64; n * nbins];
    let mut mean_frac = vec![0.0_f64; n * nbins];
    let mut radial_cv = vec![0.0_f64; n * nbins];
    for obj in 0..n {
        for bin in 0..nbins {
            // Unlike MeanFrac and CV, this division has two float32 operands
            // for a float32 image, so CellProfiler rounds the quotient too.
            let sum = hist_sum[obj * nbins + bin];
            let fd = match precision {
                PixelPrecision::Float32 => (sum as f32 / sum_by_object[obj] as f32) as f64,
                PixelPrecision::Float64 => sum / sum_by_object[obj],
            };
            let fb = hist_count[obj * nbins + bin] / count_by_object[obj];
            frac_at_d[obj * nbins + bin] = fd;
            mean_frac[obj * nbins + bin] = fd / (fb + EPS);

            let mut populated = 0usize;
            let mut means = [0.0_f64; 8];
            for octant in 0..8 {
                let count = wedge_count[(obj * nbins + bin) * 8 + octant];
                if count > 0.0 {
                    means[populated] = wedge_sum[(obj * nbins + bin) * 8 + octant] / count;
                    populated += 1;
                }
            }
            radial_cv[obj * nbins + bin] = if populated == 0 {
                0.0
            } else {
                let mean: f64 = means[..populated].iter().sum::<f64>() / populated as f64;
                let var: f64 = means[..populated]
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>()
                    / populated as f64;
                var.sqrt() / mean
            };
        }
    }

    for bin in 1..=config.bin_count {
        out.insert(
            scaled_name(config, "FracAtD", bin),
            column(&frac_at_d, n, nbins, bin - 1),
        );
        out.insert(
            scaled_name(config, "MeanFrac", bin),
            column(&mean_frac, n, nbins, bin - 1),
        );
        out.insert(
            scaled_name(config, "RadialCV", bin),
            column(&radial_cv, n, nbins, bin - 1),
        );
    }
    if emit_overflow {
        out.insert(
            overflow_name(config, "FracAtD"),
            column(&frac_at_d, n, nbins, config.bin_count),
        );
        out.insert(
            overflow_name(config, "MeanFrac"),
            column(&mean_frac, n, nbins, config.bin_count),
        );
        out.insert(
            overflow_name(config, "RadialCV"),
            column(&radial_cv, n, nbins, config.bin_count),
        );
    }

    if let Some(degree) = degree {
        compute_zernikes(&mut out, pixels, labels, rows, cols, n, config, degree);
    }

    Ok(IntensityDistributionResult { columns: out })
}

// Derived from NumPy 1.23.5 FLOAT_pairwise_sum for a contiguous histogram
// row (numpy/core/src/umath/loops_utils.h.src); see licenses/numpy.txt.
fn numpy_sum_f32(row: &[f64]) -> f32 {
    const BLOCK_SIZE: usize = 128;
    match row.len() {
        0..=7 => row.iter().fold(-0.0, |sum, &v| sum + v as f32),
        8..=BLOCK_SIZE => {
            let mut lanes = std::array::from_fn::<_, 8, _>(|i| row[i] as f32);
            let mut chunks = row[8..].chunks_exact(8);
            for chunk in &mut chunks {
                for (lane, &v) in lanes.iter_mut().zip(chunk) {
                    *lane += v as f32;
                }
            }
            let [a, b, c, d, e, f, g, h] = lanes;
            chunks
                .remainder()
                .iter()
                .fold(((a + b) + (c + d)) + ((e + f) + (g + h)), |sum, &v| {
                    sum + v as f32
                })
        }
        len => {
            let (left, right) = row.split_at((len / 2) & !7);
            numpy_sum_f32(left) + numpy_sum_f32(right)
        }
    }
}

fn column(values: &[f64], n: usize, nbins: usize, bin: usize) -> Vec<f64> {
    (0..n).map(|obj| values[obj * nbins + bin]).collect()
}

fn scaled_name(config: &Config, feature: &str, bin: usize) -> String {
    format!(
        "RadialDistribution_{feature}_{}_{bin}of{}",
        config.image_name, config.bin_count
    )
}

fn overflow_name(config: &Config, feature: &str) -> String {
    format!(
        "RadialDistribution_{feature}_{}_Overflow",
        config.image_name
    )
}

fn radial_column_names(config: &Config, emit_overflow: bool) -> Vec<String> {
    let mut names = Vec::new();
    for feature in ["FracAtD", "MeanFrac", "RadialCV"] {
        for bin in 1..=config.bin_count {
            names.push(scaled_name(config, feature, bin));
        }
        if emit_overflow {
            names.push(overflow_name(config, feature));
        }
    }
    names
}

fn zernike_column_names(config: &Config, degree: u32) -> Vec<(String, (u32, u32))> {
    let mut names = Vec::new();
    let wants_phase = matches!(config.zernikes, ZernikeOption::MagnitudesAndPhase { .. });
    for n in 0..=degree {
        for m in (n % 2..=n).step_by(2) {
            names.push((
                format!(
                    "RadialDistribution_ZernikeMagnitude_{}_{n}_{m}",
                    config.image_name
                ),
                (n, m),
            ));
            if wants_phase {
                names.push((
                    format!(
                        "RadialDistribution_ZernikePhase_{}_{n}_{m}",
                        config.image_name
                    ),
                    (n, m),
                ));
            }
        }
    }
    names
}

// ---------------------------------------------------------------------
// Distance to edge: exact Euclidean distance transform (Felzenszwalt &
// Huttenlocher), per object mask -- see the module doc comment for the
// documented divergence from real CP's color-grouped version.
// ---------------------------------------------------------------------

fn distance_to_edge(mask: &[bool], rows: usize, cols: usize) -> Vec<f64> {
    edt_sq(mask, rows, cols)
        .into_iter()
        .map(f64::sqrt)
        .collect()
}

fn edt_sq(mask: &[bool], rows: usize, cols: usize) -> Vec<f64> {
    let mut g = vec![0.0_f64; rows * cols];
    let mut col_buf = vec![0.0_f64; rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = if mask[r * cols + c] { UNREACHABLE } else { 0.0 };
        }
        let dt = dt1d(&col_buf);
        for r in 0..rows {
            g[r * cols + c] = dt[r];
        }
    }
    let mut out = vec![0.0_f64; rows * cols];
    let mut row_buf = vec![0.0_f64; cols];
    for r in 0..rows {
        row_buf.copy_from_slice(&g[r * cols..r * cols + cols]);
        let dt = dt1d(&row_buf);
        out[r * cols..r * cols + cols].copy_from_slice(&dt);
    }
    out
}

/// 1D lower-envelope-of-parabolas exact squared distance transform
/// (Felzenszwalt & Huttenlocher, 2004), `O(n)`.
fn dt1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut d = vec![0.0_f64; n];
    if n == 0 {
        return d;
    }
    let mut v = vec![0usize; n];
    let mut z = vec![0.0_f64; n + 1];
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s;
        loop {
            let vk = v[k];
            s = ((f[q] + (q * q) as f64) - (f[vk] + (vk * vk) as f64))
                / (2.0 * (q as f64 - vk as f64));
            if s <= z[k] {
                if k == 0 {
                    break;
                }
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, slot) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let vk = v[k];
        *slot = (q as f64 - vk as f64).powi(2) + f[vk];
    }
    d
}

// ---------------------------------------------------------------------
// Geodesic propagation: Dijkstra, 8-connected, weights {1, sqrt(2)},
// confined to the object's own mask. See module doc comment for the
// `centrosome/_propagate.pyx` trace that derived these exact weights.
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
struct HeapKey(f64);
impl Eq for HeapKey {}
impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for HeapKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn geodesic_distances(
    mask: &[bool],
    rows: usize,
    cols: usize,
    seed_r: usize,
    seed_c: usize,
) -> (Vec<f64>, Vec<bool>) {
    let size = rows * cols;
    let mut dist = vec![f64::INFINITY; size];
    let mut settled = vec![false; size];
    let seed_idx = seed_r * cols + seed_c;
    dist[seed_idx] = 0.0;
    let mut heap: BinaryHeap<Reverse<(HeapKey, usize)>> = BinaryHeap::new();
    heap.push(Reverse((HeapKey(0.0), seed_idx)));
    const DELTAS: [(isize, isize); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    while let Some(Reverse((HeapKey(d), idx))) = heap.pop() {
        if settled[idx] {
            continue;
        }
        settled[idx] = true;
        let r = idx / cols;
        let c = idx % cols;
        for (dr, dc) in DELTAS {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            let nidx = nr * cols + nc;
            if settled[nidx] || !mask[nidx] {
                continue;
            }
            let step = if dr != 0 && dc != 0 {
                std::f64::consts::SQRT_2
            } else {
                1.0
            };
            let candidate_dist = d + step;
            if candidate_dist < dist[nidx] {
                dist[nidx] = candidate_dist;
                heap.push(Reverse((HeapKey(candidate_dist), nidx)));
            }
        }
    }
    (dist, settled)
}

// ---------------------------------------------------------------------
// Zernikes: minimum enclosing circle (Chrystal's algorithm, duplicated from
// size_shape_moments.rs -- see module doc comment) plus the per-pixel
// complex Zernike basis from centrosome.zernike.construct_zernike_polynomials.
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Point {
    row: f64,
    col: f64,
}

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b.col - a.col) * (c.row - b.row) - (c.col - b.col) * (b.row - a.row)
}

fn convex(a: Point, b: Point, c: Point) -> bool {
    let turn = cross(a, b, c);
    turn > 0.0 || (turn == 0.0 && b.col > a.col && b.col > c.col)
}

// Column envelopes, starting vertex, U-turn handling and ordering match
// centrosome._convex_hull, not a hull of pixel *corners*. Duplicated from
// size_shape_moments.rs::convex_hull (see module doc comment).
fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut envelopes: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for p in points {
        let key = p.col as i64;
        let bounds = envelopes.entry(key).or_insert((p.row, p.row));
        bounds.0 = bounds.0.min(p.row);
        bounds.1 = bounds.1.max(p.row);
    }
    let (&start_col, &(start_min, start_max)) = envelopes.first_key_value().unwrap();
    let mut hull: Vec<Point> = Vec::new();
    let emit = |hull: &mut Vec<Point>, p: Point| {
        while hull.len() >= 2 && !convex(hull[hull.len() - 2], hull[hull.len() - 1], p) {
            hull.pop();
        }
        hull.push(p);
    };
    for (&col, &(row, _)) in &envelopes {
        emit(
            &mut hull,
            Point {
                row,
                col: col as f64,
            },
        );
    }
    for (&col, &(_, row)) in envelopes.iter().rev() {
        if col != start_col {
            emit(
                &mut hull,
                Point {
                    row,
                    col: col as f64,
                },
            );
        }
    }
    let last = Point {
        row: start_max,
        col: start_col as f64,
    };
    while hull.len() >= 2 && !convex(hull[hull.len() - 2], hull[hull.len() - 1], last) {
        hull.pop();
    }
    if start_max != start_min {
        hull.push(last);
    }
    hull
}

fn distance_squared(a: Point, b: Point) -> f64 {
    (a.row - b.row) * (a.row - b.row) + (a.col - b.col) * (a.col - b.col)
}

fn diameter(a: Point, b: Point) -> (Point, f64) {
    (
        Point {
            row: (a.row + b.row) / 2.0,
            col: (a.col + b.col) / 2.0,
        },
        distance_squared(a, b).sqrt() / 2.0,
    )
}

// Chrystal's hull algorithm. Duplicated from
// size_shape_moments.rs::enclosing_circle (see module doc comment).
fn enclosing_circle(hull: &[Point]) -> (Point, f64) {
    if hull.len() == 1 {
        return (hull[0], 0.0);
    }
    if hull.len() == 2 {
        return diameter(hull[0], hull[1]);
    }
    let (mut s0, mut s1) = (0, 1);
    let half_pi = std::f64::consts::FRAC_PI_2;
    loop {
        let (a, b) = (hull[s0], hull[s1]);
        let ab2 = distance_squared(a, b);
        let mut smallest = (f64::INFINITY, 0, 0.0, 0.0);
        for (i, &v) in hull.iter().enumerate() {
            if i == s0 || i == s1 {
                continue;
            }
            let at_b = (((a.row - b.row) * (v.row - b.row) + (a.col - b.col) * (v.col - b.col))
                / (ab2 * distance_squared(v, b)).sqrt())
            .acos()
            .abs();
            let at_a = (((b.row - a.row) * (v.row - a.row) + (b.col - a.col) * (v.col - a.col))
                / (ab2 * distance_squared(v, a)).sqrt())
            .acos()
            .abs();
            let at_v = std::f64::consts::PI - at_b - at_a;
            if at_v < smallest.0 {
                smallest = (at_v, i, at_a, at_b);
            }
        }
        let (angle, vertex, at_a, at_b) = smallest;
        if angle >= half_pi {
            return diameter(a, b);
        }
        let v = hull[vertex];
        if at_a <= half_pi && at_b <= half_pi && angle <= half_pi {
            let d =
                2.0 * (a.col * (b.row - v.row) + b.col * (v.row - a.row) + v.col * (a.row - b.row));
            let aa = a.row * a.row + a.col * a.col;
            let bb = b.row * b.row + b.col * b.col;
            let vv = v.row * v.row + v.col * v.col;
            let center = Point {
                col: (aa * (b.row - v.row) + bb * (v.row - a.row) + vv * (a.row - b.row)) / d,
                row: (aa * (v.col - b.col) + bb * (a.col - v.col) + vv * (b.col - a.col)) / d,
            };
            return (center, distance_squared(a, center).sqrt());
        }
        if at_a > half_pi {
            s0 = vertex;
        } else {
            s1 = vertex;
        }
    }
}

fn factorial(n: u32) -> f64 {
    (1..=n).fold(1.0, |acc, value| acc * f64::from(value))
}

/// One pixel's complex Zernike `(n, m)` basis value: radial polynomial
/// (Horner scheme, `centrosome.zernike.construct_zernike_lookuptable`'s
/// coefficients) times `(y + i*x)^m`, masked to zero outside the unit
/// circle. `x, y` are already normalized by the enclosing-circle radius.
fn zernike_basis(x: f64, y: f64, n: u32, m: u32) -> (f64, f64) {
    let r2 = x * x + y * y;
    let mut radial = 0.0_f64;
    for k in 0..=(n - m) / 2 {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let coefficient = sign * factorial(n - k)
            / (factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k));
        radial = radial * r2 + coefficient;
    }
    if r2 > 1.0 {
        radial = 0.0;
    }
    let (mut re, mut im) = (1.0_f64, 0.0_f64);
    for _ in 0..m {
        (re, im) = (re * y - im * x, re * x + im * y);
    }
    (radial * re, radial * im)
}

#[allow(clippy::too_many_arguments)]
fn compute_zernikes(
    out: &mut BTreeMap<String, Vec<f64>>,
    pixels: &[f64],
    labels: &[u32],
    rows: usize,
    cols: usize,
    n: usize,
    config: &Config,
    degree: u32,
) {
    let names = zernike_column_names(config, degree);
    let mut magnitude: BTreeMap<(u32, u32), Vec<f64>> = BTreeMap::new();
    let mut phase: BTreeMap<(u32, u32), Vec<f64>> = BTreeMap::new();
    for &(_, (nn, m)) in &names {
        magnitude.entry((nn, m)).or_insert_with(|| vec![0.0; n]);
        phase.entry((nn, m)).or_insert_with(|| vec![0.0; n]);
    }

    let mut points_per_label: Vec<Vec<(Point, f64)>> = vec![Vec::new(); n];
    for idx in 0..rows * cols {
        let id = labels[idx];
        if id == 0 {
            continue;
        }
        let row = (idx / cols) as f64;
        let col = (idx % cols) as f64;
        points_per_label[id as usize - 1].push((Point { row, col }, pixels[idx]));
    }

    for (obj, points) in points_per_label.iter().enumerate() {
        if points.is_empty() {
            // Absent from CellProfiler's `ijv`: scipy.ndimage.sum over an
            // empty selection is 0 for every (n, m), so vr = vi = areas = 0.
            // magnitude = sqrt(0)/0 = NaN; phase = arctan2(0, 0) = 0.
            for (&key, mag) in magnitude.iter_mut() {
                mag[obj] = f64::NAN;
                if let Some(ph) = phase.get_mut(&key) {
                    ph[obj] = 0.0;
                }
            }
            continue;
        }
        let hull = convex_hull(&points.iter().map(|(p, _)| *p).collect::<Vec<_>>());
        let (center, radius) = enclosing_circle(&hull);
        let area = points.len() as f64;
        for (&(nn, m), mag) in magnitude.iter_mut() {
            let mut vr = 0.0_f64;
            let mut vi = 0.0_f64;
            for &(p, value) in points {
                let x = (p.col - center.col) / radius;
                let y = (p.row - center.row) / radius;
                let (zr, zi) = zernike_basis(x, y, nn, m);
                vr += value * zr;
                vi += value * zi;
            }
            mag[obj] = (vr * vr + vi * vi).sqrt() / area;
            if let Some(ph) = phase.get_mut(&(nn, m)) {
                ph[obj] = vr.atan2(vi);
            }
        }
    }

    let wants_phase = matches!(config.zernikes, ZernikeOption::MagnitudesAndPhase { .. });
    for (name, (nn, m)) in names {
        if name.contains("ZernikeMagnitude") {
            out.insert(name, magnitude[&(nn, m)].clone());
        } else if wants_phase {
            out.insert(name, phase[&(nn, m)].clone());
        }
    }
}

#[cfg(test)]
#[allow(clippy::useless_vec, clippy::approx_constant, clippy::needless_borrow)]
mod tests {
    use super::*;

    fn assert_values(actual: &[f64], expected: &[f64], label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch (actual {}, expected {})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            let tol = 1e-9 + 1e-6 * e.abs();
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: actual {a:?} vs expected {e:?} (tolerance {tol:?})"
            );
        }
    }

    #[test]
    fn rejects_bad_shape() {
        let pixels = [1.0];
        let labels: [u32; 1] = [1];
        let config = Config {
            image_name: "DNA".into(),
            bin_count: 4,
            wants_scaled: true,
            maximum_radius: 100.0,
            zernikes: ZernikeOption::None,
        };
        assert!(measure(&pixels, &labels, 0, 1, &config).is_err());
        assert!(measure(&pixels[..0], &labels[..0], 1, 1, &config).is_err());
    }

    #[test]
    fn rejects_bad_config() {
        let pixels = [1.0, 1.0];
        let labels: [u32; 2] = [1, 1];
        let mut config = Config {
            image_name: "DNA".into(),
            bin_count: 0,
            wants_scaled: true,
            maximum_radius: 100.0,
            zernikes: ZernikeOption::None,
        };
        assert!(measure(&pixels, &labels, 1, 2, &config).is_err());
        config.bin_count = 4;
        config.wants_scaled = false;
        config.maximum_radius = 0.0;
        assert!(measure(&pixels, &labels, 1, 2, &config).is_err());
        config.maximum_radius = 10.0;
        config.zernikes = ZernikeOption::Magnitudes { degree: 0 };
        assert!(measure(&pixels, &labels, 1, 2, &config).is_err());
    }

    #[test]
    fn no_objects_reports_every_selected_column_as_empty() {
        let pixels = [0.0; 6];
        let labels: [u32; 6] = [0; 6];
        let config = Config {
            image_name: "DNA".into(),
            bin_count: 3,
            wants_scaled: true,
            maximum_radius: 10.0,
            zernikes: ZernikeOption::Magnitudes { degree: 2 },
        };
        let result = measure(&pixels, &labels, 2, 3, &config).unwrap();
        assert_eq!(
            result.columns["RadialDistribution_FracAtD_DNA_1of3"],
            Vec::<f64>::new()
        );
        assert_eq!(
            result.columns["RadialDistribution_ZernikeMagnitude_DNA_2_0"],
            Vec::<f64>::new()
        );
        assert!(!result
            .columns
            .contains_key("RadialDistribution_FracAtD_DNA_Overflow"));
    }

    #[test]
    fn unscaled_adds_overflow_columns() {
        let pixels = [1.0; 9];
        let labels: [u32; 9] = [1; 9];
        let config = Config {
            image_name: "DNA".into(),
            bin_count: 2,
            wants_scaled: false,
            maximum_radius: 1.0,
            zernikes: ZernikeOption::None,
        };
        let result = measure(&pixels, &labels, 3, 3, &config).unwrap();
        assert!(result
            .columns
            .contains_key("RadialDistribution_FracAtD_DNA_Overflow"));
        assert!(result
            .columns
            .contains_key("RadialDistribution_MeanFrac_DNA_Overflow"));
        assert!(result
            .columns
            .contains_key("RadialDistribution_RadialCV_DNA_Overflow"));
    }

    #[test]
    fn a_disconnected_label_fragment_is_excluded_like_real_good_mask() {
        // Two 1-pixel-wide arms of the same label joined only diagonally
        // (not 8-connected through the mask after the corner pixel is
        // background) -- the unreachable arm should not corrupt the
        // reachable arm's statistics; the whole object keeps finite output.
        let mut labels = vec![0u32; 5 * 5];
        for (r, c) in [(0, 0), (0, 1), (4, 4), (4, 3)] {
            labels[r * 5 + c] = 1;
        }
        let pixels = vec![1.0; 25];
        let config = Config {
            image_name: "DNA".into(),
            bin_count: 2,
            wants_scaled: true,
            maximum_radius: 10.0,
            zernikes: ZernikeOption::None,
        };
        let result = measure(&pixels, &labels, 5, 5, &config).unwrap();
        for col in result.columns.values() {
            assert_eq!(col.len(), 1);
            assert!(col[0].is_finite() || col[0] == 0.0);
        }
    }

    fn assert_pilot_strict(actual: f64, expected: f64, label: &str) {
        let tol = 1e-10 + 1e-9 * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{label}: actual {actual:?} vs expected {expected:?} (tolerance {tol:?})"
        );
    }

    include!("../../../fixtures/intensity_distribution/float_precision.rs");

    // Generated by scripts/reference/generate_intensity_distribution.py;
    // real-oracle fixtures from fixtures/intensity_distribution/. One
    // #[test] per fixture file.
    include!("../../../fixtures/intensity_distribution/oracle_tests.rs");
}

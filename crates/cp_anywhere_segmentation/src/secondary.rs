//! `IdentifySecondaryObjects` (CellProfiler 4.2.8), standalone, no
//! Python/filesystem dependency, using this crate's existing declump,
//! threshold, and hole-filling primitives.
//!
//! # Declared scope (this increment)
//!
//! `Propagation` (the module's own default method) and `Distance - N`
//! are implemented and oracle-verified end to end. `Distance - B`,
//! `Watershed - Gradient`, and `Watershed - Image` are tracked as
//! explicit, not-yet-implemented gaps (see `spec/segmentation-modules.md`)
//! -- callers get a clear error naming the method, never a silent
//! fallback to a different one.

use crate::declump::neighbor;
use crate::{
    DeclumpSettings, FillHoles, IdentifyPrimaryObjectsConfig, ThresholdMethod, ThresholdScope,
};
use std::collections::{BTreeMap, BTreeSet};

/// `IdentifySecondaryObjects.method`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecondaryMethod {
    /// `M_PROPAGATION`, the module's own default.
    Propagation,
    /// `M_DISTANCE_N`: expand primary objects by a fixed pixel radius,
    /// with no secondary image or threshold involved at all.
    DistanceN,
}

/// Mirrors the subset of `IdentifySecondaryObjects` settings this
/// module implements -- see `spec/segmentation-modules.md` for the
/// settings this does *not* yet cover (`Distance - B`,
/// `Watershed - Gradient`, `Watershed - Image`, `wants_discard_edge`).
#[derive(Clone, Copy, Debug)]
pub struct SecondaryObjectsConfig {
    pub method: SecondaryMethod,
    /// `IdentifySecondaryObjects.regularization_factor` (`Propagation` only).
    pub regularization_factor: f64,
    /// `IdentifySecondaryObjects.distance_to_dilate` (`Distance - N` only).
    pub distance_to_dilate: f64,
    /// `IdentifySecondaryObjects.fill_holes`.
    pub fill_holes: bool,
    /// The secondary image's own threshold settings (`Propagation`
    /// only -- `Distance - N` never thresholds anything). Reuses
    /// `IdentifyPrimaryObjectsConfig`'s threshold-only fields; the
    /// size/border/declump fields are ignored (the secondary
    /// threshold call always disables them, matching
    /// `IdentifySecondaryObjects._threshold_image`'s own plain
    /// threshold-and-binarize, no object filtering at all).
    pub threshold_method: ThresholdMethod,
    pub threshold_correction_factor: f64,
    pub threshold_range: (f64, f64),
    pub threshold_smoothing_scale: f64,
    pub log_transform: bool,
    pub threshold_scope: ThresholdScope,
}

#[derive(Debug)]
pub struct SecondaryObjectsResult {
    pub labels: Vec<u32>,
    /// Grown labels before overlap-based filtering and renumbering.
    pub small_removed_labels: Vec<u32>,
    /// `None` for `Distance - N`, which never thresholds an image.
    pub threshold: Option<f64>,
}

/// `IdentifySecondaryObjects.run()`: grows a secondary object around
/// each primary ("seed") object.
///
/// `primary_labels_in` is `objects.unedited_segmented`, before primary
/// filtering and renumbering. Seed preparation retains all pixels of
/// edge-touching unedited objects; other pixels are seeds only where
/// `primary_kept_labels` is nonzero, matching CellProfiler's `labels_in`
/// construction. `primary_kept_labels` is `objects.segmented`, the final
/// primary object set, whose IDs may differ from the unedited IDs.
/// After growth, each secondary ID maps to the maximum final primary
/// ID overlapping it (`scipy.ndimage.maximum`); zero-overlap objects
/// are removed. Filtering is not based on membership of the seed ID
/// in the final primary set.
#[allow(clippy::too_many_arguments)]
pub fn identify_secondary_objects(
    image: &[f64],
    integer_pixels: Option<&[i64]>,
    image_mask: Option<&[bool]>,
    primary_labels_in: &[u32],
    primary_kept_labels: &[u32],
    rows: usize,
    cols: usize,
    config: &SecondaryObjectsConfig,
) -> Result<SecondaryObjectsResult, String> {
    let n = rows
        .checked_mul(cols)
        .ok_or("rows * cols overflowed usize")?;
    if image.len() != n || primary_labels_in.len() != n || primary_kept_labels.len() != n {
        return Err(
            "image, primary_labels_in, and primary_kept_labels must match rows * cols".into(),
        );
    }
    if n == 0 {
        return Err("image must be nonempty".into());
    }
    // CP starts with unedited labels, retains every edge-touching label,
    // and removes other seed pixels where final primary segmentation is 0.
    let mut edge_ids = BTreeSet::new();
    for row in 0..rows {
        edge_ids.insert(primary_labels_in[row * cols]);
        edge_ids.insert(primary_labels_in[row * cols + cols - 1]);
    }
    for col in 0..cols {
        edge_ids.insert(primary_labels_in[col]);
        edge_ids.insert(primary_labels_in[(rows - 1) * cols + col]);
    }
    let seeds: Vec<u32> = primary_labels_in
        .iter()
        .zip(primary_kept_labels)
        .map(|(&id, &kept)| {
            if kept != 0 || edge_ids.contains(&id) {
                id
            } else {
                0
            }
        })
        .collect();
    let primary_labels_in = seeds.as_slice();
    let (mut labels_out, threshold) = match config.method {
        SecondaryMethod::Propagation => {
            let threshold_config = IdentifyPrimaryObjectsConfig {
                threshold_method: config.threshold_method,
                threshold_correction_factor: config.threshold_correction_factor,
                threshold_range: config.threshold_range,
                threshold_smoothing_scale: config.threshold_smoothing_scale,
                log_transform: config.log_transform,
                threshold_scope: config.threshold_scope,
                exclude_size: false,
                min_diameter: 0.0,
                max_diameter: f64::INFINITY,
                exclude_border_objects: false,
                maximum_object_count: None,
                fill_holes: FillHoles::Never,
                declump: DeclumpSettings {
                    unclump: crate::declump::UnclumpMethod::None,
                    watershed: crate::declump::WatershedMethod::None,
                    smoothing: crate::declump::Smoothing::Automatic,
                    suppression: crate::declump::MaximaSuppression::Automatic,
                    low_resolution_maxima: false,
                },
            };
            let thresholded = crate::identify_primary_objects(
                image,
                integer_pixels,
                image_mask,
                rows,
                cols,
                &threshold_config,
            )?;
            let mask: Vec<bool> = thresholded.labels.iter().map(|&l| l > 0).collect();
            let (labels_out, _distances) = propagate_weighted(
                Some(image),
                primary_labels_in,
                &mask,
                cols,
                rows,
                config.regularization_factor,
            );
            (labels_out, Some(thresholded.threshold))
        }
        SecondaryMethod::DistanceN => {
            let (sq_distance, source) = nearest_labeled_source(primary_labels_in, cols, rows);
            let mut labels_out = vec![0u32; n];
            let limit = config.distance_to_dilate * config.distance_to_dilate;
            for p in 0..n {
                if sq_distance[p] <= limit {
                    labels_out[p] = primary_labels_in[source[p]];
                }
            }
            (labels_out, None)
        }
    };
    if config.fill_holes {
        // CP's mask is exactly labels_out == 0: the graph traversal is
        // unchanged, but writes to existing foreground are forbidden.
        let before = labels_out.clone();
        crate::holes::fill_labeled_holes(&mut labels_out, rows, cols, None);
        for (filled, &original) in labels_out.iter_mut().zip(&before) {
            if original != 0 {
                *filled = original;
            }
        }
    }
    let small_removed_labels = labels_out.clone();
    // scipy.ndimage.maximum(segmented, labels_out, index) maps original
    // seed IDs to the retained (possibly renumbered) primary IDs.
    let mut lookup: BTreeMap<u32, u32> = BTreeMap::new();
    for (&grown, &kept) in labels_out.iter().zip(primary_kept_labels) {
        if grown != 0 {
            lookup
                .entry(grown)
                .and_modify(|id| *id = (*id).max(kept))
                .or_insert(kept);
        }
    }
    for id in &mut labels_out {
        *id = lookup.get(id).copied().unwrap_or(0);
    }
    Ok(SecondaryObjectsResult {
        labels: labels_out,
        small_removed_labels,
        threshold,
    })
}

#[derive(Clone, Copy)]
struct PropagationEntry {
    key: (u64, u32, usize),
    pixel: usize,
    label: u32,
}

/// Matches centrosome.heap's push/pop operations, but accepts its initial
/// row-major seed array verbatim. `heap_from_numpy2` does not heapify that
/// array, so inserting the seeds through `declump::MinHeap::push` changes
/// which label first records an equal-distance neighbor.
struct PropagationHeap {
    data: Vec<PropagationEntry>,
}

impl PropagationHeap {
    fn push(&mut self, entry: PropagationEntry) {
        let mut child = self.data.len();
        self.data.push(entry);
        while child > 0 {
            let parent = (child - 1) / 2;
            if self.data[child].key >= self.data[parent].key {
                break;
            }
            self.data.swap(parent, child);
            child = parent;
        }
    }

    fn pop(&mut self) -> Option<PropagationEntry> {
        if self.data.is_empty() {
            return None;
        }
        let entry = self.data.swap_remove(0);
        let mut parent = 0;
        loop {
            let left = parent * 2 + 1;
            if left >= self.data.len() {
                break;
            }
            let right = left + 1;
            let mut smallest = parent;
            if self.data[left].key < self.data[smallest].key {
                smallest = left;
            }
            if right < self.data.len() && self.data[right].key < self.data[smallest].key {
                smallest = right;
            }
            if smallest == parent {
                break;
            }
            self.data.swap(parent, smallest);
            parent = smallest;
        }
        Some(entry)
    }
}

/// `centrosome.propagate.propagate(image, labels, mask, weight)`,
/// verified against the real compiled extension (not just the Cython
/// source read by eye): the weighted shortest-path region growing
/// `IdentifySecondaryObjects`'s `Propagation` method is built on.
///
/// `image` is `None` for the pure-geometric case (every pairwise pixel
/// difference is 0); when `Some`, the per-edge cost is
/// `sqrt(pixel_diff^2 + manhattan_distance * weight^2)`, where
/// `pixel_diff` sums `|image[p1+delta] - image[p2+delta]|` over the
/// full 3x3 neighborhood around both endpoints (edge-clamped), and
/// `manhattan_distance = |di| + |dj|` (1 for an orthogonal step, 2 for
/// a diagonal one, giving Euclidean spatial cost for a one-pixel step).
/// `seeds` gives the initial
/// labels (0 = unlabeled); `mask` bounds where labels may spread.
/// Returns `(labels_out, distances)`, matching the real function's
/// own two-value return exactly, including that every originally
/// seeded pixel keeps its own label in the output even if it sits
/// outside `mask` (real centrosome's `labels_out[labels > 0] =
/// labels[labels > 0]` final restoration step, not a convenience).
pub(crate) fn propagate_weighted(
    image: Option<&[f64]>,
    seeds: &[u32],
    mask: &[bool],
    w: usize,
    h: usize,
    weight: f64,
) -> (Vec<u32>, Vec<f64>) {
    let n = seeds.len();
    let mut out = vec![0u32; n];
    let mut distance = vec![-1.0f64; n];
    for p in 0..n {
        if seeds[p] != 0 {
            distance[p] = 0.0;
        }
    }
    let mut heap = PropagationHeap {
        data: seeds
            .iter()
            .zip(mask)
            .enumerate()
            .filter(|(_, (label, in_mask))| **label != 0 && **in_mask)
            .map(|(pixel, (&label, _))| PropagationEntry {
                key: (0, label, pixel),
                pixel,
                label,
            })
            .collect(),
    };
    while let Some(e) = heap.pop() {
        if out[e.pixel] != 0 {
            continue;
        }
        out[e.pixel] = e.label;
        let d0 = distance[e.pixel];
        let (i1, j1) = (e.pixel / w, e.pixel % w);
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let Some(q) = neighbor(e.pixel, w, h, dy, dx) else {
                    continue;
                };
                if out[q] != 0 || !mask[q] {
                    continue;
                }
                let (i2, j2) = (q / w, q % w);
                let edge = edge_distance(image, i1, j1, i2, j2, w, h, weight);
                let d = d0 + edge;
                if distance[q] < 0.0 || distance[q] > d {
                    distance[q] = d;
                    heap.push(PropagationEntry {
                        // centrosome converts positive doubles to two
                        // signed ints, discarding the low bit of the
                        // mantissa (see declump::propagate's own
                        // identical comment -- same real technique).
                        key: (d.to_bits() >> 1, e.label, q),
                        pixel: q,
                        label: e.label,
                    });
                }
            }
        }
    }
    for p in 0..n {
        if seeds[p] != 0 {
            out[p] = seeds[p];
        }
    }
    (out, distance)
}

#[allow(clippy::too_many_arguments)]
fn edge_distance(
    image: Option<&[f64]>,
    i1: usize,
    j1: usize,
    i2: usize,
    j2: usize,
    w: usize,
    h: usize,
    weight: f64,
) -> f64 {
    let manhattan = (i1 as isize - i2 as isize).unsigned_abs() as f64
        + (j1 as isize - j2 as isize).unsigned_abs() as f64;
    let Some(image) = image else {
        return (manhattan * weight * weight).sqrt();
    };
    let clamped_fetch = |i: isize, j: isize| -> f64 {
        let i = i.clamp(0, h as isize - 1) as usize;
        let j = j.clamp(0, w as isize - 1) as usize;
        image[i * w + j]
    };
    let mut pixel_diff = 0.0;
    for di in -1_isize..=1 {
        for dj in -1_isize..=1 {
            let v1 = clamped_fetch(i1 as isize + di, j1 as isize + dj);
            let v2 = clamped_fetch(i2 as isize + di, j2 as isize + dj);
            pixel_diff += (v1 - v2).abs();
        }
    }
    (pixel_diff * pixel_diff + manhattan * weight * weight).sqrt()
}

/// `scipy.ndimage.distance_transform_edt(labels_in == 0, return_indices=True)`:
/// for each pixel, the squared Euclidean distance to the nearest
/// pixel where `labels_in != 0`, and that nearest pixel's own flat
/// index (row-major). A labeled pixel's own distance is 0 and its
/// own index is itself. Verified against the real scipy function
/// (not re-derived from scratch): a separable two-pass extension of
/// this crate's own `edt_1d` lower-envelope technique, carrying a
/// "nearest source" payload through both passes instead of just a
/// distance, since `IdentifySecondaryObjects`'s `"Distance - N"`
/// method needs to know *which* primary object each expanded pixel
/// came from, not merely how far it is.
pub(crate) fn nearest_labeled_source(
    labels_in: &[u32],
    w: usize,
    h: usize,
) -> (Vec<f64>, Vec<usize>) {
    let inf = ((w * w + h * h) as f64) * 1e6 + 1e6;
    // Pass 1: per column, nearest labeled row and its squared distance.
    let mut d1 = vec![0.0f64; labels_in.len()];
    let mut src1 = vec![0usize; labels_in.len()];
    for x in 0..w {
        let f: Vec<f64> = (0..h)
            .map(|y| if labels_in[y * w + x] != 0 { 0.0 } else { inf })
            .collect();
        let payload: Vec<usize> = (0..h).map(|y| y * w + x).collect();
        let (d, src) = edt_1d_indexed(&f, &payload);
        for y in 0..h {
            d1[y * w + x] = d[y];
            src1[y * w + x] = src[y];
        }
    }
    // Pass 2: per row, minimize over columns using pass 1's result as
    // the base cost, carrying pass 1's source pixel as the payload.
    let mut out_d = vec![0.0f64; labels_in.len()];
    let mut out_src = vec![0usize; labels_in.len()];
    for y in 0..h {
        let row_f: Vec<f64> = (0..w).map(|x| d1[y * w + x]).collect();
        let row_payload: Vec<usize> = (0..w).map(|x| src1[y * w + x]).collect();
        let (d, src) = edt_1d_indexed(&row_f, &row_payload);
        for x in 0..w {
            out_d[y * w + x] = d[x];
            out_src[y * w + x] = src[x];
        }
    }
    (out_d, out_src)
}

/// Like `declump`'s own `edt_1d`, but each sample `f[v]` carries a
/// `payload[v]` that is propagated to the output when `v` wins the
/// lower envelope at a query point, instead of returning only `v`
/// itself. The envelope construction (the `v`/`z` arrays) is
/// unaffected by `payload` -- only the final reconstruction pass
/// reads it.
#[allow(clippy::needless_range_loop)]
fn edt_1d_indexed(f: &[f64], payload: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = ((f[q] + (q * q) as f64) - (f[v[k]] + (v[k] * v[k]) as f64))
            / (2.0 * q as f64 - 2.0 * v[k] as f64);
        while k > 0 && s <= z[k] {
            k -= 1;
            s = ((f[q] + (q * q) as f64) - (f[v[k]] + (v[k] * v[k]) as f64))
                / (2.0 * q as f64 - 2.0 * v[k] as f64);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        if k + 1 < z.len() {
            z[k + 1] = f64::INFINITY;
        }
    }
    let mut d = vec![0.0f64; n];
    let mut source = vec![0usize; n];
    let mut k = 0usize;
    for q in 0..n {
        while k < n && z[k + 1] < q as f64 {
            k += 1;
        }
        let dv = q as f64 - v[k] as f64;
        d[q] = dv * dv + f[v[k]];
        source[q] = payload[v[k]];
    }
    (d, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagate_weighted_preserves_real_centrosome_initial_seed_order() {
        // Compiled centrosome 1.2.3 loads np.argwhere's row-major seed
        // array directly, without heapifying it by label. The first
        // seed (id 2) therefore claims the tied pixel before id 1.
        let seeds = [0, 2, 0, 1];
        for weight in [0.0, 0.05, 1.0] {
            let (labels, distances) =
                propagate_weighted(Some(&[1.0; 4]), &seeds, &[true; 4], 4, 1, weight);
            assert_eq!(labels, [2, 2, 2, 1], "weight={weight}");
            assert_eq!(distances, [weight, 0.0, weight, 0.0]);
        }
    }

    #[test]
    fn propagate_weighted_matches_real_centrosome_zero_weight_ties() {
        // Genuine compiled centrosome output, including its non-stable
        // pop/push order after loading three unsorted initial labels.
        let seeds = [0, 3, 0, 0, 2, 0, 0, 1, 0];
        let (labels, distances) =
            propagate_weighted(Some(&[1.0; 9]), &seeds, &[true; 9], 9, 1, 0.0);
        assert_eq!(labels, [3, 3, 3, 2, 2, 1, 1, 1, 1]);
        assert_eq!(distances, [0.0; 9]);
    }

    #[test]
    fn propagate_weighted_matches_real_centrosome_masked_multi_pixel_ties() {
        // Compiled centrosome 1.2.3, with interleaved multi-pixel
        // labels, masked background, and one seed outside the mask.
        #[rustfmt::skip]
        let seeds = [
            0, 3, 3, 0, 2,
            0, 3, 0, 0, 2,
            0, 0, 0, 0, 0,
            1, 1, 0, 0, 0,
        ];
        let mut mask = [true; 20];
        for p in [4, 7, 10] {
            mask[p] = false;
        }
        let (labels, distances) = propagate_weighted(Some(&[1.0; 20]), &seeds, &mask, 5, 4, 0.0);
        #[rustfmt::skip]
        let expected = [
            3, 3, 3, 1, 2,
            3, 3, 0, 1, 2,
            0, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        assert_eq!(labels, expected);
        let mut expected_distances = [0.0; 20];
        expected_distances[7] = -1.0;
        expected_distances[10] = -1.0;
        assert_eq!(distances, expected_distances);
    }

    #[test]
    fn propagation_diagonal_cost_matches_real_centrosome() {
        // Genuine compiled centrosome, 2x2 constant image, seed at (0,0).
        // The diagonal cost is sqrt(2 * weight^2), not 2 * weight.
        for (weight, diagonal) in [(0.05, 0.07071067811865477), (1.0, std::f64::consts::SQRT_2)] {
            for image in [None, Some([1.0; 4].as_slice())] {
                let (labels, distances) =
                    propagate_weighted(image, &[1, 0, 0, 0], &[true; 4], 2, 2, weight);
                assert_eq!(labels, [1, 1, 1, 1]);
                for (actual, expected) in distances.iter().zip([0.0, weight, weight, diagonal]) {
                    assert!((actual - expected).abs() < 1e-14, "{actual} != {expected}");
                }
            }
        }
    }

    #[test]
    fn propagate_weighted_matches_real_centrosome_with_an_image_term() {
        // 1x6 row, seeds at columns 1 (label 1) and 4 (label 2), image
        // with a "cliff" at column 3. Verified against a real,
        // directly-invoked `centrosome.propagate.propagate` call with
        // the same image/seeds/weight (not hand-derived): the labels
        // split down the middle regardless of weight (the cliff
        // dominates either way here), but the exact distances only
        // match if the weighted formula itself is right.
        let image: Vec<f64> = vec![0.1, 0.1, 0.1, 0.9, 0.9, 0.9];
        let mut seeds = vec![0u32; 6];
        seeds[1] = 1;
        seeds[4] = 2;
        let mask = vec![true; 6];

        let (labels, dist) = propagate_weighted(Some(&image), &seeds, &mask, 6, 1, 0.05);
        assert_eq!(labels, vec![1, 1, 1, 2, 2, 2]);
        let expected_dist = [0.05, 0.0, 2.400520776831561, 2.400520776831561, 0.0, 0.05];
        for (got, want) in dist.iter().zip(expected_dist) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }

        let (labels_w1, dist_w1) = propagate_weighted(Some(&image), &seeds, &mask, 6, 1, 1.0);
        assert_eq!(labels_w1, vec![1, 1, 1, 2, 2, 2]);
        let expected_dist_w1 = [1.0, 0.0, 2.6, 2.6, 0.0, 1.0];
        for (got, want) in dist_w1.iter().zip(expected_dist_w1) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn propagate_weighted_restores_seeds_outside_the_mask() {
        // Real centrosome's `labels_out[labels > 0] = labels[labels > 0]`
        // final step: a seed pixel keeps its own label even when the
        // mask excludes it (it just never actively propagates).
        let image: Vec<f64> = vec![0.0; 6];
        let mut seeds = vec![0u32; 6];
        seeds[0] = 1;
        seeds[3] = 2;
        let mut mask = vec![true; 6];
        mask[0] = false; // seed 1 sits outside the mask.
        let (labels, _dist) = propagate_weighted(Some(&image), &seeds, &mask, 6, 1, 1.0);
        assert_eq!(labels[0], 1, "excluded seed keeps its own label");
        // Only seed 2 actively propagates (seed 1 was never pushed).
        assert_eq!(labels[1], 2);
        assert_eq!(labels[2], 2);
    }

    #[test]
    fn nearest_labeled_source_matches_real_scipy_distance_transform_edt() {
        // 1x10 row, labels at columns 2 (id 1) and 7 (id 2). Verified
        // against a real `scipy.ndimage.distance_transform_edt(labels_in
        // == 0, return_indices=True)` call: distances
        // [2,1,0,1,2,2,1,0,1,2]^2 (squared) and the nearest source
        // resolving to column 2 for the left half, column 7 for the
        // right half.
        let labels_in: Vec<u32> = vec![0, 0, 1, 0, 0, 0, 0, 2, 0, 0];
        let (d, src) = nearest_labeled_source(&labels_in, 10, 1);
        let expected_sq_dist = [4.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 4.0];
        for (got, want) in d.iter().zip(expected_sq_dist) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        let nearest_id: Vec<u32> = src.iter().map(|&i| labels_in[i]).collect();
        assert_eq!(nearest_id, vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn identify_secondary_objects_propagation_matches_a_real_end_to_end_oracle_run() {
        // Real CellProfiler 4.2.8 IdentifySecondaryObjects, Propagation
        // method, regularization_factor=0.05, global Otsu threshold,
        // fill_holes=True, on a 10x12 radial-falloff image with two
        // primary seeds at (5,3) and (5,8) -- verified by directly
        // running the real module (not re-derived): the two secondary
        // objects split cleanly down the middle, each roughly circular,
        // with the exact boundary pixels below.
        #[rustfmt::skip]
        let image: Vec<f64> = vec![
            0.02817468419244984, 0.10247253214424934, 0.15016341440120262, 0.16666666666666663, 0.15016341440120262, 0.10247253214424934, 0.10247253214424934, 0.15016341440120262, 0.16666666666666663, 0.15016341440120262, 0.10247253214424934, 0.02817468419244984,
            0.16666666666666663, 0.2546440075000701, 0.31281572906372324, 0.33333333333333337, 0.31281572906372324, 0.2546440075000701, 0.2546440075000701, 0.31281572906372324, 0.33333333333333337, 0.31281572906372324, 0.2546440075000701, 0.16666666666666663,
            0.29289321881345254, 0.39907478742266844, 0.4729537233052701, 0.5, 0.4729537233052701, 0.39907478742266844, 0.39907478742266844, 0.4729537233052701, 0.5, 0.4729537233052701, 0.39907478742266844, 0.29289321881345254,
            0.39907478742266844, 0.5285954792089682, 0.6273220037500351, 0.6666666666666667, 0.6273220037500351, 0.5285954792089682, 0.5285954792089682, 0.6273220037500351, 0.6666666666666667, 0.6273220037500351, 0.5285954792089682, 0.39907478742266844,
            0.4729537233052701, 0.6273220037500351, 0.7642977396044841, 0.8333333333333334, 0.7642977396044841, 0.6273220037500351, 0.6273220037500351, 0.7642977396044841, 0.8333333333333334, 0.7642977396044841, 0.6273220037500351, 0.4729537233052701,
            0.5, 0.6666666666666667, 0.8333333333333334, 1.0, 0.8333333333333334, 0.6666666666666667, 0.6666666666666667, 0.8333333333333334, 1.0, 0.8333333333333334, 0.6666666666666667, 0.5,
            0.4729537233052701, 0.6273220037500351, 0.7642977396044841, 0.8333333333333334, 0.7642977396044841, 0.6273220037500351, 0.6273220037500351, 0.7642977396044841, 0.8333333333333334, 0.7642977396044841, 0.6273220037500351, 0.4729537233052701,
            0.39907478742266844, 0.5285954792089682, 0.6273220037500351, 0.6666666666666667, 0.6273220037500351, 0.5285954792089682, 0.5285954792089682, 0.6273220037500351, 0.6666666666666667, 0.6273220037500351, 0.5285954792089682, 0.39907478742266844,
            0.29289321881345254, 0.39907478742266844, 0.4729537233052701, 0.5, 0.4729537233052701, 0.39907478742266844, 0.39907478742266844, 0.4729537233052701, 0.5, 0.4729537233052701, 0.39907478742266844, 0.29289321881345254,
            0.16666666666666663, 0.2546440075000701, 0.31281572906372324, 0.33333333333333337, 0.31281572906372324, 0.2546440075000701, 0.2546440075000701, 0.31281572906372324, 0.33333333333333337, 0.31281572906372324, 0.2546440075000701, 0.16666666666666663,
        ];
        let mut primary = vec![0u32; 120];
        primary[5 * 12 + 3] = 1;
        primary[5 * 12 + 8] = 2;

        #[rustfmt::skip]
        let expected: Vec<u32> = vec![
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,1,1,1,2,2,2,2,2,0,
            1,1,1,1,1,1,2,2,2,2,2,2,
            1,1,1,1,1,1,2,2,2,2,2,2,
            1,1,1,1,1,1,2,2,2,2,2,2,
            1,1,1,1,1,1,2,2,2,2,2,2,
            1,1,1,1,1,1,2,2,2,2,2,2,
            0,1,1,1,1,1,2,2,2,2,2,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
        ];

        let config = SecondaryObjectsConfig {
            method: SecondaryMethod::Propagation,
            regularization_factor: 0.05,
            distance_to_dilate: 0.0,
            fill_holes: true,
            threshold_method: ThresholdMethod::OtsuTwoClass,
            threshold_correction_factor: 1.0,
            threshold_range: (0.0, 1.0),
            threshold_smoothing_scale: 0.0,
            log_transform: false,
            threshold_scope: ThresholdScope::Global,
        };
        let result =
            identify_secondary_objects(&image, None, None, &primary, &primary, 10, 12, &config)
                .unwrap();
        assert_eq!(result.labels, expected);
        assert!(result.threshold.is_some());
    }

    #[test]
    fn identify_secondary_objects_distance_n_expands_by_a_fixed_radius() {
        // No image/threshold involved at all -- a pure geometric
        // expansion, verified directly against the already
        // oracle-matched `nearest_labeled_source` (itself matched to
        // real scipy): radius 1.5 keeps only pixels within that
        // Euclidean distance of a seed.
        let primary: Vec<u32> = vec![0, 0, 1, 0, 0, 0, 0, 2, 0, 0];
        let image = vec![0.0; 10];
        let config = SecondaryObjectsConfig {
            method: SecondaryMethod::DistanceN,
            regularization_factor: 0.0,
            distance_to_dilate: 1.5,
            fill_holes: false,
            threshold_method: ThresholdMethod::OtsuTwoClass,
            threshold_correction_factor: 1.0,
            threshold_range: (0.0, 1.0),
            threshold_smoothing_scale: 0.0,
            log_transform: false,
            threshold_scope: ThresholdScope::Global,
        };
        let result =
            identify_secondary_objects(&image, None, None, &primary, &primary, 1, 10, &config)
                .unwrap();
        assert_eq!(result.labels, vec![0, 1, 1, 1, 0, 0, 2, 2, 2, 0]);
        assert!(result.threshold.is_none());
    }
}

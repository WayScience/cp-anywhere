//! CellProfiler 4.2.8 `MeasureGranularity` (2D case), oracle-verified against a
//! pinned reference CellProfiler/scipy/skimage install.
//!
//! This is a from-scratch Rust port -- no `ndarray`, `scipy`, or `skimage`
//! dependency -- of `measuregranularity.py`'s `run_on_image_setting` (2D
//! branch only). Every numerical primitive below (bilinear/cubic
//! `map_coordinates`, the cubic B-spline prefilter, grayscale
//! erosion/dilation, and morphological reconstruction by dilation) was
//! independently verified against the real, pinned CellProfiler 4.2.8
//! reference venv (numpy 1.23.5, scipy 1.9.3, scikit-image 0.21.0) with small
//! adversarial probes before being ported here; see the findings below each
//! helper. All 12 real-oracle fixtures in `fixtures/granularity/` pass
//! exactly (see the `oracle_*` tests at the bottom of this file).
//!
//! Known, confirmed-real-CellProfiler limitation (not a port bug): CP's own
//! `run_on_image_setting` divides by `orig_shape[dim] - 1` when rescaling
//! `rec` back to full resolution and crashes with `ZeroDivisionError` for any
//! 1-row or 1-column image. [`measure`] returns a clear [`Err`] for that case
//! instead of panicking or silently producing 0/NaN.

use std::collections::BTreeMap;

/// Settings mirroring `MeasureGranularity`'s module settings.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// "Subsampling factor for granularity measurements" (`(0, 1]` downsamples).
    pub subsample_size: f64,
    /// "Subsampling factor for background reduction" (`(0, 1]` downsamples).
    pub image_sample_size: f64,
    /// "Radius of structuring element", in pixels of the (possibly
    /// subsampled) working image.
    pub element_size: usize,
    /// "Range of the granular spectrum": number of erosion/reconstruction
    /// scales to report.
    pub granular_spectrum_length: usize,
}

/// Per-image and per-object-set granularity spectra.
#[derive(Debug)]
pub struct GranularityResult {
    /// `image[scale_index]` is the image-level `GS<scale_index + 1>` value.
    pub image: Vec<f64>,
    /// `objects[name][scale_index]` is one value per object (1-based object
    /// IDs in ascending order) at that scale; an object with zero pixels
    /// under the mask reports `NaN` forever (matching real CellProfiler,
    /// since its baseline mean is `NaN` and `NaN` never recovers through the
    /// per-scale subtraction). An object set with zero objects reports an
    /// empty vector at every scale.
    pub objects: BTreeMap<String, Vec<Vec<f64>>>,
}

/// `f64::EPSILON`, spelled out to mirror `numpy.finfo(float).eps` call sites
/// in the original Python 1:1.
const NP_EPS: f64 = f64::EPSILON;

fn ceil_usize(x: f64) -> usize {
    x.ceil() as usize
}

/// scipy.ndimage's `mode='mirror'` ("whole-sample symmetric", no duplicated
/// edge sample) index reflection, confirmed against
/// `scipy/ndimage/src/ni_interpolation.c`'s `map_coordinate` (`NI_EXTEND_MIRROR`
/// branch): index `-1` reflects to `1`, not `0`; period is `2 * (len - 1)`.
fn mirror_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let len = len as isize;
    let period = 2 * (len - 1);
    let mut m = idx.rem_euclid(period);
    if m >= len {
        m = period - m;
    }
    m as usize
}

/// Cubic B-spline interpolation weights for fractional offset `t` in
/// `[0, 1)`, at knot positions `floor(x) - 1 .. floor(x) + 2`. Transcribed
/// from `get_spline_interpolation_weights`'s `order == 3` case in
/// `scipy/ndimage/src/ni_splines.c` (scipy 1.9.3).
fn cubic_weights(t: f64) -> [f64; 4] {
    let y = t;
    let z = 1.0 - t;
    let w1 = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    let w2 = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    let w0 = z * z * z / 6.0;
    let w3 = 1.0 - w0 - w1 - w2;
    [w0, w1, w2, w3]
}

/// `scipy.ndimage.map_coordinates(data, (r, c), order=1, mode='constant',
/// cval=0.0)` for a single output point, on a 2D array.
///
/// Probe finding (see `probe3.py`/`ni_interpolation.c`'s `NI_GeometricTransform`):
/// mode='constant' is a HARD per-axis domain check on the coordinate itself,
/// not a per-sample blend -- if `r` or `c` is `< 0` or `> len - 1` for its
/// axis, the ENTIRE output is `cval`, even if every actually-touched stencil
/// sample would otherwise be in-bounds.
fn sample_bilinear(data: &[f64], rows: usize, cols: usize, r: f64, c: f64) -> f64 {
    if r < 0.0 || r > (rows - 1) as f64 || c < 0.0 || c > (cols - 1) as f64 {
        return 0.0;
    }
    let r0f = r.floor();
    let rf = r - r0f;
    let r0 = r0f as isize;
    let c0f = c.floor();
    let cf = c - c0f;
    let c0 = c0f as isize;
    let wr = [1.0 - rf, rf];
    let wc = [1.0 - cf, cf];
    let mut total = 0.0;
    for (i, &w_r) in wr.iter().enumerate() {
        let ri = mirror_index(r0 + i as isize, rows);
        for (j, &w_c) in wc.iter().enumerate() {
            let cj = mirror_index(c0 + j as isize, cols);
            total += w_r * w_c * data[ri * cols + cj];
        }
    }
    total
}

/// `scipy.ndimage.map_coordinates(coeffs, (r, c), order=3, mode='constant',
/// cval=0.0, prefilter=False)` for a single output point, on already-prefiltered
/// cubic B-spline coefficients.
///
/// Probe finding: the per-axis out-of-domain check is identical to order=1
/// (hard cval cutoff at `< 0` or `> len - 1`), but stencil samples that spill
/// past the array edge while the coordinate itself stays in-domain are
/// extended with MIRROR indexing, not `cval` -- confirmed via
/// `_get_spline_boundary_mode` in `ni_interpolation.c`, which maps
/// `NI_EXTEND_CONSTANT` to `NI_EXTEND_MIRROR` for exactly this purpose.
fn sample_cubic(data: &[f64], rows: usize, cols: usize, r: f64, c: f64) -> f64 {
    if r < 0.0 || r > (rows - 1) as f64 || c < 0.0 || c > (cols - 1) as f64 {
        return 0.0;
    }
    let rff = r.floor();
    let rt = r - rff;
    let r0 = rff as isize - 1;
    let cff = c.floor();
    let ct = c - cff;
    let c0 = cff as isize - 1;
    let wr = cubic_weights(rt);
    let wc = cubic_weights(ct);
    let mut total = 0.0;
    for (i, &w_r) in wr.iter().enumerate() {
        let ri = mirror_index(r0 + i as isize, rows);
        for (j, &w_c) in wc.iter().enumerate() {
            let cj = mirror_index(c0 + j as isize, cols);
            total += w_r * w_c * data[ri * cols + cj];
        }
    }
    total
}

/// In-place cubic B-spline prefilter along one axis (one row or column),
/// mirror ("whole-sample symmetric") boundary conditions.
///
/// Transcribed from `apply_filter`/`_init_causal_mirror`/
/// `_init_anticausal_mirror`/`_apply_filter_gain` in
/// `scipy/ndimage/src/ni_splines.c` (scipy 1.9.3), single-pole cubic case
/// (`z = sqrt(3) - 2`).
///
/// Probe finding (`probe5.py`): `scipy.ndimage.spline_filter(..., mode=
/// 'constant')` produces BIT-IDENTICAL coefficients to `mode='mirror'` --
/// confirmed directly against `_get_spline_boundary_mode` in
/// `ni_interpolation.c`, which routes `NI_EXTEND_CONSTANT` to
/// `NI_EXTEND_MIRROR` for prefiltering ("Modes without an analytic prefilter
/// ... use mirror extension"). So the mask resample's nominal
/// `mode='constant'` prefilter step always behaves like mirror, regardless
/// of the requested mode; only the final per-point evaluation's `cval`
/// cutoff (see [`sample_cubic`]) is actually "constant".
fn cubic_bspline_prefilter_1d(data: &mut [f64]) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let z = 3.0_f64.sqrt() - 2.0;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in data.iter_mut() {
        *v *= gain;
    }
    let z_n_1 = z.powi((n - 1) as i32);
    let mut c0 = data[0] + z_n_1 * data[n - 1];
    let mut z_i = z;
    for i in 1..(n - 1) {
        c0 += z_i * (data[i] + z_n_1 * data[n - 1 - i]);
        z_i *= z;
    }
    c0 /= 1.0 - z_n_1 * z_n_1;
    data[0] = c0;
    for i in 1..n {
        data[i] += z * data[i - 1];
    }
    data[n - 1] = (z * data[n - 2] + data[n - 1]) * z / (z * z - 1.0);
    for i in (0..(n - 1)).rev() {
        data[i] = z * (data[i + 1] - data[i]);
    }
}

/// Separable 2D cubic B-spline prefilter (rows, then columns).
fn cubic_bspline_prefilter_2d(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = data.to_vec();
    for row in out.chunks_exact_mut(cols) {
        cubic_bspline_prefilter_1d(row);
    }
    let mut col_buf = vec![0.0; rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = out[r * cols + c];
        }
        cubic_bspline_prefilter_1d(&mut col_buf);
        for r in 0..rows {
            out[r * cols + c] = col_buf[r];
        }
    }
    out
}

/// Resample `data` (bilinear, order=1) onto an `out_rows x out_cols` grid
/// where output `(r, c)` samples source coordinate `(r / scale, c / scale)`
/// -- `numpy.mgrid[0:out_rows, 0:out_cols] / scale` in the original.
fn resample_bilinear(
    data: &[f64],
    rows: usize,
    cols: usize,
    out_rows: usize,
    out_cols: usize,
    scale: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; out_rows * out_cols];
    for r in 0..out_rows {
        let ii = r as f64 / scale;
        for c in 0..out_cols {
            let jj = c as f64 / scale;
            out[r * out_cols + c] = sample_bilinear(data, rows, cols, ii, jj);
        }
    }
    out
}

/// Resample a boolean mask (cubic prefilter on the 0.0/1.0 cast, threshold
/// `> 0.9`), matching `map_coordinates(mask.astype(float), ...) > 0.9` with
/// its default `order=3, prefilter=True`.
fn resample_mask(
    mask: &[bool],
    rows: usize,
    cols: usize,
    out_rows: usize,
    out_cols: usize,
    scale: f64,
) -> Vec<bool> {
    let mask_f: Vec<f64> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let coeffs = cubic_bspline_prefilter_2d(&mask_f, rows, cols);
    let mut out = vec![false; out_rows * out_cols];
    for r in 0..out_rows {
        let ii = r as f64 / scale;
        for c in 0..out_cols {
            let jj = c as f64 / scale;
            out[r * out_cols + c] = sample_cubic(&coeffs, rows, cols, ii, jj) > 0.9;
        }
    }
    out
}

/// `skimage.morphology.disk(radius, dtype=bool)`'s true offsets relative to
/// its own center: `(2 * radius + 1)^2` square, true where
/// `dr^2 + dc^2 <= radius^2` (confirmed against a live `disk()` call).
fn disk_offsets(radius: usize) -> Vec<(isize, isize)> {
    let r = radius as isize;
    let mut offsets = Vec::new();
    for dr in -r..=r {
        for dc in -r..=r {
            if dr * dr + dc * dc <= r * r {
                offsets.push((dr, dc));
            }
        }
    }
    offsets
}

/// Grayscale erosion (min filter over `offsets`), ignoring out-of-bounds
/// neighbors entirely (equivalent to padding with `+inf`).
///
/// Probe finding (`probe7_morph_match.py`): `skimage.morphology.erosion`
/// with an explicit `footprint` does NOT use scipy's usual `grey_erosion`
/// default border mode (`'reflect'`, i.e. duplicate-edge) -- it simply
/// excludes any neighbor offset that falls outside the array from the
/// min/max, confirmed by exhaustive comparison against a naive
/// bounds-checked reference over many random small arrays/radii.
fn grey_erosion(img: &[f64], rows: usize, cols: usize, offsets: &[(isize, isize)]) -> Vec<f64> {
    extremum_filter(img, rows, cols, offsets, f64::min)
}

/// Grayscale dilation (max filter over `offsets`), ignoring out-of-bounds
/// neighbors entirely (equivalent to padding with `-inf`). See
/// [`grey_erosion`] for the border-handling probe finding (dilation is the
/// same filter with `max` instead of `min`).
fn grey_dilation(img: &[f64], rows: usize, cols: usize, offsets: &[(isize, isize)]) -> Vec<f64> {
    extremum_filter(img, rows, cols, offsets, f64::max)
}

fn extremum_filter(
    img: &[f64],
    rows: usize,
    cols: usize,
    offsets: &[(isize, isize)],
    combine: fn(f64, f64) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut best: Option<f64> = None;
            for &(dr, dc) in offsets {
                let rr = r as isize + dr;
                let cc = c as isize + dc;
                if rr >= 0 && rr < rows as isize && cc >= 0 && cc < cols as isize {
                    let v = img[rr as usize * cols + cc as usize];
                    best = Some(match best {
                        Some(b) => combine(b, v),
                        None => v,
                    });
                }
            }
            // `offsets` always includes `(0, 0)` (disk(r) with r >= 0
            // contains its own center), so every pixel has at least one
            // in-bounds neighbor (itself).
            out[r * cols + c] = best.expect("disk offsets always include the center");
        }
    }
    out
}

/// Morphological reconstruction by dilation: `skimage.morphology.
/// reconstruction(seed, mask_image, method='dilation', footprint=...)`.
///
/// Probe finding (`probe8_reconstruction.py`): skimage's fast hybrid
/// algorithm and this naive "dilate, cap by mask, repeat until a fixed
/// point" loop produce bit-identical results over many random cases --
/// expected, since reconstruction by dilation is *defined* as the unique
/// maximal fixed point of that iteration; any correct algorithm computing it
/// must agree, regardless of internal strategy.
fn reconstruct_by_dilation(
    seed: &[f64],
    mask_image: &[f64],
    rows: usize,
    cols: usize,
    offsets: &[(isize, isize)],
) -> Vec<f64> {
    let mut cur = seed.to_vec();
    // Each pixel can only move towards its mask cap and is bounded by it;
    // a fixed point is reached well within a few times the pixel count for
    // any real image. This bound only guards against a logic bug turning
    // into an infinite loop, not a legitimate slow convergence.
    let max_iter = rows
        .saturating_mul(cols)
        .saturating_mul(4)
        .saturating_add(1_000);
    for _ in 0..max_iter {
        let dilated = grey_dilation(&cur, rows, cols, offsets);
        let mut changed = false;
        let mut next = vec![0.0; rows * cols];
        for i in 0..rows * cols {
            let v = dilated[i].min(mask_image[i]);
            if v != cur[i] {
                changed = true;
            }
            next[i] = v;
        }
        if !changed {
            return next;
        }
        cur = next;
    }
    cur
}

/// Python's two-argument builtin `max(a, b)`: `b` wins only if `b > a`, so a
/// `NaN` first argument "wins" (stays `NaN`) while a `NaN` second argument
/// "loses" (first argument, non-`NaN`, is kept). This is NOT the same as
/// `numpy.maximum`, which always propagates `NaN` -- see [`np_maximum`]. The
/// real module calls plain `max(startmean, numpy.finfo(float).eps)` for the
/// image-level clamp.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// `numpy.maximum(a, b)`: propagates `NaN` regardless of argument order,
/// unlike [`py_max`]. The real module calls `numpy.maximum(current_mean,
/// numpy.finfo(float).eps)` for each object's baseline clamp -- so an object
/// with zero pixels under the mask (`current_mean` is `NaN`) gets a `NaN`
/// `start_mean` and reports `NaN` at every scale forever, since the
/// per-scale division by `start_mean` can never turn a `NaN` back into a
/// real number.
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Mean of `data[i]` over indices where `keep[i]` is true; `NaN` (via `0.0 /
/// 0.0`) if no indices are kept, matching `numpy.mean` on an empty slice.
fn masked_mean(data: &[f64], keep: &[bool]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            sum += data[i];
            count += 1;
        }
    }
    sum / count as f64
}

/// Per-object mean of `data` over `masked_labels == k` for `k` in
/// `1..=nobjects`; `NaN` for any label with zero matching pixels.
fn per_object_means(data: &[f64], masked_labels: &[u32], nobjects: usize) -> Vec<f64> {
    let mut sums = vec![0.0; nobjects];
    let mut counts = vec![0usize; nobjects];
    for (i, &lab) in masked_labels.iter().enumerate() {
        if lab > 0 {
            let k = lab as usize - 1;
            sums[k] += data[i];
            counts[k] += 1;
        }
    }
    sums.iter()
        .zip(&counts)
        .map(|(&s, &c)| s / c as f64)
        .collect()
}

struct ObjectState {
    masked_labels: Vec<u32>,
    nobjects: usize,
    current_mean: Vec<f64>,
    start_mean: Vec<f64>,
}

/// Measure 2D `MeasureGranularity` image- and object-level spectra.
///
/// `pixels`/`mask` are row-major, length `rows * cols`. Each `objects` entry's
/// labels array is also row-major, length `rows * cols`, `0` meaning
/// background. `mask: None` means every pixel is valid.
///
/// Returns `Err` if `rows == 1` or `cols == 1` -- matching a genuine,
/// confirmed-live real-CellProfiler limitation: `run_on_image_setting`
/// divides by `orig_shape[dim] - 1` when projecting the reconstructed image
/// back to full resolution and raises `ZeroDivisionError` for a 1-row or
/// 1-column image. See `scripts/reference/generate_granularity.py` for the
/// equivalent real-oracle-side note.
pub fn measure(
    pixels: &[f64],
    objects: &[(&str, &[u32])],
    mask: Option<&[bool]>,
    rows: usize,
    cols: usize,
    config: &Config,
) -> Result<GranularityResult, String> {
    if rows == 1 || cols == 1 {
        return Err(
            "MeasureGranularity on a 1-row or 1-column image divides by (dim - 1) in real \
             CellProfiler and raises ZeroDivisionError there; this is a real limitation, not \
             something this port papers over"
                .into(),
        );
    }
    let size = rows
        .checked_mul(cols)
        .ok_or("rows * cols overflowed usize")?;
    if pixels.len() != size {
        return Err(format!(
            "pixels length {} does not match rows*cols {size}",
            pixels.len()
        ));
    }
    if let Some(m) = mask {
        if m.len() != size {
            return Err(format!(
                "mask length {} does not match rows*cols {size}",
                m.len()
            ));
        }
    }
    for (name, labels) in objects {
        if labels.len() != size {
            return Err(format!(
                "object {name:?} labels length {} does not match rows*cols {size}",
                labels.len()
            ));
        }
    }
    if config.subsample_size <= 0.0 || config.image_sample_size <= 0.0 {
        return Err("subsample_size and image_sample_size must be positive".into());
    }

    let orig_mask: Vec<bool> = match mask {
        Some(m) => m.to_vec(),
        None => vec![true; size],
    };
    let orig_rows_f = rows as f64;
    let orig_cols_f = cols as f64;

    // Downsample the image and mask ("granularity measurements" subsample).
    let subsample_size = config.subsample_size;
    let (new_shape_float, grid_rows, grid_cols, ds_pixels, ds_mask) = if subsample_size < 1.0 {
        let new_shape_float = [orig_rows_f * subsample_size, orig_cols_f * subsample_size];
        let grid_rows = ceil_usize(new_shape_float[0]);
        let grid_cols = ceil_usize(new_shape_float[1]);
        let ds_pixels = resample_bilinear(pixels, rows, cols, grid_rows, grid_cols, subsample_size);
        let ds_mask = resample_mask(&orig_mask, rows, cols, grid_rows, grid_cols, subsample_size);
        (new_shape_float, grid_rows, grid_cols, ds_pixels, ds_mask)
    } else {
        (
            [orig_rows_f, orig_cols_f],
            rows,
            cols,
            pixels.to_vec(),
            orig_mask.clone(),
        )
    };
    if grid_rows == 0 || grid_cols == 0 {
        return Err("subsample_size produced an empty downsampled grid".into());
    }

    // Remove background pixels using a greyscale tophat filter
    // ("background reduction" subsample).
    let image_sample_size = config.image_sample_size;
    let (back_shape_float, back_rows, back_cols, back_pixels0, back_mask0) =
        if image_sample_size < 1.0 {
            let back_shape_float = [
                new_shape_float[0] * image_sample_size,
                new_shape_float[1] * image_sample_size,
            ];
            let back_rows = ceil_usize(back_shape_float[0]);
            let back_cols = ceil_usize(back_shape_float[1]);
            let back_pixels0 = resample_bilinear(
                &ds_pixels,
                grid_rows,
                grid_cols,
                back_rows,
                back_cols,
                image_sample_size,
            );
            let back_mask0 = resample_mask(
                &ds_mask,
                grid_rows,
                grid_cols,
                back_rows,
                back_cols,
                image_sample_size,
            );
            (
                back_shape_float,
                back_rows,
                back_cols,
                back_pixels0,
                back_mask0,
            )
        } else {
            (
                new_shape_float,
                grid_rows,
                grid_cols,
                ds_pixels.clone(),
                ds_mask.clone(),
            )
        };
    if back_rows == 0 || back_cols == 0 {
        return Err("image_sample_size produced an empty background grid".into());
    }

    let selem = disk_offsets(config.element_size);
    let masked: Vec<f64> = (0..back_rows * back_cols)
        .map(|i| if back_mask0[i] { back_pixels0[i] } else { 0.0 })
        .collect();
    let eroded = grey_erosion(&masked, back_rows, back_cols, &selem);
    let masked2: Vec<f64> = (0..back_rows * back_cols)
        .map(|i| if back_mask0[i] { eroded[i] } else { 0.0 })
        .collect();
    let dilated = grey_dilation(&masked2, back_rows, back_cols, &selem);

    let back_pixels_final: Vec<f64> = if image_sample_size < 1.0 {
        let mut out = vec![0.0; grid_rows * grid_cols];
        for r in 0..grid_rows {
            let ii = r as f64 * (back_shape_float[0] - 1.0) / (new_shape_float[0] - 1.0);
            for c in 0..grid_cols {
                let jj = c as f64 * (back_shape_float[1] - 1.0) / (new_shape_float[1] - 1.0);
                out[r * grid_cols + c] = sample_bilinear(&dilated, back_rows, back_cols, ii, jj);
            }
        }
        out
    } else {
        dilated
    };

    let ds_pixels_bgsub: Vec<f64> = ds_pixels
        .iter()
        .zip(&back_pixels_final)
        .map(|(&p, &b)| {
            let v = p - b;
            if v < 0.0 {
                0.0
            } else {
                v
            }
        })
        .collect();

    // Per-object-set full-resolution baseline means, independent of the
    // downsampling above.
    let mut obj_state: BTreeMap<String, ObjectState> = BTreeMap::new();
    for &(name, labels) in objects {
        let nobjects = labels.iter().copied().max().unwrap_or(0) as usize;
        let masked_labels: Vec<u32> = (0..size)
            .map(|i| if orig_mask[i] { labels[i] } else { 0 })
            .collect();
        let current_mean = per_object_means(pixels, &masked_labels, nobjects);
        let start_mean: Vec<f64> = current_mean
            .iter()
            .map(|&m| np_maximum(m, NP_EPS))
            .collect();
        obj_state.insert(
            name.to_owned(),
            ObjectState {
                masked_labels,
                nobjects,
                current_mean,
                start_mean,
            },
        );
    }

    let ng = config.granular_spectrum_length;
    let startmean = masked_mean(&ds_pixels_bgsub, &ds_mask);
    let mut ero: Vec<f64> = (0..grid_rows * grid_cols)
        .map(|i| if ds_mask[i] { ds_pixels_bgsub[i] } else { 0.0 })
        .collect();
    let mut currentmean = startmean;
    // Plain Python `max`, not `numpy.maximum` -- see [`py_max`]'s doc comment.
    let startmean_clamped = py_max(startmean, NP_EPS);

    let footprint1 = disk_offsets(1);
    let mut image_out = Vec::with_capacity(ng);
    let mut objects_out: BTreeMap<String, Vec<Vec<f64>>> = objects
        .iter()
        .map(|&(name, _)| (name.to_owned(), Vec::with_capacity(ng)))
        .collect();

    for _scale in 1..=ng {
        let prevmean = currentmean;
        let ero_masked: Vec<f64> = (0..grid_rows * grid_cols)
            .map(|i| if ds_mask[i] { ero[i] } else { 0.0 })
            .collect();
        ero = grey_erosion(&ero_masked, grid_rows, grid_cols, &footprint1);
        let rec =
            reconstruct_by_dilation(&ero, &ds_pixels_bgsub, grid_rows, grid_cols, &footprint1);
        currentmean = masked_mean(&rec, &ds_mask);
        let gs = (prevmean - currentmean) * 100.0 / startmean_clamped;
        image_out.push(gs);

        // Rescale `rec` (at the downsampled grid resolution) back up to the
        // ORIGINAL full resolution for the object-level means.
        let mut rec_full = vec![0.0; size];
        for r in 0..rows {
            let ii = r as f64 * (new_shape_float[0] - 1.0) / (orig_rows_f - 1.0);
            for c in 0..cols {
                let jj = c as f64 * (new_shape_float[1] - 1.0) / (orig_cols_f - 1.0);
                rec_full[r * cols + c] = sample_bilinear(&rec, grid_rows, grid_cols, ii, jj);
            }
        }

        for &(name, _labels) in objects {
            let state = obj_state
                .get_mut(name)
                .expect("inserted above for every name");
            let gss = if state.nobjects > 0 {
                let new_mean = per_object_means(&rec_full, &state.masked_labels, state.nobjects);
                let gss: Vec<f64> = state
                    .current_mean
                    .iter()
                    .zip(&new_mean)
                    .zip(&state.start_mean)
                    .map(|((&cur, &new), &start)| (cur - new) * 100.0 / start)
                    .collect();
                state.current_mean = new_mean;
                gss
            } else {
                Vec::new()
            };
            objects_out
                .get_mut(name)
                .expect("inserted above for every name")
                .push(gss);
        }
    }

    Ok(GranularityResult {
        image: image_out,
        objects: objects_out,
    })
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
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

    // Generated by scripts/reference/generate_granularity.py; real-oracle
    // fixtures from fixtures/granularity/. One #[test] per fixture file.
    include!("../../../fixtures/granularity/oracle_tests.rs");

    #[test]
    fn rejects_single_row_image() {
        let pixels = [1.0, 2.0, 3.0];
        let labels: [u32; 3] = [1, 1, 1];
        let err = measure(
            &pixels,
            &[("Cells", &labels[..])],
            None,
            1,
            3,
            &Config {
                subsample_size: 1.0,
                image_sample_size: 1.0,
                element_size: 1,
                granular_spectrum_length: 2,
            },
        )
        .unwrap_err();
        assert!(err.contains("1-row"));
    }

    #[test]
    fn rejects_mismatched_pixel_length() {
        let pixels = [1.0, 2.0, 3.0, 4.0];
        let labels: [u32; 4] = [1, 1, 1, 1];
        let err = measure(
            &pixels[..3],
            &[("Cells", &labels[..])],
            None,
            2,
            2,
            &Config {
                subsample_size: 1.0,
                image_sample_size: 1.0,
                element_size: 1,
                granular_spectrum_length: 2,
            },
        )
        .unwrap_err();
        assert!(err.contains("pixels length"));
    }
}

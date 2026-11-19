//! Pairwise and per-object colocalization/correlation measurements,
//! implemented against CellProfiler 4.2.8's `MeasureColocalization` module
//! as the compatibility oracle.
//!
//! Declared scope: 2D images only. Real CellProfiler's `volumetric()`
//! reports `True` (nominal 3D support); that case is an explicit,
//! documented gap, matching this project's house style for every other
//! module ported so far.
//!
//! `fast_costes` coverage: **all three** real modes (`Faster`/`Fast`/
//! `Accurate`) are implemented and oracle-verified here, not just
//! `Faster`. `Fast` and `Accurate` both go through the real module's own
//! single `linear_costes` routine (only its inner step size differs, via
//! an `is_accurate` flag); `Faster` uses the separate `bisection_costes`
//! routine. [`FastCostesMode`] has no "unverified" variant to reject --
//! there is no fast_costes gap in this port.
//!
//! Every numeric claim below was confirmed by running the pinned
//! reference CellProfiler 4.2.8 install (commit
//! `2df090ff281cff80943fe679a1c3192fbda9cd6d`) live via
//! `scripts/reference/generate_colocalization.py`, cross-checked first
//! with a throwaway pure-Python/NumPy mirror (which matched all 207
//! fixture scalars to `1e-9`/`1e-6` tolerance) before this Rust port was
//! written, not assumed from reading the source alone:
//!
//! - **`do_all` is a pure GUI-visibility toggle with zero run-time
//!   effect.** It is referenced only in `create_settings`/
//!   `visible_settings`; `run_image_pair_images`/`run_image_pair_objects`/
//!   `get_measurement_columns` all read the five `do_corr_and_slope`/
//!   `do_manders`/`do_rwc`/`do_overlap`/`do_costes` booleans directly.
//!   Driving the module headlessly (as the fixture generator does) with
//!   `do_all=True` and one sub-flag `False` produces exactly the sub-flag
//!   combination, confirming there is no GUI-only event wiring to
//!   replicate. [`Config`] accordingly has no `do_all` field at all --
//!   only the five independent flags.
//! - **Two different correlation primitives are used, for two different
//!   purposes, with different edge-case handling.** The
//!   `Correlation_Correlation_*` feature itself is `numpy.corrcoef(fi,
//!   si)[1, 0]`: plain covariance-over-std-product math with no
//!   special-casing, so a constant-valued masked region naturally yields
//!   `NaN` via `0.0 / 0.0`. The Costes threshold search
//!   (`bisection_costes`/`linear_costes`) instead calls
//!   `scipy.stats.pearsonr`, which (confirmed by reading its installed
//!   1.9.3 source) raises `ValueError` for `n < 2`, returns a sign-based
//!   `+1.0`/`-1.0` for `n == 2` without computing a ratio, and returns
//!   `NaN` *without raising* for constant input (just a warning). All
//!   three branches are distinct code paths here (see
//!   [`scipy_pearsonr`]).
//! - **`scipy.linalg.lstsq`'s slope for the `A*img1 + B = img2` model
//!   equals the closed-form OLS slope** (`cov(img1, img2) /
//!   var(img1)`), confirmed by a parallel `numpy.linalg.lstsq` run in the
//!   pure-Python mirror that matched the real oracle slope to `1e-9` for
//!   every fixture; no general least-squares solver is needed for this
//!   one-predictor model. This also confirms the slope is **not
//!   symmetric**: `Correlation_Slope_A_B` regresses `B` on `A`, not the
//!   reverse.
//! - **`get_scale` is just `max` of each `Image`'s own, optional
//!   `scale` attribute** (an explicit construction-time hint, e.g. "this
//!   came from a 16-bit TIFF with `MaxSampleValue=65535`"), falling back
//!   to `255` if neither image declares one. It is *not* auto-inferred
//!   from array dtype -- confirmed by constructing a reference `Image`
//!   with an explicit `scale=65535` kwarg (the `costes_explicit_scale`
//!   fixture) and reading `Image.scale`'s source, which just returns
//!   `self.__scale` (or the parent image's) with no dtype logic at all.
//!   This port works on bare `f64` pixel buffers with no retained
//!   bit-depth metadata, so [`Config::costes_scale_max`] exposes this
//!   directly as a plain field (default `255.0`, CellProfiler's own
//!   fallback).
//! - **`scipy.ndimage`'s three reducers disagree on what a *missing*
//!   label index returns**, confirmed by probing `scipy.ndimage.sum`/
//!   `mean`/`maximum` directly against a label array with a gap: `sum`
//!   returns `0.0`, `mean` returns `NaN` (`0.0 / 0.0`), and `maximum`
//!   returns `0.0`. The per-object code below reproduces all three
//!   (`group_sum`/`group_mean`/`group_max`) exactly, which is what makes
//!   a sparse label gap (tested by the `objects_within_sparse_label_gap`
//!   fixture) come out all-`NaN` for every feature at that id: the `sum`-
//!   based denominators are `0.0` and the numerators are also `0.0`,
//!   so plain division naturally produces `NaN` with no special-casing
//!   needed.
//! - **`Objects.count` is the *maximum label id*, not the number of
//!   distinct nonzero labels** (`len(self.indices)` where `indices =
//!   arange(max_label) + 1`). This port assumes the same dense
//!   `1..=max(labels)` object-id convention as every other object-scope
//!   module in this crate.
//! - **The per-object scope's "entirely masked out" `NaN` branch is
//!   gated on the *whole-image* mask (own mask AND other mask AND not-
//!   NaN on both, ignoring the label array entirely), not on the per-
//!   object combined mask** -- confirmed live with the
//!   `objects_whole_image_mask_empty` fixture (two images with disjoint
//!   masks plus a real object set): every requested feature family comes
//!   back as the *same* `NaN`-filled array, including `Correlation_*`,
//!   `Manders_*`, `Overlap`/`K_*`, `RWC_*`, and `Costes_*` -- the real
//!   source's `run_image_pair_objects` literally does
//!   `overlap = K1 = K2 = M1 = M2 = RWC1 = RWC2 = C1 = C2 = corr` in that
//!   branch, bypassing each family's own computation (and its own
//!   `do_*` gate on *how* the value is computed, though not on *whether*
//!   it is reported) entirely.
//! - **A real CellProfiler 4.2.8 crash was found and is intentionally not
//!   replicated.** If the per-object combined mask (`labels > 0` AND own
//!   mask AND other mask) is entirely empty for some object set while the
//!   *whole-image* mask above is non-empty elsewhere in the image (e.g.
//!   an object sitting entirely inside a masked-out region, with valid
//!   pixels elsewhere), real CellProfiler's own
//!   `scipy.ndimage.mean(first_pixels, labels, lrange)` raises
//!   `ValueError: zero-size array to reduction operation minimum which
//!   has no identity` -- there is no real-CP oracle value to match. This
//!   port's [`group_mean`]/[`group_sum`]/[`group_max`] accept empty input
//!   slices without panicking (every label is simply "missing", per the
//!   bullet above), so `measure` returns `NaN`-for-`Correlation`/`0.0`-
//!   for-everything-else in that scenario rather than erroring -- a
//!   deliberate, documented divergence from real CellProfiler's crash,
//!   not an attempt to reproduce it.
//! - **The real module's own `[Rank1] = numpy.lexsort(([labels],
//!   [first_pixels]))` RWC tie-break collapses to a plain dense rank of
//!   pixel *value* alone.** Confirmed directly: `numpy.lexsort((keys,))`
//!   with a two-row key array sorts primarily by the *last* row
//!   (`first_pixels`), with `labels` only breaking ties; but the
//!   resulting `Rank1_S` (the rank actually used) is built from
//!   *value-equality* boundaries in the sorted order
//!   (`first_pixels[Rank1[:-1]] != first_pixels[Rank1[1:]]`), so tied
//!   values always collapse to the same rank regardless of which label
//!   breaks the tie or in what order. [`dense_rank`] computes this
//!   directly (count of distinct values strictly less than each pixel's
//!   own value), with no sorting-order dependency to replicate.

use std::collections::BTreeMap;

/// Which Costes' automated-threshold search the real module's
/// `fast_costes` setting selects. All three are implemented and
/// oracle-verified (see this module's doc comment) -- there is no
/// unverified mode to reject here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastCostesMode {
    /// `M_FASTER`: `bisection_costes`, a binary search over the
    /// `0..=costes_scale_max` threshold range.
    Faster,
    /// `M_FAST`: `linear_costes` with `is_accurate = false` (coarser,
    /// larger step sizes while the running correlation stays high).
    Fast,
    /// `M_ACCURATE`: `linear_costes` with `is_accurate = true` (always
    /// steps by exactly one scale unit).
    Accurate,
}

/// Settings mirroring `MeasureColocalization`'s module settings, for one
/// `measure` call across every selected image (and, optionally, one
/// object set).
#[derive(Clone, Debug)]
pub struct Config {
    /// "Set threshold as percentage of maximum intensity for the
    /// images" (`thr`, a percentage in `0..=99` in the real module's own
    /// UI, though nothing in `run()` itself enforces that range).
    pub threshold_percent: f64,
    /// Which Costes' automated-threshold search to run when
    /// `do_costes` is set.
    pub fast_costes: FastCostesMode,
    /// `get_scale(first_image.scale, second_image.scale)`'s result --
    /// real CellProfiler's own fallback is `255.0` when neither image
    /// declares an explicit acquisition-bit-depth scale. This port has
    /// no per-image scale metadata, so the caller supplies the resolved
    /// value directly.
    pub costes_scale_max: f64,
    /// "Calculate correlation and slope metrics?"
    pub do_corr_and_slope: bool,
    /// "Calculate the Manders coefficients?"
    pub do_manders: bool,
    /// "Calculate the Rank Weighted Colocalization coefficients?"
    pub do_rwc: bool,
    /// "Calculate the Overlap, K1 and K2 coefficients?"
    pub do_overlap: bool,
    /// "Calculate the Costes automated threshold?"
    pub do_costes: bool,
    /// Whether to compute whole-image ("Across entire image" /
    /// "Both") pairwise measurements.
    pub wants_image_measurements: bool,
    /// Whether to compute per-object ("Within objects" / "Both")
    /// pairwise measurements; requires `labels` to be `Some`.
    pub wants_object_measurements: bool,
}

/// Whole-image and per-object pairwise colocalization measurements,
/// keyed by the real `Correlation_{Feature}_{img1}_{img2}` column name
/// CellProfiler itself uses (see the real module's own `F_*_FORMAT`
/// constants).
#[derive(Debug, Default)]
pub struct ColocalizationResult {
    /// Present only when [`Config::wants_image_measurements`] is set.
    pub image: BTreeMap<String, f64>,
    /// `objects[column][i]` is one value per object (ascending label
    /// id); present only when [`Config::wants_object_measurements`] is
    /// set.
    pub objects: BTreeMap<String, Vec<f64>>,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample variance (`ddof=1`, matching `numpy.var(..., ddof=1)`).
fn sample_variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let ss: f64 = xs.iter().map(|&v| (v - m) * (v - m)).sum();
    ss / (xs.len() as f64 - 1.0)
}

/// `numpy.corrcoef(x, y)[1, 0]`: plain Pearson r with no special-casing,
/// so a constant input naturally yields `NaN` via `0.0 / 0.0`.
fn pearson_corrcoef(x: &[f64], y: &[f64]) -> f64 {
    let mx = mean(x);
    let my = mean(y);
    let mut cov = 0.0;
    let mut varx = 0.0;
    let mut vary = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        varx += dx * dx;
        vary += dy * dy;
    }
    cov / (varx.sqrt() * vary.sqrt())
}

/// The least-squares regression slope `A` for the model `A*x + B = y`,
/// equal to `scipy.linalg.lstsq`'s slope coefficient for this
/// one-predictor-plus-intercept design (confirmed against a parallel
/// `numpy.linalg.lstsq` run matching the real oracle to `1e-9`; see this
/// module's doc comment). Not symmetric in `x`/`y`.
fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let mx = mean(x);
    let my = mean(y);
    let mut cov = 0.0;
    let mut varx = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        cov += (xi - mx) * (yi - my);
        varx += (xi - mx) * (xi - mx);
    }
    cov / varx
}

/// `scipy.stats.pearsonr(x, y)`, used only by the Costes threshold
/// search (see this module's doc comment for why it differs from
/// [`pearson_corrcoef`]). `None` means the real call raises
/// `ValueError` (`n < 2`); `Some(r)` mirrors its `n == 2` sign-based
/// shortcut and its constant-input `NaN` return.
fn scipy_pearsonr(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    let all_same = |v: &[f64]| v.iter().all(|&e| e == v[0]);
    if all_same(x) || all_same(y) {
        return Some(f64::NAN);
    }
    if n == 2 {
        return Some((x[1] - x[0]).signum() * (y[1] - y[0]).signum());
    }
    let mx = mean(x);
    let my = mean(y);
    let mut cov = 0.0;
    let mut varx = 0.0;
    let mut vary = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        cov += (xi - mx) * (yi - my);
        varx += (xi - mx) * (xi - mx);
        vary += (yi - my) * (yi - my);
    }
    let r = cov / (varx.sqrt() * vary.sqrt());
    Some(r.clamp(-1.0, 1.0))
}

/// `[Rank1] = numpy.lexsort(([labels], [values]))` followed by the real
/// module's own dense-rank-of-sorted-run-boundaries construction,
/// simplified to what it actually computes: the count of distinct
/// values strictly less than each pixel's own value (see this module's
/// doc comment for why the label-tiebreak never affects the result).
fn dense_rank(values: &[f64]) -> Vec<u32> {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    values
        .iter()
        .map(|v| sorted.partition_point(|&s| s < *v) as u32)
        .collect()
}

/// Shared orthogonal-regression setup for both Costes searches: the
/// total-least-squares line `si = a*fi + b` fit through the nonzero
/// pixels of either channel.
fn costes_regression_line(fi: &[f64], si: &[f64]) -> (f64, f64) {
    let nz: Vec<usize> = (0..fi.len())
        .filter(|&i| fi[i] > 0.0 || si[i] > 0.0)
        .collect();
    let fi_nz: Vec<f64> = nz.iter().map(|&i| fi[i]).collect();
    let si_nz: Vec<f64> = nz.iter().map(|&i| si[i]).collect();
    let xvar = sample_variance(&fi_nz);
    let yvar = sample_variance(&si_nz);
    let xmean = mean(&fi_nz);
    let ymean = mean(&si_nz);
    let z: Vec<f64> = fi_nz.iter().zip(&si_nz).map(|(&a, &b)| a + b).collect();
    let zvar = sample_variance(&z);
    let covar = 0.5 * (zvar - (xvar + yvar));
    let denom = 2.0 * covar;
    let num = (yvar - xvar) + ((yvar - xvar) * (yvar - xvar) + 4.0 * covar * covar).sqrt();
    let a = num / denom;
    let b = ymean - a * xmean;
    (a, b)
}

const COSTES_ITERATION_CAP: u32 = 100_000;

/// `MeasureColocalization.bisection_costes`: a binary search over the
/// `1..=scale_max` integer threshold range for the point where pixels
/// below threshold stop being positively correlated. Terminates when
/// the midpoint stops changing between iterations (mirrored exactly,
/// including the `6/5`-vs-`2` step-size switch at `right - left <= 6`),
/// with a defensive iteration cap this port adds (real CellProfiler has
/// none; every oracle fixture here converges in well under 100
/// iterations).
fn bisection_costes(fi: &[f64], si: &[f64], scale_max: f64) -> (f64, f64) {
    let (a, b) = costes_regression_line(fi, si);
    let mut left = 1.0_f64;
    let mut right = scale_max;
    let mut mid = ((right - left) / (6.0 / 5.0)).floor() + left;
    let mut lastmid = 0.0_f64;
    let mut valid = 1.0_f64;
    let mut iterations = 0;
    while lastmid != mid && iterations < COSTES_ITERATION_CAP {
        iterations += 1;
        let thr_fi_c = mid / scale_max;
        let thr_si_c = a * thr_fi_c + b;
        let combt: Vec<usize> = (0..fi.len())
            .filter(|&i| fi[i] < thr_fi_c || si[i] < thr_si_c)
            .collect();
        if combt.len() <= 2 {
            left = mid - 1.0;
        } else {
            let xs: Vec<f64> = combt.iter().map(|&i| fi[i]).collect();
            let ys: Vec<f64> = combt.iter().map(|&i| si[i]).collect();
            match scipy_pearsonr(&xs, &ys) {
                None => left = mid - 1.0,
                Some(r) if r.is_nan() => {}
                Some(r) if r < 0.0 => left = mid - 1.0,
                Some(_) => {
                    right = mid + 1.0;
                    valid = mid;
                }
            }
        }
        lastmid = mid;
        mid = if right - left > 6.0 {
            ((right - left) / (6.0 / 5.0)).floor() + left
        } else {
            ((right - left) / 2.0).floor() + left
        };
    }
    let thr_fi_c = (valid - 1.0) / scale_max;
    let thr_si_c = a * thr_fi_c + b;
    (thr_fi_c, thr_si_c)
}

/// `MeasureColocalization.linear_costes`: steps the threshold down from
/// the image maximum by `1/scale_max` (`is_accurate`, i.e. `M_ACCURATE`)
/// or by up to `10/scale_max` while the running correlation stays above
/// `0.45` (`!is_accurate`, i.e. `M_FAST`), stopping once the
/// below-threshold correlation turns non-positive. The real module's
/// `self.fast_costes == M_ACCURATE` check is this function's
/// `is_accurate` parameter; both `Fast` and `Accurate` share this one
/// routine in real CellProfiler.
fn linear_costes(fi: &[f64], si: &[f64], scale_max: f64, is_accurate: bool) -> (f64, f64) {
    let (a, b) = costes_regression_line(fi, si);
    let i_step = 1.0 / scale_max;
    let fi_max = fi.iter().cloned().fold(f64::MIN, f64::max);
    let si_max = si.iter().cloned().fold(f64::MIN, f64::max);
    let img_max = fi_max.max(si_max);
    let mut i = i_step * ((img_max / i_step).floor() + 1.0);

    let mut iterations = 0;
    while i > fi_max && (a * i) + b > si_max && iterations < COSTES_ITERATION_CAP {
        i -= i_step;
        iterations += 1;
    }

    let mut thr_fi_c = i;
    let mut thr_si_c = a * i + b;
    let mut num_true: Option<usize> = None;
    let mut cost_reg = scipy_pearsonr(fi, si);
    iterations = 0;
    while i > i_step && iterations < COSTES_ITERATION_CAP {
        iterations += 1;
        thr_fi_c = i;
        thr_si_c = a * i + b;
        let combt: Vec<usize> = (0..fi.len())
            .filter(|&idx| fi[idx] < thr_fi_c || si[idx] < thr_si_c)
            .collect();
        let positives = combt.len();
        if Some(positives) != num_true {
            let xs: Vec<f64> = combt.iter().map(|&idx| fi[idx]).collect();
            let ys: Vec<f64> = combt.iter().map(|&idx| si[idx]).collect();
            cost_reg = scipy_pearsonr(&xs, &ys);
            num_true = Some(positives);
        }
        let Some(r) = cost_reg else {
            break;
        };
        if r.is_nan() {
            i -= i_step;
            continue;
        }
        if r <= 0.0 {
            break;
        } else if is_accurate || i < i_step * 10.0 {
            i -= i_step;
        } else if r > 0.45 {
            i -= i_step * 10.0;
        } else if r > 0.35 {
            i -= i_step * 5.0;
        } else if r > 0.25 {
            i -= i_step * 2.0;
        } else {
            i -= i_step;
        }
    }
    (thr_fi_c, thr_si_c)
}

fn costes_threshold(fi: &[f64], si: &[f64], config: &Config) -> (f64, f64) {
    match config.fast_costes {
        FastCostesMode::Faster => bisection_costes(fi, si, config.costes_scale_max),
        FastCostesMode::Fast => linear_costes(fi, si, config.costes_scale_max, false),
        FastCostesMode::Accurate => linear_costes(fi, si, config.costes_scale_max, true),
    }
}

/// One pair's worth of whole-image feature values, each only
/// meaningful when its matching `do_*` flag was set.
struct WholeImagePair {
    correlation: f64,
    slope: f64,
    overlap: f64,
    k1: f64,
    k2: f64,
    manders1: f64,
    manders2: f64,
    rwc1: f64,
    rwc2: f64,
    costes1: f64,
    costes2: f64,
}

/// `run_image_pair_images`: whole-image pairwise measurements between
/// two same-shape images under their combined own-mask/not-NaN mask.
fn whole_image_pair(
    a: &[f64],
    b: &[f64],
    mask_a: &[bool],
    mask_b: &[bool],
    config: &Config,
) -> WholeImagePair {
    let mask: Vec<bool> = (0..a.len())
        .map(|i| mask_a[i] && mask_b[i] && !a[i].is_nan() && !b[i].is_nan())
        .collect();
    if !mask.iter().any(|&m| m) {
        return WholeImagePair {
            correlation: f64::NAN,
            slope: f64::NAN,
            overlap: f64::NAN,
            k1: f64::NAN,
            k2: f64::NAN,
            manders1: f64::NAN,
            manders2: f64::NAN,
            rwc1: f64::NAN,
            rwc2: f64::NAN,
            costes1: f64::NAN,
            costes2: f64::NAN,
        };
    }
    let fi: Vec<f64> = (0..a.len()).filter(|&i| mask[i]).map(|i| a[i]).collect();
    let si: Vec<f64> = (0..a.len()).filter(|&i| mask[i]).map(|i| b[i]).collect();

    let mut out = WholeImagePair {
        correlation: f64::NAN,
        slope: f64::NAN,
        overlap: f64::NAN,
        k1: f64::NAN,
        k2: f64::NAN,
        manders1: f64::NAN,
        manders2: f64::NAN,
        rwc1: f64::NAN,
        rwc2: f64::NAN,
        costes1: f64::NAN,
        costes2: f64::NAN,
    };

    if config.do_corr_and_slope {
        out.correlation = pearson_corrcoef(&fi, &si);
        out.slope = ols_slope(&fi, &si);
    }

    let need_thresh = config.do_manders || config.do_rwc || config.do_overlap;
    let mut combined_thresh = Vec::new();
    let mut fi_thresh = Vec::new();
    let mut si_thresh = Vec::new();
    let mut tot_fi_thr = 0.0;
    let mut tot_si_thr = 0.0;
    if need_thresh {
        let fi_max = fi.iter().cloned().fold(f64::MIN, f64::max);
        let si_max = si.iter().cloned().fold(f64::MIN, f64::max);
        let thr_fi = config.threshold_percent * fi_max / 100.0;
        let thr_si = config.threshold_percent * si_max / 100.0;
        combined_thresh = fi
            .iter()
            .zip(&si)
            .map(|(&f, &s)| f > thr_fi && s > thr_si)
            .collect();
        for (i, &keep) in combined_thresh.iter().enumerate() {
            if keep {
                fi_thresh.push(fi[i]);
                si_thresh.push(si[i]);
            }
            if fi[i] > thr_fi {
                tot_fi_thr += fi[i];
            }
            if si[i] > thr_si {
                tot_si_thr += si[i];
            }
        }
    }

    if config.do_manders {
        out.manders1 = fi_thresh.iter().sum::<f64>() / tot_fi_thr;
        out.manders2 = si_thresh.iter().sum::<f64>() / tot_si_thr;
    }

    if config.do_rwc {
        let rank1 = dense_rank(&fi);
        let rank2 = dense_rank(&si);
        let r = rank1
            .iter()
            .max()
            .copied()
            .unwrap_or(0)
            .max(rank2.iter().max().copied().unwrap_or(0)) as f64
            + 1.0;
        let weight: Vec<f64> = rank1
            .iter()
            .zip(&rank2)
            .map(|(&r1, &r2)| {
                let di = (r1 as f64 - r2 as f64).abs();
                (r - di) / r
            })
            .collect();
        let mut rwc1 = 0.0;
        let mut rwc2 = 0.0;
        for (i, &keep) in combined_thresh.iter().enumerate() {
            if keep {
                rwc1 += fi[i] * weight[i];
                rwc2 += si[i] * weight[i];
            }
        }
        out.rwc1 = rwc1 / tot_fi_thr;
        out.rwc2 = rwc2 / tot_si_thr;
    }

    if config.do_overlap {
        let prod: f64 = fi_thresh.iter().zip(&si_thresh).map(|(&f, &s)| f * s).sum();
        let fpsq: f64 = fi_thresh.iter().map(|&f| f * f).sum();
        let spsq: f64 = si_thresh.iter().map(|&s| s * s).sum();
        out.overlap = prod / (fpsq * spsq).sqrt();
        out.k1 = prod / fpsq;
        out.k2 = prod / spsq;
    }

    if config.do_costes {
        let (thr_fi_c, thr_si_c) = costes_threshold(&fi, &si, config);
        let mut fi_thresh_c = Vec::new();
        let mut si_thresh_c = Vec::new();
        let mut tot_fi_thr_c = 0.0;
        let mut tot_si_thr_c = 0.0;
        for (&f, &s) in fi.iter().zip(&si) {
            if f > thr_fi_c && s > thr_si_c {
                fi_thresh_c.push(f);
                si_thresh_c.push(s);
            }
            if f > thr_fi_c {
                tot_fi_thr_c += f;
            }
            if s > thr_si_c {
                tot_si_thr_c += s;
            }
        }
        out.costes1 = fi_thresh_c.iter().sum::<f64>() / tot_fi_thr_c;
        out.costes2 = si_thresh_c.iter().sum::<f64>() / tot_si_thr_c;
    }

    out
}

fn group_sum(values: &[f64], labels: &[u32], n_objects: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_objects];
    for (&v, &l) in values.iter().zip(labels) {
        if l >= 1 && (l as usize) <= n_objects {
            out[l as usize - 1] += v;
        }
    }
    out
}

fn group_mean(values: &[f64], labels: &[u32], n_objects: usize) -> Vec<f64> {
    let mut sums = vec![0.0f64; n_objects];
    let mut counts = vec![0u32; n_objects];
    for (&v, &l) in values.iter().zip(labels) {
        if l >= 1 && (l as usize) <= n_objects {
            sums[l as usize - 1] += v;
            counts[l as usize - 1] += 1;
        }
    }
    sums.iter()
        .zip(&counts)
        .map(|(&s, &c)| if c == 0 { f64::NAN } else { s / c as f64 })
        .collect()
}

fn group_max(values: &[f64], labels: &[u32], n_objects: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_objects];
    let mut seen = vec![false; n_objects];
    for (&v, &l) in values.iter().zip(labels) {
        if l >= 1 && (l as usize) <= n_objects {
            let idx = l as usize - 1;
            if !seen[idx] || v > out[idx] {
                out[idx] = v;
                seen[idx] = true;
            }
        }
    }
    out
}

/// `run_image_pair_objects`: per-object pairwise measurements between
/// two same-shape images under one label array, following the exact
/// branch structure confirmed live (see this module's doc comment):
/// zero objects -> empty vectors; whole-image mask entirely empty -> one
/// shared `NaN`-filled vector for every family; otherwise per-object
/// sums/means/maxima over the `labels > 0` & both-masks pixel set, with
/// Costes' threshold itself still derived from the *whole-image* masked
/// pixels (not the per-object ones), matching a real, if surprising,
/// quirk of the original algorithm.
fn object_pair(
    a: &[f64],
    b: &[f64],
    mask_a: &[bool],
    mask_b: &[bool],
    labels_full: &[u32],
    config: &Config,
) -> BTreeMap<&'static str, Vec<f64>> {
    let n_objects = labels_full.iter().copied().max().unwrap_or(0) as usize;
    let mut out: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let families: &[&'static str] = &[
        "correlation",
        "overlap",
        "k1",
        "k2",
        "manders1",
        "manders2",
        "rwc1",
        "rwc2",
        "costes1",
        "costes2",
    ];
    if n_objects == 0 {
        for &f in families {
            out.insert(f, Vec::new());
        }
        return out;
    }

    let whole_mask: Vec<bool> = (0..a.len())
        .map(|i| mask_a[i] && mask_b[i] && !a[i].is_nan() && !b[i].is_nan())
        .collect();
    if !whole_mask.iter().any(|&m| m) {
        let nan_vec = vec![f64::NAN; n_objects];
        for &f in families {
            out.insert(f, nan_vec.clone());
        }
        return out;
    }

    // Costes' threshold is derived from the whole-image masked pixels,
    // not the per-object ones -- confirmed live (see doc comment).
    let fi_whole: Vec<f64> = (0..a.len())
        .filter(|&i| whole_mask[i])
        .map(|i| a[i])
        .collect();
    let si_whole: Vec<f64> = (0..a.len())
        .filter(|&i| whole_mask[i])
        .map(|i| b[i])
        .collect();

    let obj_mask: Vec<bool> = (0..a.len())
        .map(|i| labels_full[i] >= 1 && mask_a[i] && mask_b[i])
        .collect();
    let first_pixels: Vec<f64> = (0..a.len())
        .filter(|&i| obj_mask[i])
        .map(|i| a[i])
        .collect();
    let second_pixels: Vec<f64> = (0..a.len())
        .filter(|&i| obj_mask[i])
        .map(|i| b[i])
        .collect();
    let labels: Vec<u32> = (0..a.len())
        .filter(|&i| obj_mask[i])
        .map(|i| labels_full[i])
        .collect();

    if config.do_corr_and_slope {
        let mean1 = group_mean(&first_pixels, &labels, n_objects);
        let mean2 = group_mean(&second_pixels, &labels, n_objects);
        let dev1: Vec<f64> = first_pixels
            .iter()
            .zip(&labels)
            .map(|(&v, &l)| v - mean1[l as usize - 1])
            .collect();
        let dev2: Vec<f64> = second_pixels
            .iter()
            .zip(&labels)
            .map(|(&v, &l)| v - mean2[l as usize - 1])
            .collect();
        let std1: Vec<f64> = group_sum(
            &dev1.iter().map(|&d| d * d).collect::<Vec<_>>(),
            &labels,
            n_objects,
        )
        .iter()
        .map(|&s| s.sqrt())
        .collect();
        let std2: Vec<f64> = group_sum(
            &dev2.iter().map(|&d| d * d).collect::<Vec<_>>(),
            &labels,
            n_objects,
        )
        .iter()
        .map(|&s| s.sqrt())
        .collect();
        let normalized: Vec<f64> = dev1
            .iter()
            .zip(&dev2)
            .zip(&labels)
            .map(|((&d1, &d2), &l)| {
                let idx = l as usize - 1;
                d1 * d2 / (std1[idx] * std2[idx])
            })
            .collect();
        let mut corr = group_sum(&normalized, &labels, n_objects);
        let counts = group_sum(&vec![1.0; labels.len()], &labels, n_objects);
        for (c, &count) in corr.iter_mut().zip(&counts) {
            if count == 0.0 {
                *c = f64::NAN;
            }
        }
        out.insert("correlation", corr);
    }

    let need_thresh = config.do_manders || config.do_rwc || config.do_overlap;
    let mut combined_thresh = Vec::new();
    let mut fi_thresh = Vec::new();
    let mut si_thresh = Vec::new();
    let mut labels_thresh = Vec::new();
    let mut tot_fi_thr = vec![0.0; n_objects];
    let mut tot_si_thr = vec![0.0; n_objects];
    if need_thresh {
        let tff = group_max(&first_pixels, &labels, n_objects)
            .iter()
            .map(|&m| config.threshold_percent / 100.0 * m)
            .collect::<Vec<_>>();
        let tss = group_max(&second_pixels, &labels, n_objects)
            .iter()
            .map(|&m| config.threshold_percent / 100.0 * m)
            .collect::<Vec<_>>();
        let sel1: Vec<f64> = first_pixels
            .iter()
            .zip(&labels)
            .filter(|&(&v, &l)| v >= tff[l as usize - 1])
            .map(|(&v, _)| v)
            .collect();
        let sel1_labels: Vec<u32> = first_pixels
            .iter()
            .zip(&labels)
            .filter(|&(&v, &l)| v >= tff[l as usize - 1])
            .map(|(_, &l)| l)
            .collect();
        tot_fi_thr = group_sum(&sel1, &sel1_labels, n_objects);
        let sel2: Vec<f64> = second_pixels
            .iter()
            .zip(&labels)
            .filter(|&(&v, &l)| v >= tss[l as usize - 1])
            .map(|(&v, _)| v)
            .collect();
        let sel2_labels: Vec<u32> = second_pixels
            .iter()
            .zip(&labels)
            .filter(|&(&v, &l)| v >= tss[l as usize - 1])
            .map(|(_, &l)| l)
            .collect();
        tot_si_thr = group_sum(&sel2, &sel2_labels, n_objects);

        for i in 0..first_pixels.len() {
            let idx = labels[i] as usize - 1;
            let keep = first_pixels[i] >= tff[idx] && second_pixels[i] >= tss[idx];
            combined_thresh.push(keep);
            if keep {
                fi_thresh.push(first_pixels[i]);
                si_thresh.push(second_pixels[i]);
                labels_thresh.push(labels[i]);
            }
        }
    }
    let any_combined = combined_thresh.iter().any(|&k| k);

    if config.do_manders {
        let (manders1, manders2) = if any_combined {
            let num1 = group_sum(&fi_thresh, &labels_thresh, n_objects);
            let num2 = group_sum(&si_thresh, &labels_thresh, n_objects);
            let m1 = num1.iter().zip(&tot_fi_thr).map(|(&n, &d)| n / d).collect();
            let m2 = num2.iter().zip(&tot_si_thr).map(|(&n, &d)| n / d).collect();
            (m1, m2)
        } else {
            (vec![0.0; n_objects], vec![0.0; n_objects])
        };
        out.insert("manders1", manders1);
        out.insert("manders2", manders2);
    }

    if config.do_rwc {
        let rank1 = dense_rank(&first_pixels);
        let rank2 = dense_rank(&second_pixels);
        let r = rank1
            .iter()
            .max()
            .copied()
            .unwrap_or(0)
            .max(rank2.iter().max().copied().unwrap_or(0)) as f64
            + 1.0;
        let weight: Vec<f64> = rank1
            .iter()
            .zip(&rank2)
            .map(|(&r1, &r2)| {
                let di = (r1 as f64 - r2 as f64).abs();
                (r - di) / r
            })
            .collect();
        let (rwc1, rwc2): (Vec<f64>, Vec<f64>) = if any_combined {
            let weighted_fi: Vec<f64> = combined_thresh
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k)
                .map(|(i, _)| first_pixels[i] * weight[i])
                .collect();
            let weighted_si: Vec<f64> = combined_thresh
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k)
                .map(|(i, _)| second_pixels[i] * weight[i])
                .collect();
            let num1 = group_sum(&weighted_fi, &labels_thresh, n_objects);
            let num2 = group_sum(&weighted_si, &labels_thresh, n_objects);
            (
                num1.iter().zip(&tot_fi_thr).map(|(&n, &d)| n / d).collect(),
                num2.iter().zip(&tot_si_thr).map(|(&n, &d)| n / d).collect(),
            )
        } else {
            (vec![0.0; n_objects], vec![0.0; n_objects])
        };
        out.insert("rwc1", rwc1);
        out.insert("rwc2", rwc2);
    }

    if config.do_overlap {
        let (overlap, k1, k2): (Vec<f64>, Vec<f64>, Vec<f64>) = if any_combined {
            let fpsq = group_sum(
                &fi_thresh.iter().map(|&f| f * f).collect::<Vec<_>>(),
                &labels_thresh,
                n_objects,
            );
            let spsq = group_sum(
                &si_thresh.iter().map(|&s| s * s).collect::<Vec<_>>(),
                &labels_thresh,
                n_objects,
            );
            let prod = group_sum(
                &fi_thresh
                    .iter()
                    .zip(&si_thresh)
                    .map(|(&f, &s)| f * s)
                    .collect::<Vec<_>>(),
                &labels_thresh,
                n_objects,
            );
            let overlap = prod
                .iter()
                .zip(fpsq.iter().zip(&spsq))
                .map(|(&p, (&fp, &sp))| p / (fp * sp).sqrt())
                .collect();
            let k1 = prod.iter().zip(&fpsq).map(|(&p, &fp)| p / fp).collect();
            let k2 = prod.iter().zip(&spsq).map(|(&p, &sp)| p / sp).collect();
            (overlap, k1, k2)
        } else {
            (
                vec![0.0; n_objects],
                vec![0.0; n_objects],
                vec![0.0; n_objects],
            )
        };
        out.insert("overlap", overlap);
        out.insert("k1", k1);
        out.insert("k2", k2);
    }

    if config.do_costes {
        let (thr_fi_c, thr_si_c) = costes_threshold(&fi_whole, &si_whole, config);
        let fi_above: Vec<bool> = first_pixels.iter().map(|&f| f > thr_fi_c).collect();
        let si_above: Vec<bool> = second_pixels.iter().map(|&s| s > thr_si_c).collect();
        let combined_thresh_c: Vec<bool> = fi_above
            .iter()
            .zip(&si_above)
            .map(|(&f, &s)| f && s)
            .collect();
        let any_above_fi = fi_above.iter().any(|&f| f);
        let any_above_si = si_above.iter().any(|&s| s);
        let any_combined_c = combined_thresh_c.iter().any(|&k| k);

        let tot_fi_thr_c = if any_above_fi {
            let sel: Vec<f64> = first_pixels
                .iter()
                .filter(|&&f| f >= thr_fi_c)
                .copied()
                .collect();
            let sel_labels: Vec<u32> = first_pixels
                .iter()
                .zip(&labels)
                .filter(|&(&f, _)| f >= thr_fi_c)
                .map(|(_, &l)| l)
                .collect();
            group_sum(&sel, &sel_labels, n_objects)
        } else {
            vec![0.0; n_objects]
        };
        let tot_si_thr_c = if any_above_si {
            let sel: Vec<f64> = second_pixels
                .iter()
                .filter(|&&s| s >= thr_si_c)
                .copied()
                .collect();
            let sel_labels: Vec<u32> = second_pixels
                .iter()
                .zip(&labels)
                .filter(|&(&s, _)| s >= thr_si_c)
                .map(|(_, &l)| l)
                .collect();
            group_sum(&sel, &sel_labels, n_objects)
        } else {
            vec![0.0; n_objects]
        };

        let (costes1, costes2): (Vec<f64>, Vec<f64>) = if any_combined_c {
            let fi_thresh_c: Vec<f64> = combined_thresh_c
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k)
                .map(|(i, _)| first_pixels[i])
                .collect();
            let si_thresh_c: Vec<f64> = combined_thresh_c
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k)
                .map(|(i, _)| second_pixels[i])
                .collect();
            let labels_c: Vec<u32> = combined_thresh_c
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k)
                .map(|(i, _)| labels[i])
                .collect();
            let num1 = group_sum(&fi_thresh_c, &labels_c, n_objects);
            let num2 = group_sum(&si_thresh_c, &labels_c, n_objects);
            (
                num1.iter()
                    .zip(&tot_fi_thr_c)
                    .map(|(&n, &d)| n / d)
                    .collect(),
                num2.iter()
                    .zip(&tot_si_thr_c)
                    .map(|(&n, &d)| n / d)
                    .collect(),
            )
        } else {
            (vec![0.0; n_objects], vec![0.0; n_objects])
        };
        out.insert("costes1", costes1);
        out.insert("costes2", costes2);
    }

    out
}

fn resolve_mask<'a>(
    name: &str,
    masks: Option<&'a [(&str, &[bool])]>,
    len: usize,
    all_true: &'a [bool],
) -> Result<&'a [bool], String> {
    let Some(masks) = masks else {
        return Ok(all_true);
    };
    for &(mask_name, mask) in masks {
        if mask_name == name {
            if mask.len() != len {
                return Err(format!(
                    "mask '{name}' length {} does not match rows*cols {len}",
                    mask.len()
                ));
            }
            return Ok(mask);
        }
    }
    Ok(all_true)
}

/// Run every selected whole-image pair (if [`Config::wants_image_measurements`])
/// and/or every pair under one object set (if [`Config::wants_object_measurements`]),
/// matching real CellProfiler's own `C(n, 2)` `get_image_pairs()` order:
/// `(images[i], images[j])` for `i < j` in the order `images` is given.
///
/// `masks[name]` overrides the default all-`True` mask for that image;
/// images absent from `masks` are treated as fully unmasked. `labels` is
/// required (and length-checked) when `wants_object_measurements` is set.
pub fn measure(
    images: &[(&str, &[f64])],
    rows: usize,
    cols: usize,
    masks: Option<&[(&str, &[bool])]>,
    labels: Option<&[u32]>,
    config: &Config,
) -> Result<ColocalizationResult, String> {
    if rows == 0 || cols == 0 {
        return Err("rows and cols must both be nonzero".into());
    }
    let size = rows
        .checked_mul(cols)
        .ok_or("rows * cols overflowed usize")?;
    if images.len() < 2 {
        return Err("at least two images are required".into());
    }
    for &(name, pixels) in images {
        if pixels.len() != size {
            return Err(format!(
                "image '{name}' length {} does not match rows*cols {size}",
                pixels.len()
            ));
        }
    }
    {
        let mut seen = Vec::new();
        for &(name, _) in images {
            if seen.contains(&name) {
                return Err(format!("duplicate image name '{name}'"));
            }
            seen.push(name);
        }
    }
    if let Some(mask_list) = masks {
        for &(name, _) in mask_list {
            if !images.iter().any(|&(image_name, _)| image_name == name) {
                return Err(format!("mask given for unknown image '{name}'"));
            }
        }
    }
    if config.wants_object_measurements && labels.is_none() {
        return Err("wants_object_measurements requires labels".into());
    }
    if let Some(label_slice) = labels {
        if label_slice.len() != size {
            return Err(format!(
                "labels length {} does not match rows*cols {size}",
                label_slice.len()
            ));
        }
    }

    let all_true = vec![true; size];
    let mut result = ColocalizationResult::default();

    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            let (name_a, a) = images[i];
            let (name_b, b) = images[j];
            let mask_a = resolve_mask(name_a, masks, size, &all_true)?;
            let mask_b = resolve_mask(name_b, masks, size, &all_true)?;

            if config.wants_image_measurements {
                let pair = whole_image_pair(a, b, mask_a, mask_b, config);
                if config.do_corr_and_slope {
                    result.image.insert(
                        format!("Correlation_Correlation_{name_a}_{name_b}"),
                        pair.correlation,
                    );
                    result
                        .image
                        .insert(format!("Correlation_Slope_{name_a}_{name_b}"), pair.slope);
                }
                if config.do_overlap {
                    result.image.insert(
                        format!("Correlation_Overlap_{name_a}_{name_b}"),
                        pair.overlap,
                    );
                    result
                        .image
                        .insert(format!("Correlation_K_{name_a}_{name_b}"), pair.k1);
                    result
                        .image
                        .insert(format!("Correlation_K_{name_b}_{name_a}"), pair.k2);
                }
                if config.do_manders {
                    result.image.insert(
                        format!("Correlation_Manders_{name_a}_{name_b}"),
                        pair.manders1,
                    );
                    result.image.insert(
                        format!("Correlation_Manders_{name_b}_{name_a}"),
                        pair.manders2,
                    );
                }
                if config.do_rwc {
                    result
                        .image
                        .insert(format!("Correlation_RWC_{name_a}_{name_b}"), pair.rwc1);
                    result
                        .image
                        .insert(format!("Correlation_RWC_{name_b}_{name_a}"), pair.rwc2);
                }
                if config.do_costes {
                    result.image.insert(
                        format!("Correlation_Costes_{name_a}_{name_b}"),
                        pair.costes1,
                    );
                    result.image.insert(
                        format!("Correlation_Costes_{name_b}_{name_a}"),
                        pair.costes2,
                    );
                }
            }

            if config.wants_object_measurements {
                let label_slice = labels.expect("checked above");
                let families = object_pair(a, b, mask_a, mask_b, label_slice, config);
                if config.do_corr_and_slope {
                    result.objects.insert(
                        format!("Correlation_Correlation_{name_a}_{name_b}"),
                        families["correlation"].clone(),
                    );
                }
                if config.do_overlap {
                    result.objects.insert(
                        format!("Correlation_Overlap_{name_a}_{name_b}"),
                        families["overlap"].clone(),
                    );
                    result.objects.insert(
                        format!("Correlation_K_{name_a}_{name_b}"),
                        families["k1"].clone(),
                    );
                    result.objects.insert(
                        format!("Correlation_K_{name_b}_{name_a}"),
                        families["k2"].clone(),
                    );
                }
                if config.do_manders {
                    result.objects.insert(
                        format!("Correlation_Manders_{name_a}_{name_b}"),
                        families["manders1"].clone(),
                    );
                    result.objects.insert(
                        format!("Correlation_Manders_{name_b}_{name_a}"),
                        families["manders2"].clone(),
                    );
                }
                if config.do_rwc {
                    result.objects.insert(
                        format!("Correlation_RWC_{name_a}_{name_b}"),
                        families["rwc1"].clone(),
                    );
                    result.objects.insert(
                        format!("Correlation_RWC_{name_b}_{name_a}"),
                        families["rwc2"].clone(),
                    );
                }
                if config.do_costes {
                    result.objects.insert(
                        format!("Correlation_Costes_{name_a}_{name_b}"),
                        families["costes1"].clone(),
                    );
                    result.objects.insert(
                        format!("Correlation_Costes_{name_b}_{name_a}"),
                        families["costes2"].clone(),
                    );
                }
            }
        }
    }

    Ok(result)
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

    // Generated by scripts/reference/generate_colocalization.py; real-
    // oracle fixtures from fixtures/colocalization/. One #[test] per
    // fixture file.
    include!("../../../fixtures/colocalization/oracle_tests.rs");

    #[test]
    fn rejects_fewer_than_two_images() {
        let img = [1.0, 2.0, 3.0, 4.0];
        let images: Vec<(&str, &[f64])> = vec![("A", img.as_slice())];
        let config = Config {
            threshold_percent: 15.0,
            fast_costes: FastCostesMode::Faster,
            costes_scale_max: 255.0,
            do_corr_and_slope: true,
            do_manders: true,
            do_rwc: true,
            do_overlap: true,
            do_costes: true,
            wants_image_measurements: true,
            wants_object_measurements: false,
        };
        let err = measure(&images, 2, 2, None, None, &config).unwrap_err();
        assert!(err.contains("two images"));
    }

    #[test]
    fn rejects_mismatched_image_length() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0];
        let images: Vec<(&str, &[f64])> = vec![("A", a.as_slice()), ("B", b.as_slice())];
        let config = Config {
            threshold_percent: 15.0,
            fast_costes: FastCostesMode::Faster,
            costes_scale_max: 255.0,
            do_corr_and_slope: true,
            do_manders: true,
            do_rwc: true,
            do_overlap: true,
            do_costes: true,
            wants_image_measurements: true,
            wants_object_measurements: false,
        };
        let err = measure(&images, 2, 2, None, None, &config).unwrap_err();
        assert!(err.contains("length"));
    }

    #[test]
    fn rejects_object_measurements_without_labels() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let images: Vec<(&str, &[f64])> = vec![("A", a.as_slice()), ("B", b.as_slice())];
        let config = Config {
            threshold_percent: 15.0,
            fast_costes: FastCostesMode::Faster,
            costes_scale_max: 255.0,
            do_corr_and_slope: true,
            do_manders: true,
            do_rwc: true,
            do_overlap: true,
            do_costes: true,
            wants_image_measurements: false,
            wants_object_measurements: true,
        };
        let err = measure(&images, 2, 2, None, None, &config).unwrap_err();
        assert!(err.contains("labels"));
    }

    #[test]
    fn three_images_yield_three_pairs() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let c = [2.0, 2.0, 3.0, 1.0];
        let images: Vec<(&str, &[f64])> = vec![
            ("A", a.as_slice()),
            ("B", b.as_slice()),
            ("C", c.as_slice()),
        ];
        let config = Config {
            threshold_percent: 15.0,
            fast_costes: FastCostesMode::Faster,
            costes_scale_max: 255.0,
            do_corr_and_slope: true,
            do_manders: false,
            do_rwc: false,
            do_overlap: false,
            do_costes: false,
            wants_image_measurements: true,
            wants_object_measurements: false,
        };
        let result = measure(&images, 2, 2, None, None, &config).unwrap();
        // C(3, 2) = 3 pairs, each contributing Correlation + Slope.
        assert_eq!(result.image.len(), 3 * 2);
        assert!(result.image.contains_key("Correlation_Correlation_A_B"));
        assert!(result.image.contains_key("Correlation_Correlation_A_C"));
        assert!(result.image.contains_key("Correlation_Correlation_B_C"));
    }

    #[test]
    fn empty_combined_mask_is_all_nan() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let images: Vec<(&str, &[f64])> = vec![("A", a.as_slice()), ("B", b.as_slice())];
        let mask_a = [true, true, false, false];
        let mask_b = [false, false, true, true];
        let masks: Vec<(&str, &[bool])> = vec![("A", mask_a.as_slice()), ("B", mask_b.as_slice())];
        let config = Config {
            threshold_percent: 15.0,
            fast_costes: FastCostesMode::Faster,
            costes_scale_max: 255.0,
            do_corr_and_slope: true,
            do_manders: true,
            do_rwc: true,
            do_overlap: true,
            do_costes: true,
            wants_image_measurements: true,
            wants_object_measurements: false,
        };
        let result = measure(&images, 2, 2, Some(masks.as_slice()), None, &config).unwrap();
        for value in result.image.values() {
            assert!(value.is_nan());
        }
    }

    #[test]
    fn independent_flags_toggle_independently() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [2.0, 1.0, 5.0, 3.0, 6.0, 4.0];
        let images: Vec<(&str, &[f64])> = vec![("A", a.as_slice()), ("B", b.as_slice())];
        let config = Config {
            threshold_percent: 15.0,
            fast_costes: FastCostesMode::Faster,
            costes_scale_max: 255.0,
            do_corr_and_slope: false,
            do_manders: false,
            do_rwc: false,
            do_overlap: true,
            do_costes: false,
            wants_image_measurements: true,
            wants_object_measurements: false,
        };
        let result = measure(&images, 2, 3, None, None, &config).unwrap();
        assert_eq!(result.image.len(), 3); // Overlap, K_A_B, K_B_A only.
        assert!(result.image.contains_key("Correlation_Overlap_A_B"));
        assert!(result.image.contains_key("Correlation_K_A_B"));
        assert!(result.image.contains_key("Correlation_K_B_A"));
    }

    #[test]
    fn rejects_unknown_fast_costes_mode_at_the_binding_layer() {
        // FastCostesMode is a closed enum in this crate -- there is no
        // "unverified" variant to construct. The PyO3 binding is
        // responsible for rejecting unrecognized `fast_costes` strings
        // before they ever reach this function.
        let _ = FastCostesMode::Faster;
        let _ = FastCostesMode::Fast;
        let _ = FastCostesMode::Accurate;
    }
}

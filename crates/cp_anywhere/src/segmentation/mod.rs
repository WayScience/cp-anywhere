//! CellProfiler-compatible object segmentation primitives, implemented
//! against CellProfiler 4.2.8's `IdentifyPrimaryObjects` module (and the
//! shared `Threshold` module it uses) as the compatibility oracle -- not
//! a generic image-processing reimplementation. Has no Python, browser,
//! or filesystem dependency.
//!
//! # Declared scope
//!
//! Every global and adaptive threshold operation (Manual, Otsu two/
//! three-class, Li, Robust Background, Sauvola), log transform,
//! threshold smoothing, declumping/watershed (Intensity/Shape/
//! Propagate), hole filling (including nested-object absorption),
//! border exclusion, and size/count filtering -- see
//! `spec/segmentation-modules.md` for the exact settings mapping and the
//! explicit remaining gaps (not silent approximations). `build_config`
//! is the single, binding-agnostic entry point every FFI layer
//! (`cp_anywhere_python`, `cp_anywhere_wasm`) uses to turn its own
//! string/scalar settings surface into a real `IdentifyPrimaryObjectsConfig`.
//!
//! ## Pixel units and Otsu variants (important)
//!
//! CellProfiler's `Threshold.manual_threshold` and `threshold_range`
//! settings are hard-bounded to `[0, 1]` in the GUI/pipeline
//! (`FloatRange((0, 1), minval=0, maxval=1)` in `threshold.py`) --
//! CellProfiler's own Global threshold machinery assumes images have
//! already been rescaled to `[0, 1]` (its default `Image(..., convert=True)`
//! behavior divides integer-dtype pixels by their dtype's max value,
//! e.g. 255 for `uint8`; float-dtype pixels are never rescaled). This
//! Rust core does **not** perform that rescale itself -- it is
//! unit-agnostic, operating on whatever `f64` values it is given -- the
//! Python binding (`cp_anywhere.segment`) performs the CellProfiler-
//! matching rescale before calling in, so end users can pass raw
//! integer-dtype images directly.
//!
//! Two Otsu variants are provided because CellProfiler's own
//! `skimage.filters.threshold_otsu` dispatches on the *pixel array's
//! dtype at the time it is called* (after any rescale), not on whether
//! the values happen to be integral:
//! - `OtsuTwoClass`: a 256-bin uniform histogram over `[min, max]` of
//!   the (already unit-converted) pixel values, matching
//!   `threshold_otsu`'s float-dtype path exactly -- this is what real
//!   CellProfiler uses for the common case (any rescaled image, which
//!   is always float32/float64 after rescale).
//! - `OtsuTwoClassInteger`: an exact one-bin-per-value histogram,
//!   matching `threshold_otsu`'s integer-dtype path exactly -- only
//!   correct if the caller's numpy array dtype was genuinely integer
//!   *and not rescaled* when CellProfiler's Otsu ran on it (e.g. a
//!   `convert=False` pipeline, matching this project's measurement
//!   crate's own convention, not CellProfiler's default).
//!
//! 3D/volumetric segmentation and `IdentifySecondaryObjects`/
//! `IdentifyTertiaryObjects` are explicit, documented gaps, not silent
//! approximations.

use std::collections::BTreeMap;

/// How the global foreground/background threshold value is chosen,
/// matching `cellprofiler.modules.threshold.Threshold.get_threshold`.
/// See the module docs for the float-vs-integer Otsu distinction.
#[derive(Clone, Copy, Debug)]
pub enum ThresholdMethod {
    /// `TM_MANUAL`: used as-is, with no correction factor or range clamp
    /// (CellProfiler's own `get_threshold` returns the manual value
    /// directly, bypassing `_correct_global_threshold`).
    Manual(f64),
    /// `TM_OTSU`, two-class, global scope, float-dtype histogram path
    /// (the common case -- see module docs).
    OtsuTwoClass,
    /// `TM_OTSU`, two-class, global scope, integer-dtype bincount path
    /// (requires `integer_pixels` -- see module docs).
    OtsuTwoClassInteger,
    /// `TM_OTSU`, three-class, global scope: a 128-bin histogram split
    /// into 3 classes (matching `skimage.filters.threshold_multiotsu`'s
    /// exact index-weighted-moment brute-force criterion, not a
    /// value-weighted approximation -- verified bit-close against the
    /// real function). `assign_middle_to_foreground=true` uses the
    /// lower split (middle class counts as foreground, matching
    /// CellProfiler's `TM_OTSU`/`O_THREE_CLASS`/`"Foreground"`
    /// combination); `false` uses the upper split (`"Background"`).
    OtsuThreeClass { assign_middle_to_foreground: bool },
    /// `TM_LI`: Li's iterative Minimum Cross-Entropy threshold -- see
    /// the `threshold_li` function.
    Li,
    /// `TM_ROBUST_BACKGROUND`: trimmed mean/median/mode plus a multiple
    /// of the standard deviation or MAD -- see
    /// `threshold_robust_background`.
    RobustBackground(RobustBackgroundConfig),
    /// `TM_SAUVOLA`: only ever reachable via `Threshold.local_operation`
    /// (CellProfiler's `global_operation` Choice does not even include
    /// `TM_SAUVOLA` -- it is adaptive-scope-only), so this errors if
    /// combined with `ThresholdScope::Global`. When used as the
    /// Adaptive guide/global threshold, `get_global_threshold` computes
    /// it exactly like `Li` (`self.threshold_operation in (TM_LI,
    /// TM_SAUVOLA)` is one shared branch in the real source) -- see
    /// `threshold_for_values`. The actual local array comes from
    /// `sauvola_threshold`, not the block+spline path every other
    /// method here uses.
    Sauvola,
}

/// `Threshold.averaging_method`, matching `RB_MEAN`/`RB_MEDIAN`/`RB_MODE`.
#[derive(Clone, Copy, Debug, Default)]
pub enum RobustAveraging {
    #[default]
    Mean,
    Median,
    Mode,
}

/// `Threshold.variance_method`, matching `RB_SD`/`RB_MAD`.
#[derive(Clone, Copy, Debug, Default)]
pub enum RobustVariance {
    #[default]
    StandardDeviation,
    MedianAbsoluteDeviation,
}

/// `TM_ROBUST_BACKGROUND`'s settings, matching
/// `Threshold.get_threshold_robust_background` exactly, including its
/// `low_chop == 0` quirk: when `lower_outlier_fraction` rounds to zero
/// pixels, the *upper* trim is also skipped entirely (CellProfiler's
/// own `im = flat_image if low_chop == 0 else flat_image[low_chop:hi_chop]`
/// -- not a narrowed-down reimplementation, the real function's exact
/// behavior).
#[derive(Clone, Copy, Debug)]
pub struct RobustBackgroundConfig {
    pub lower_outlier_fraction: f64,
    pub upper_outlier_fraction: f64,
    pub averaging_method: RobustAveraging,
    pub variance_method: RobustVariance,
    pub number_of_deviations: f64,
}

impl Default for RobustBackgroundConfig {
    fn default() -> Self {
        Self {
            lower_outlier_fraction: 0.05,
            upper_outlier_fraction: 0.05,
            averaging_method: RobustAveraging::default(),
            variance_method: RobustVariance::default(),
            number_of_deviations: 2.0,
        }
    }
}

/// Number of histogram bins for `OtsuThreeClass`, matching
/// `Threshold.get_threshold`'s explicit `nbins=128` for
/// `threshold_multiotsu` (half of `OTSU_FLOAT_BINS`, not the same).
pub const OTSU_THREE_CLASS_BINS: usize = 128;

/// Number of histogram bins for `OtsuTwoClass`, matching
/// `skimage.filters.threshold_otsu`'s default `nbins=256`.
pub const OTSU_FLOAT_BINS: usize = 256;

/// `IdentifyPrimaryObjects.fill_holes`, matching `FH_NEVER`/`FH_DECLUMP`/
/// `FH_THRESHOLDING` exactly (see `run()`'s two `fill_labeled_holes`
/// calls: the first, pre-labeling, only runs for `AfterThresholdingAndDeclump`;
/// the second, post-border/size-filtering, runs for both non-`Never`
/// variants).
///
/// A background 4-connected component is filled with a single label
/// when it does not touch the image border, passes the size check (the
/// first call only), and every foreground pixel 4-adjacent to it shares
/// the *same* label -- matching `centrosome.cpmorphology.fill_labeled_holes`
/// exactly for that common case. A hole adjacent to two or more
/// *different* labels (without a third, fully enclosing object) is left
/// as background, also matching the real function (verified directly
/// against it). A small, fully enclosed foreground object is itself
/// absorbed into its surrounding object's label, also matched (see
/// `holes::fill_labeled_holes` and `spec/segmentation-modules.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillHoles {
    #[default]
    Never,
    /// `FH_DECLUMP`: only the second (post-filter) fill.
    AfterDeclumpOnly,
    /// `FH_THRESHOLDING`: both the first (pre-labeling, size-limited to
    /// `max_diameter * max_diameter` -- not the circular size-filter
    /// area formula) and second fills.
    AfterThresholdingAndDeclump,
}

/// Mirrors the subset of `IdentifyPrimaryObjects` settings this module
/// implements. Field names follow the CellProfiler setting they encode.
#[derive(Clone, Copy, Debug)]
pub struct IdentifyPrimaryObjectsConfig {
    pub threshold_method: ThresholdMethod,
    /// `Threshold.threshold_correction_factor`; multiplies the raw
    /// threshold before clamping. Ignored for `Manual`.
    pub threshold_correction_factor: f64,
    /// `Threshold.threshold_range` (min, max); clamps the corrected
    /// threshold. Ignored for `Manual`. Defaults to `(0.0, 1.0)`,
    /// matching CellProfiler's own hard-coded Setting bounds -- widen
    /// this only if you deliberately operate outside CellProfiler's
    /// normal [0, 1]-rescaled-image convention (see module docs).
    pub threshold_range: (f64, f64),
    /// `IdentifyPrimaryObjects.exclude_size`; when true, objects outside
    /// `[min_diameter, max_diameter]` (by CellProfiler's
    /// `pi * diameter^2 / 4` circular-area approximation) are removed.
    pub exclude_size: bool,
    pub min_diameter: f64,
    pub max_diameter: f64,
    /// `IdentifyPrimaryObjects.exclude_border_objects`; when true,
    /// objects touching the image border are removed (or, if `mask` is
    /// provided to `identify_primary_objects`, objects touching the
    /// mask's own border, matching `filter_on_border`'s mask fallback --
    /// only used when no object touches the image border directly).
    pub exclude_border_objects: bool,
    /// `IdentifyPrimaryObjects.maximum_object_count`, only enforced when
    /// `Some`, matching `LIMIT_ERASE` (`LIMIT_NONE` is `None`): if the
    /// final object count exceeds this, every object is erased.
    pub maximum_object_count: Option<u32>,
    /// `IdentifyPrimaryObjects.fill_holes`; see `FillHoles`.
    pub fill_holes: FillHoles,
    /// `Threshold.threshold_smoothing_scale`; `0.0` means no smoothing
    /// (matches `apply_threshold`'s early-return branch exactly). When
    /// nonzero, the *comparison* image (not the threshold computation,
    /// which always uses the original unsmoothed pixels) is blurred
    /// with a Gaussian of `sigma = threshold_smoothing_scale / 0.6744 / 2.0`,
    /// `mode="constant"` (zero outside the array) with a mask
    /// bleed-over correction matching
    /// `centrosome.smooth.smooth_with_function_and_mask` exactly, even
    /// with no explicit mask (an implicit all-true mask still loses
    /// kernel mass at the image border under `mode="constant"`, which
    /// this correction compensates for).
    pub threshold_smoothing_scale: f64,
    /// `Threshold.log_transform`; only meaningful for `OtsuTwoClass`,
    /// `OtsuThreeClass`, and `Li` (real CellProfiler only shows this
    /// Setting for its `TM_OTSU`/`TM_LI` global operations); ignored
    /// for `Manual`, `OtsuTwoClassInteger`, and `RobustBackground`.
    /// When true, the threshold is computed on
    /// `centrosome.threshold.log_transform`'d pixels (log + stretch to
    /// `[0, 1]`, computed over the *entire* image, not just the masked
    /// subset) and then mapped back with `inverse_log_transform` before
    /// `threshold_correction_factor`/clamping are applied -- exactly
    /// matching real CellProfiler's `get_threshold` order of
    /// operations.
    pub log_transform: bool,
    /// `Threshold.threshold_scope`: `Global` (the default; a single
    /// scalar threshold for the whole image) or `Adaptive` (a per-pixel
    /// threshold array, blended across `window_size`-sized blocks via
    /// `get_adaptive_threshold`/`RectBivariateSpline`, bounded to
    /// `[0.7, 1.5] * global_guide_threshold` -- see
    /// `Threshold._correct_local_threshold`). Not supported for
    /// `Manual` or `OtsuTwoClassInteger` (real CellProfiler's
    /// `get_local_threshold` only handles `TM_OTSU`/`TM_LI`/
    /// `TM_ROBUST_BACKGROUND`/`TM_SAUVOLA`).
    pub threshold_scope: ThresholdScope,
    /// Declumping/watershed separation of touching objects, run right
    /// after the initial 8-connected labeling and before border/size
    /// filtering -- matching `IdentifyPrimaryObjects.run()`'s call to
    /// `separate_neighboring_objects` at that exact point. Defaults to
    /// `UnclumpMethod::None`/`WatershedMethod::None`, which
    /// `declump()` itself treats as an exact no-op (matching real
    /// CellProfiler's own `unclump_method == UN_NONE` early return),
    /// so leaving this at its default preserves every prior fixture's
    /// behavior unchanged.
    pub declump: DeclumpSettings,
}

/// See `IdentifyPrimaryObjectsConfig::declump`.
#[derive(Clone, Copy, Debug)]
pub struct DeclumpSettings {
    pub unclump: declump::UnclumpMethod,
    pub watershed: declump::WatershedMethod,
    pub smoothing: declump::Smoothing,
    pub suppression: declump::MaximaSuppression,
    /// `IdentifyPrimaryObjects.low_resolution_maxima`; only takes
    /// effect when `min_diameter > 10.0` (matching real CellProfiler's
    /// own gate), in which case maxima are found on a downsampled
    /// copy of the image for speed.
    pub low_resolution_maxima: bool,
}

impl Default for DeclumpSettings {
    fn default() -> Self {
        Self {
            unclump: declump::UnclumpMethod::None,
            watershed: declump::WatershedMethod::None,
            smoothing: declump::Smoothing::Automatic,
            suppression: declump::MaximaSuppression::Automatic,
            low_resolution_maxima: false,
        }
    }
}

/// See `IdentifyPrimaryObjectsConfig::threshold_scope`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ThresholdScope {
    #[default]
    Global,
    /// `TS_ADAPTIVE`. `window_size` is `Threshold.adaptive_window_size`
    /// (CellProfiler forces it odd for Sauvola only; otherwise used
    /// as-is for the block grid).
    Adaptive { window_size: usize },
}

impl Default for IdentifyPrimaryObjectsConfig {
    fn default() -> Self {
        Self {
            threshold_method: ThresholdMethod::OtsuTwoClass,
            threshold_correction_factor: 1.0,
            threshold_range: (0.0, 1.0),
            exclude_size: false,
            min_diameter: 0.0,
            max_diameter: f64::INFINITY,
            exclude_border_objects: false,
            maximum_object_count: None,
            fill_holes: FillHoles::Never,
            threshold_smoothing_scale: 0.0,
            log_transform: false,
            threshold_scope: ThresholdScope::Global,
            declump: DeclumpSettings::default(),
        }
    }
}

#[derive(Debug)]
pub struct SegmentationResult {
    /// Final object labels, one entry per pixel in row-major order;
    /// `0` is background. Matches CellProfiler's `objects.segmented`.
    pub labels: Vec<u32>,
    /// Labels after the small-object filter only (large objects not yet
    /// removed), matching CellProfiler's `objects.small_removed_segmented`.
    pub small_removed_labels: Vec<u32>,
    /// Labels immediately after declumping, before any filtering.
    pub unedited_labels: Vec<u32>,
    pub object_count: u32,
    /// The threshold actually applied (after correction/clamping).
    pub threshold: f64,
    /// Corrected per-pixel thresholds for Adaptive; None for Global.
    pub threshold_map: Option<Vec<f64>>,
    /// Corrected global guide used to constrain Adaptive thresholds.
    pub guide_threshold: Option<f64>,
}

fn clamp(value: f64, range: (f64, f64)) -> f64 {
    value.max(range.0).min(range.1)
}

/// Plain, FFI-agnostic parameters mirroring the full settings surface of
/// `IdentifyPrimaryObjectsConfig`, with every field a string/scalar
/// instead of an enum -- the same shape every binding (`cp_anywhere_python`,
/// `cp_anywhere_wasm`) exposes to its host language. `build_config` is the
/// *single* place that maps these strings to the real enums, so adding a
/// new threshold method or declump mode only needs updating one match
/// arm, not one per binding (avoiding drift between Python and WASM).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ConfigParams {
    pub threshold_method: String,
    pub threshold_value: Option<f64>,
    pub threshold_correction_factor: f64,
    pub threshold_range: (f64, f64),
    pub exclude_size: bool,
    pub min_diameter: f64,
    pub max_diameter: f64,
    pub exclude_border_objects: bool,
    pub maximum_object_count: Option<u32>,
    pub fill_holes: Option<String>,
    pub threshold_smoothing_scale: f64,
    pub assign_middle_to_foreground: Option<bool>,
    pub robust_background_lower_outlier_fraction: f64,
    pub robust_background_upper_outlier_fraction: f64,
    pub robust_background_averaging_method: Option<String>,
    pub robust_background_variance_method: Option<String>,
    pub robust_background_number_of_deviations: f64,
    pub log_transform: bool,
    pub threshold_scope: Option<String>,
    pub adaptive_window_size: usize,
    pub unclump_method: Option<String>,
    pub watershed_method: Option<String>,
    pub smoothing_filter_size: Option<f64>,
    pub maxima_suppression_size: Option<f64>,
    pub low_resolution_maxima: bool,
}

impl Default for ConfigParams {
    fn default() -> Self {
        Self {
            threshold_method: "otsu".to_string(),
            threshold_value: None,
            threshold_correction_factor: 1.0,
            threshold_range: (0.0, 1.0),
            exclude_size: false,
            min_diameter: 0.0,
            max_diameter: f64::INFINITY,
            exclude_border_objects: false,
            maximum_object_count: None,
            fill_holes: None,
            threshold_smoothing_scale: 0.0,
            assign_middle_to_foreground: None,
            robust_background_lower_outlier_fraction: 0.05,
            robust_background_upper_outlier_fraction: 0.05,
            robust_background_averaging_method: None,
            robust_background_variance_method: None,
            robust_background_number_of_deviations: 2.0,
            log_transform: false,
            threshold_scope: None,
            adaptive_window_size: 50,
            unclump_method: None,
            watershed_method: None,
            smoothing_filter_size: None,
            maxima_suppression_size: None,
            low_resolution_maxima: false,
        }
    }
}

/// Maps `ConfigParams`'s plain strings/scalars to a real
/// `IdentifyPrimaryObjectsConfig`, returning a plain `String` error (not
/// a binding-specific exception type) so every binding can wrap it in
/// whatever its host language expects (`PyValueError`, `JsValue`, ...).
pub fn build_config(params: &ConfigParams) -> Result<IdentifyPrimaryObjectsConfig, String> {
    let method = match params.threshold_method.as_str() {
        "manual" => ThresholdMethod::Manual(params.threshold_value.ok_or_else(|| {
            "threshold_value is required for threshold_method='manual'".to_string()
        })?),
        "otsu" => ThresholdMethod::OtsuTwoClass,
        "otsu_integer" => ThresholdMethod::OtsuTwoClassInteger,
        "otsu_three_class" => ThresholdMethod::OtsuThreeClass {
            assign_middle_to_foreground: params.assign_middle_to_foreground.unwrap_or(true),
        },
        "li" => ThresholdMethod::Li,
        "sauvola" => ThresholdMethod::Sauvola,
        "robust_background" => {
            let averaging_method = match params.robust_background_averaging_method.as_deref() {
                None | Some("mean") => RobustAveraging::Mean,
                Some("median") => RobustAveraging::Median,
                Some("mode") => RobustAveraging::Mode,
                Some(other) => {
                    return Err(format!(
                        "unsupported robust_background_averaging_method: {other:?}; \
                         use 'mean', 'median', or 'mode'"
                    ))
                }
            };
            let variance_method = match params.robust_background_variance_method.as_deref() {
                None | Some("sd") => RobustVariance::StandardDeviation,
                Some("mad") => RobustVariance::MedianAbsoluteDeviation,
                Some(other) => {
                    return Err(format!(
                        "unsupported robust_background_variance_method: {other:?}; \
                         use 'sd' or 'mad'"
                    ))
                }
            };
            ThresholdMethod::RobustBackground(RobustBackgroundConfig {
                lower_outlier_fraction: params.robust_background_lower_outlier_fraction,
                upper_outlier_fraction: params.robust_background_upper_outlier_fraction,
                averaging_method,
                variance_method,
                number_of_deviations: params.robust_background_number_of_deviations,
            })
        }
        other => {
            return Err(format!(
                "unsupported threshold_method: {other:?}; use 'manual', 'otsu', \
                 'otsu_integer', 'otsu_three_class', 'li', 'robust_background', or 'sauvola'"
            ))
        }
    };
    let fill_holes = match params.fill_holes.as_deref() {
        None | Some("never") => FillHoles::Never,
        Some("after_declump_only") => FillHoles::AfterDeclumpOnly,
        Some("after_thresholding_and_declump") => FillHoles::AfterThresholdingAndDeclump,
        Some(other) => {
            return Err(format!(
                "unsupported fill_holes: {other:?}; use 'never', \
                 'after_declump_only', or 'after_thresholding_and_declump'"
            ))
        }
    };
    let threshold_scope = match params.threshold_scope.as_deref() {
        None | Some("global") => ThresholdScope::Global,
        Some("adaptive") => ThresholdScope::Adaptive {
            window_size: params.adaptive_window_size,
        },
        Some(other) => {
            return Err(format!(
                "unsupported threshold_scope: {other:?}; use 'global' or 'adaptive'"
            ))
        }
    };
    let declump = DeclumpSettings {
        unclump: match params.unclump_method.as_deref() {
            None | Some("none") => declump::UnclumpMethod::None,
            Some("intensity") => declump::UnclumpMethod::Intensity,
            Some("shape") => declump::UnclumpMethod::Shape,
            Some(other) => {
                return Err(format!(
                    "unsupported unclump_method: {other:?}; use 'none', 'intensity', or 'shape'"
                ))
            }
        },
        watershed: match params.watershed_method.as_deref() {
            None | Some("none") => declump::WatershedMethod::None,
            Some("intensity") => declump::WatershedMethod::Intensity,
            Some("shape") => declump::WatershedMethod::Shape,
            Some("propagate") => declump::WatershedMethod::Propagate,
            Some(other) => {
                return Err(format!(
                    "unsupported watershed_method: {other:?}; use 'none', 'intensity', \
                     'shape', or 'propagate'"
                ))
            }
        },
        smoothing: match params.smoothing_filter_size {
            None => declump::Smoothing::Automatic,
            Some(size) => declump::Smoothing::Manual(size),
        },
        suppression: match params.maxima_suppression_size {
            None => declump::MaximaSuppression::Automatic,
            Some(size) => declump::MaximaSuppression::Manual(size),
        },
        low_resolution_maxima: params.low_resolution_maxima,
    };
    Ok(IdentifyPrimaryObjectsConfig {
        threshold_method: method,
        threshold_correction_factor: params.threshold_correction_factor,
        threshold_range: params.threshold_range,
        exclude_size: params.exclude_size,
        min_diameter: params.min_diameter,
        max_diameter: params.max_diameter,
        exclude_border_objects: params.exclude_border_objects,
        maximum_object_count: params.maximum_object_count,
        fill_holes,
        threshold_smoothing_scale: params.threshold_smoothing_scale,
        log_transform: params.log_transform,
        threshold_scope,
        declump,
    })
}

/// Shared two-class Otsu maximization over a precomputed histogram,
/// matching `skimage.filters.threshold_otsu`'s cumulative-sum
/// variance-maximization loop exactly for either histogram variant.
fn otsu_from_histogram(counts: &[f64], bin_centers: &[f64]) -> f64 {
    let n_bins = counts.len();
    let mut weight1 = vec![0.0f64; n_bins];
    let mut running = 0.0;
    for i in 0..n_bins {
        running += counts[i];
        weight1[i] = running;
    }
    let mut weight2 = vec![0.0f64; n_bins];
    running = 0.0;
    for i in (0..n_bins).rev() {
        running += counts[i];
        weight2[i] = running;
    }
    let mut cum_mean1 = vec![0.0f64; n_bins];
    running = 0.0;
    for i in 0..n_bins {
        running += counts[i] * bin_centers[i];
        cum_mean1[i] = running / weight1[i];
    }
    let mut cum_mean2 = vec![0.0f64; n_bins];
    running = 0.0;
    for i in (0..n_bins).rev() {
        running += counts[i] * bin_centers[i];
        cum_mean2[i] = running / weight2[i];
    }

    let mut best_idx = 0usize;
    let mut best_variance = f64::NEG_INFINITY;
    for i in 0..n_bins.saturating_sub(1) {
        let variance = weight1[i] * weight2[i + 1] * (cum_mean1[i] - cum_mean2[i + 1]).powi(2);
        if variance > best_variance {
            best_variance = variance;
            best_idx = i;
        }
    }
    bin_centers[best_idx]
}

/// Exact integer-dtype Otsu two-class threshold, matching
/// `skimage.filters.threshold_otsu`'s bincount-based path: one bin per
/// integer value from `min` to `max` inclusive (no histogram binning
/// approximation).
fn otsu_threshold_integer(values: &[i64]) -> Result<f64, String> {
    if values.is_empty() {
        return Ok(0.0);
    }
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    if min == max {
        return Ok(min as f64);
    }
    let n_bins = (max - min + 1) as usize;
    let mut counts = vec![0.0f64; n_bins];
    for &value in values {
        counts[(value - min) as usize] += 1.0;
    }
    let bin_centers: Vec<f64> = (0..n_bins).map(|i| (min + i as i64) as f64).collect();
    Ok(otsu_from_histogram(&counts, &bin_centers))
}

/// Float-dtype Otsu two-class threshold, matching
/// `skimage.filters.threshold_otsu`'s default path: a 256-bin uniform
/// histogram over `[min(values), max(values)]`, matching
/// `numpy.histogram`'s bin-edge/bin-center and fast uniform-binning
/// convention (truncating bin assignment, last value clamped into the
/// final bin).
fn otsu_threshold_float(values: &[f64], n_bins: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let first = values[0];
    if values.iter().all(|&v| v == first) {
        return first;
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let width = (max - min) / n_bins as f64;
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| min + width * (i as f64 + 0.5))
        .collect();
    let mut counts = vec![0.0f64; n_bins];
    for &value in values {
        let mut idx = ((value - min) / width) as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= n_bins {
            idx = n_bins as isize - 1;
        }
        counts[idx as usize] += 1.0;
    }
    otsu_from_histogram(&counts, &bin_centers)
}

/// 256(ish)-bin uniform histogram over `[min, max]` of `values`,
/// matching `skimage.exposure.histogram(values, nbins, source_range="image",
/// normalize=True)`'s binning convention exactly (same as
/// `otsu_threshold_float`'s own binning, generalized to any `n_bins`).
/// Returns `(probabilities, bin_centers)`.
fn histogram_probabilities(values: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let width = if max > min {
        (max - min) / n_bins as f64
    } else {
        1.0
    };
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| min + width * (i as f64 + 0.5))
        .collect();
    let mut counts = vec![0.0f64; n_bins];
    for &value in values {
        let mut idx = if max > min {
            ((value - min) / width) as isize
        } else {
            0
        };
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= n_bins {
            idx = n_bins as isize - 1;
        }
        counts[idx as usize] += 1.0;
    }
    let total: f64 = counts.iter().sum();
    let probabilities: Vec<f64> = if total > 0.0 {
        counts.iter().map(|&c| c / total).collect()
    } else {
        counts
    };
    (probabilities, bin_centers)
}

/// Three-class Otsu threshold, matching
/// `skimage.filters.threshold_multiotsu(values, classes=3, nbins=128)`'s
/// exact brute-force criterion (`_get_multiotsu_thresh_indices`, index-
/// weighted moments, not value-weighted -- verified bit-close against
/// the real function; see module tests). Returns `(lower, upper)`
/// threshold values. Errors if fewer than 3 histogram bins have
/// nonzero probability, matching `threshold_multiotsu`'s own
/// `ValueError` in that case.
fn otsu_threshold_three_class(values: &[f64], n_bins: usize) -> Result<(f64, f64), String> {
    const MIN_NONZERO_BINS: usize = 3;
    if values.is_empty() {
        return Err("otsu_threshold_three_class requires at least one value".to_string());
    }
    let (prob, bin_centers) = histogram_probabilities(values, n_bins);
    if prob.iter().filter(|&&p| p > 0.0).count() < MIN_NONZERO_BINS {
        return Err(
            "after discretization, fewer than 3 distinct values are present; \
             cannot threshold into 3 classes"
                .to_string(),
        );
    }

    let mut zeroth = vec![0.0f64; n_bins];
    let mut first = vec![0.0f64; n_bins];
    zeroth[0] = prob[0];
    first[0] = prob[0]; // matches the real source's i=0 special case exactly (not 0 * prob[0])
    for i in 1..n_bins {
        zeroth[i] = zeroth[i - 1] + prob[i];
        first[i] = first[i - 1] + (i as f64) * prob[i];
    }
    let var_between = |i: usize, j: usize| -> f64 {
        if i == 0 {
            if zeroth[j] > 0.0 {
                first[j] * first[j] / zeroth[j]
            } else {
                0.0
            }
        } else {
            let z = zeroth[j] - zeroth[i - 1];
            if z > 0.0 {
                let f = first[j] - first[i - 1];
                f * f / z
            } else {
                0.0
            }
        }
    };

    let mut best = (0usize, 1usize);
    let mut best_sigma = f64::NEG_INFINITY;
    for t0 in 0..n_bins.saturating_sub(2) {
        for t1 in (t0 + 1)..n_bins.saturating_sub(1) {
            let sigma =
                var_between(0, t0) + var_between(t0 + 1, t1) + var_between(t1 + 1, n_bins - 1);
            if sigma > best_sigma {
                best_sigma = sigma;
                best = (t0, t1);
            }
        }
    }
    Ok((bin_centers[best.0], bin_centers[best.1]))
}

/// Li's iterative Minimum Cross-Entropy threshold, matching
/// `skimage.filters.threshold_li`'s float-dtype path exactly (the path
/// always taken here, since pixels reaching this function are always
/// `f64`), with CellProfiler's own tolerance formula:
/// `max(min(diff(unique(values))) / 2, 0.5 / 65536)`. Errors if fewer
/// than 2 distinct values are present, matching the real
/// `numpy.min` over an empty `diff(unique(...))` array crashing in
/// CellProfiler's own `get_threshold` for a literally constant image.
fn threshold_li(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("threshold_li requires at least one value".to_string());
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut unique = sorted.clone();
    unique.dedup();
    if unique.len() < 2 {
        return Err(
            "threshold_li requires at least two distinct values (CellProfiler's own \
             tolerance computation divides by the smallest gap between distinct values)"
                .to_string(),
        );
    }
    let min_gap = unique
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f64::INFINITY, f64::min);
    let tolerance = (min_gap / 2.0).max(0.5 / 65536.0);

    Ok(threshold_li_with_tolerance(values, tolerance))
}

/// Local Li uses ONE tolerance computed from the full (unmasked) image,
/// unlike the global guide, whose tolerance uses only included pixels.
fn threshold_li_with_tolerance(values: &[f64], tolerance: f64) -> f64 {
    let image_min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let shifted: Vec<f64> = values.iter().map(|&v| v - image_min).collect();

    let mut t_next = shifted.iter().sum::<f64>() / shifted.len() as f64;
    let mut t_curr = -2.0 * tolerance;
    while (t_next - t_curr).abs() > tolerance {
        t_curr = t_next;
        let (fore_sum, fore_count, back_sum, back_count) = shifted.iter().fold(
            (0.0_f64, 0usize, 0.0_f64, 0usize),
            |(fs, fc, bs, bc), &v| {
                if v > t_curr {
                    (fs + v, fc + 1, bs, bc)
                } else {
                    (fs, fc, bs + v, bc + 1)
                }
            },
        );
        let mean_fore = fore_sum / fore_count as f64;
        let mean_back = back_sum / back_count as f64;
        if mean_back == 0.0 {
            break;
        }
        t_next = (mean_back - mean_fore) / (mean_back.ln() - mean_fore.ln());
    }
    t_next + image_min
}

/// `np.percentile(sorted_values, p)` with linear interpolation (numpy's
/// default `method="linear"`), given an already-sorted slice.
fn percentile_linear_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let frac = rank - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * frac
}

/// `numpy.median`, given an already-sorted slice.
fn median_sorted(sorted: &[f64]) -> f64 {
    percentile_linear_sorted(sorted, 50.0)
}

/// `centrosome.threshold.mad`: median absolute deviation from the
/// median. Does not require sorted input (sorts its own copies).
fn median_absolute_deviation(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = median_sorted(&sorted);
    let mut deviations: Vec<f64> = values.iter().map(|&v| (v - med).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    median_sorted(&deviations)
}

/// `numpy.std` (population standard deviation, `ddof=0`).
fn population_std(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// `centrosome.threshold.binned_mode`: bin into `ceil(sqrt(n))` equal-
/// width bins over `[min, max]` via truncating division (not
/// clamping), find the most populated bin, return the 50th-percentile-
/// of-that-bin's-width point via
/// `np.percentile(a, 100 * (idx + 0.5) / n_bins)` (note: a percentile
/// of the *whole* sorted array, not a value restricted to the bin --
/// matches the real function exactly, including that subtlety).
///
/// Values exactly equal to `max` truncate to bin index `n_bins`
/// (`numpy.bincount` silently grows to accommodate it, matching the
/// real `(a - a_min) / (a_max - a_min) * n_bins).astype(int)`): no
/// clamping to `n_bins - 1` happens here or in the real code. Errors
/// if that overflow bin turns out to be the most populated one, since
/// `100 * (n_bins + 0.5) / n_bins` exceeds 100 and the real
/// `numpy.percentile` call itself raises `ValueError: Percentiles must
/// be in the range [0, 100]` in that case -- a genuine real-CellProfiler
/// crash, not something to silently paper over.
fn binned_mode(values: &[f64]) -> Result<f64, String> {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_bins = (values.len() as f64).sqrt().ceil() as usize;
    let mut counts = vec![0u64; n_bins + 1];
    for &v in values {
        let idx = if max > min {
            (((v - min) / (max - min)) * n_bins as f64) as usize
        } else {
            0
        };
        counts[idx] += 1;
    }
    // `numpy.argmax` returns the FIRST index on ties; Rust's
    // `max_by_key` returns the LAST, so pick first-wins explicitly.
    let mut best_idx = 0usize;
    let mut best_count = counts[0];
    for (idx, &count) in counts.iter().enumerate().skip(1) {
        if count > best_count {
            best_idx = idx;
            best_count = count;
        }
    }
    let q = 100.0 * (best_idx as f64 + 0.5) / n_bins as f64;
    if !(0.0..=100.0).contains(&q) {
        return Err(format!(
            "binned_mode's most-populated bin is the top-edge overflow bin \
             (percentile {q} is out of [0, 100]), matching a genuine crash in \
             centrosome.threshold.binned_mode/numpy.percentile for this input"
        ));
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(percentile_linear_sorted(&sorted, q))
}

/// Robust Background threshold, matching
/// `Threshold.get_threshold_robust_background` exactly (see
/// `RobustBackgroundConfig`'s docs for its `low_chop == 0` quirk).
fn threshold_robust_background(
    values: &[f64],
    config: &RobustBackgroundConfig,
) -> Result<f64, String> {
    const MIN_PIXELS: usize = 3;
    if values.len() < MIN_PIXELS {
        return Ok(0.0);
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if sorted[0] == sorted[n - 1] {
        return Ok(sorted[0]);
    }
    let low_chop = (n as f64 * config.lower_outlier_fraction).round_ties_even() as usize;
    let hi_chop = n - (n as f64 * config.upper_outlier_fraction).round_ties_even() as usize;
    let trimmed: &[f64] = if low_chop == 0 {
        &sorted
    } else {
        &sorted[low_chop..hi_chop]
    };

    let average = match config.averaging_method {
        RobustAveraging::Mean => trimmed.iter().sum::<f64>() / trimmed.len() as f64,
        RobustAveraging::Median => {
            let mut copy = trimmed.to_vec();
            copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
            median_sorted(&copy)
        }
        RobustAveraging::Mode => binned_mode(trimmed)?,
    };
    let variance = match config.variance_method {
        RobustVariance::StandardDeviation => population_std(trimmed),
        RobustVariance::MedianAbsoluteDeviation => median_absolute_deviation(trimmed),
    };
    Ok(average + variance * config.number_of_deviations)
}

/// `centrosome.threshold.log_transform`, operating over the *entire*
/// pixel array (not a masked subset -- matches the real function,
/// which is always called on the full image before any masked subset
/// is extracted for threshold computation). Returns the transformed
/// image plus `(log_min, log_max)` for `inverse_log_transform`.
fn log_transform(pixels: &[f64]) -> (Vec<f64>, f64, f64) {
    let orig_min = pixels.iter().cloned().fold(f64::INFINITY, f64::min);
    let orig_max = pixels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let noise_min = orig_min + (orig_max - orig_min) / 256.0 + f64::EPSILON;
    let limage: Vec<f64> = pixels
        .iter()
        .map(|&v| if v < noise_min { noise_min } else { v }.ln())
        .collect();
    let log_min = limage.iter().cloned().fold(f64::INFINITY, f64::min);
    let log_max = limage.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let transformed = stretch_to_unit(&limage, log_min, log_max);
    (transformed, log_min, log_max)
}

/// `centrosome.filter.stretch` (unmasked path): maps `[minval, maxval]`
/// to `[0, 1]`, with `stretch`'s own degenerate-constant-image branch
/// (returns all-zeros if `minval < 0`, all-ones if `minval > 1`, or the
/// input unchanged otherwise) when `minval == maxval`.
fn stretch_to_unit(values: &[f64], minval: f64, maxval: f64) -> Vec<f64> {
    if minval == maxval {
        if minval < 0.0 {
            vec![0.0; values.len()]
        } else if minval > 1.0 {
            vec![1.0; values.len()]
        } else {
            values.to_vec()
        }
    } else {
        values
            .iter()
            .map(|&v| (v - minval) / (maxval - minval))
            .collect()
    }
}

/// `centrosome.threshold.inverse_log_transform`:
/// `exp(unstretch(value, log_min, log_max))` where
/// `unstretch(x, lo, hi) = x * (hi - lo) + lo`.
fn inverse_log_transform(value: f64, log_min: f64, log_max: f64) -> f64 {
    (value * (log_max - log_min) + log_min).exp()
}

/// Not-a-knot B-spline knot vector, matching
/// `scipy.interpolate._bsplines._not_a_knot` -- the knot vector that
/// `scipy.interpolate.RectBivariateSpline(..., s=0)` constructs for
/// each axis internally (verified against real scipy's `get_knots()`).
fn not_a_knot_vector(x: &[f64], degree: usize) -> Vec<f64> {
    let m = x.len();
    let interior: Vec<f64> = if degree % 2 == 1 {
        let k2 = degree.div_ceil(2);
        x[k2..m - k2].to_vec()
    } else {
        let k2 = degree / 2;
        let midpoints: Vec<f64> = (0..m - 1).map(|i| (x[i] + x[i + 1]) / 2.0).collect();
        midpoints[k2..midpoints.len() - k2].to_vec()
    };
    let mut knots = vec![x[0]; degree + 1];
    knots.extend(interior);
    knots.extend(std::iter::repeat_n(x[m - 1], degree + 1));
    knots
}

/// NURBS book Algorithm A2.1 "FindSpan": the knot span index `i` such
/// that `knots[i] <= x < knots[i + 1]`, clamped so `x` at or beyond the
/// last distinct knot value maps to the last valid span `n` -- the
/// standard closed-right-endpoint convention also used by
/// FITPACK/scipy's B-spline evaluation. `n` is the highest basis-
/// function index (`n_basis - 1`).
fn find_span(n: usize, degree: usize, x: f64, knots: &[f64]) -> usize {
    if x >= knots[n + 1] {
        return n;
    }
    if x <= knots[degree] {
        return degree;
    }
    let mut low = degree;
    let mut high = n + 1;
    let mut mid = (low + high) / 2;
    while x < knots[mid] || x >= knots[mid + 1] {
        if x < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

/// NURBS book Algorithm A2.2 "BasisFuns": the `degree + 1` nonzero
/// B-spline basis function values at `x`, for global basis-function
/// indices `span - degree ..= span`.
fn basis_funs(span: usize, x: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let mut n = vec![0.0; degree + 1];
    n[0] = 1.0;
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    for j in 1..=degree {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// Dense B-spline design matrix (one row per evaluation point, one
/// column per basis function), matching
/// `scipy.interpolate.BSpline.design_matrix(x, knots, degree).toarray()`
/// (verified bit-close against the real scipy routine; see module
/// tests).
fn bspline_design_matrix(x: &[f64], knots: &[f64], degree: usize, n_basis: usize) -> Vec<Vec<f64>> {
    let n = n_basis - 1;
    x.iter()
        .map(|&xi| {
            let span = find_span(n, degree, xi, knots);
            let values = basis_funs(span, xi, degree, knots);
            let mut row = vec![0.0; n_basis];
            for (offset, &v) in values.iter().enumerate() {
                row[span - degree + offset] = v;
            }
            row
        })
        .collect()
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    (0..cols)
        .map(|c| (0..rows).map(|r| matrix[r][c]).collect())
        .collect()
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let k = b.len();
    let m = if k > 0 { b[0].len() } else { 0 };
    (0..n)
        .map(|i| {
            (0..m)
                .map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum())
                .collect()
        })
        .collect()
}

/// Solves `a @ x = b` for `x`, where `a` is `n x n` and `b` is `n x m`
/// (one or more right-hand-side columns), via Gaussian elimination
/// with partial pivoting. Used to fit tensor-product B-spline
/// coefficients for 2D grid interpolation.
fn solve_linear_system(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    let m = b[0].len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend_from_slice(&b[i]);
            row
        })
        .collect();
    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, row_data) in aug.iter().enumerate().skip(col + 1) {
            if row_data[col].abs() > max_val {
                max_val = row_data[col].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err("singular matrix while fitting B-spline interpolation".to_string());
        }
        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        for value in aug[col].iter_mut().skip(col) {
            *value /= pivot;
        }
        let pivot_row_data = aug[col].clone();
        for (row, row_data) in aug.iter_mut().enumerate() {
            if row != col {
                let factor = row_data[col];
                if factor != 0.0 {
                    for k in col..(n + m) {
                        row_data[k] -= factor * pivot_row_data[k];
                    }
                }
            }
        }
    }
    Ok((0..n).map(|i| aug[i][n..n + m].to_vec()).collect())
}

/// Tensor-product B-spline interpolation through a rectangular grid of
/// values, matching
/// `scipy.interpolate.RectBivariateSpline(x, y, z, kx=.., ky=.., s=0)`
/// evaluated at `(eval_x, eval_y)` -- verified bit-close (~1e-12)
/// against the real scipy/FITPACK routine for both odd and mixed
/// odd/even degrees, via the mathematically equivalent (for pure
/// interpolation, no smoothing) separable not-a-knot B-spline fit: fit
/// along x, then fit the result along y (see module tests). `z` has
/// `x.len()` rows and `y.len()` columns.
fn rect_bivariate_spline_interpolate(
    x: &[f64],
    y: &[f64],
    z: &[Vec<f64>],
    kx: usize,
    ky: usize,
    eval_x: &[f64],
    eval_y: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    let tx = not_a_knot_vector(x, kx);
    let ty = not_a_knot_vector(y, ky);
    let bx = bspline_design_matrix(x, &tx, kx, x.len());
    let by = bspline_design_matrix(y, &ty, ky, y.len());
    let d = solve_linear_system(&bx, z)?;
    let d_t = transpose(&d);
    let c_t = solve_linear_system(&by, &d_t)?;
    let c = transpose(&c_t);
    let bx_eval = bspline_design_matrix(eval_x, &tx, kx, x.len());
    let by_eval = bspline_design_matrix(eval_y, &ty, ky, y.len());
    let bc = matmul(&bx_eval, &c);
    let by_eval_t = transpose(&by_eval);
    Ok(matmul(&bc, &by_eval_t))
}

/// `numpy.linspace(start, end, n)`: `n` evenly spaced points including
/// both endpoints, with the last point forced to exactly `end` (not
/// derived via step accumulation) to avoid floating-point drift,
/// matching numpy's own implementation detail.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    let mut result: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
    let last = result.len() - 1;
    result[last] = end;
    result
}

/// `numpy.pad(..., mode="reflect")` boundary index mapping: reflects
/// without repeating the edge value (period `2 * (n - 1)`).
fn reflect_index(idx: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut m = idx % period;
    if m < 0 {
        m += period;
    }
    if m >= n as isize {
        m = period - m;
    }
    m as usize
}

/// `skimage.filters.threshold_sauvola` for `f64` pixel data: `R`
/// (the dynamic range of standard deviation) is always `1.0` for
/// float64 images (`skimage.util.dtype_limits` gives `(-1, 1)` for any
/// floating dtype; verified against the real function), so this
/// hard-codes it rather than threading a parameter through.
/// `window_size` must be odd (CellProfiler itself increments an even
/// `adaptive_window_size` by 1 before calling this).
fn sauvola_threshold(pixels: &[f64], rows: usize, cols: usize, window_size: usize) -> Vec<f64> {
    const K: f64 = 0.2;
    const R: f64 = 1.0;
    let half = (window_size / 2) as isize;
    let mut out = vec![0.0; rows * cols];
    let mut window = Vec::with_capacity(window_size * window_size);
    for i in 0..rows {
        for j in 0..cols {
            window.clear();
            for di in -half..=half {
                let ii = reflect_index(i as isize + di, rows);
                for dj in -half..=half {
                    let jj = reflect_index(j as isize + dj, cols);
                    window.push(pixels[ii * cols + jj]);
                }
            }
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let std = population_std(&window);
            out[i * cols + j] = mean * (1.0 + K * (std / R - 1.0));
        }
    }
    out
}

/// `Threshold._get_adaptive_threshold`: breaks the image into
/// `rows / window_size x cols / window_size` blocks, computes a
/// per-block threshold (via `block_threshold_fn`, which receives only
/// the mask-included pixel values in that block; it is not called at
/// all when the block is empty or constant -- those cases are handled
/// here exactly like the real function), then blends the block grid
/// across the full image with `rect_bivariate_spline_interpolate`.
/// Errors exactly like the real function if the window is too large
/// for the image (fewer than 2 blocks along either axis).
fn get_adaptive_threshold(
    pixels: &[f64],
    mask: Option<&[bool]>,
    rows: usize,
    cols: usize,
    window_size: usize,
    block_threshold_fn: impl Fn(&[f64]) -> Result<f64, String>,
) -> Result<Vec<f64>, String> {
    let nblocks_x = rows / window_size;
    let nblocks_y = cols / window_size;
    if nblocks_x < 2 || nblocks_y < 2 {
        return Err(format!(
            "Adaptive window cannot exceed 50% of an image dimension.\n\
             Window of {window_size}px is too large for a {cols}x{rows} image"
        ));
    }
    let increment_x = rows as f64 / nblocks_x as f64;
    let increment_y = cols as f64 / nblocks_y as f64;

    let mut block_threshold = vec![vec![0.0; nblocks_y]; nblocks_x];
    for (i, block_row) in block_threshold.iter_mut().enumerate() {
        let i0 = (i as f64 * increment_x) as usize;
        let i1 = ((i + 1) as f64 * increment_x) as usize;
        for (j, cell) in block_row.iter_mut().enumerate() {
            let j0 = (j as f64 * increment_y) as usize;
            let j1 = ((j + 1) as f64 * increment_y) as usize;
            let mut block_values = Vec::new();
            for r in i0..i1 {
                for c in j0..j1 {
                    let idx = r * cols + c;
                    let include = mask.map(|m| m[idx]).unwrap_or(true);
                    if include {
                        block_values.push(pixels[idx]);
                    }
                }
            }
            *cell = if block_values.is_empty() {
                0.0
            } else if block_values.iter().all(|&v| v == block_values[0]) {
                block_values[0]
            } else {
                block_threshold_fn(&block_values)?
            };
        }
    }

    let spline_order = 3.min(nblocks_x.min(nblocks_y) - 1);
    let x_start = (increment_x / 2.0).trunc();
    let x_end = ((nblocks_x as f64 - 0.5) * increment_x).trunc();
    let y_start = (increment_y / 2.0).trunc();
    let y_end = ((nblocks_y as f64 - 0.5) * increment_y).trunc();
    let block_x_coords = linspace(x_start, x_end, nblocks_x);
    let block_y_coords = linspace(y_start, y_end, nblocks_y);

    let thresh_out_x_coords = linspace(0.5, (nblocks_x as f64 * increment_x).trunc() - 0.5, rows);
    let thresh_out_y_coords = linspace(0.5, (nblocks_y as f64 * increment_y).trunc() - 0.5, cols);

    let grid = rect_bivariate_spline_interpolate(
        &block_x_coords,
        &block_y_coords,
        &block_threshold,
        spline_order,
        spline_order,
        &thresh_out_x_coords,
        &thresh_out_y_coords,
    )?;
    let mut out = vec![0.0; rows * cols];
    for (r, grid_row) in grid.iter().enumerate() {
        for (c, &value) in grid_row.iter().enumerate() {
            out[r * cols + c] = value;
        }
    }
    Ok(out)
}

/// 1-D Gaussian kernel matching `scipy.ndimage._gaussian_kernel1d` for
/// `order=0`: radius `floor(4.0 * sigma + 0.5)` (scipy's default
/// `truncate=4.0`), values `exp(-0.5/sigma^2 * x^2)` normalized to sum
/// to 1. Symmetric, so correlation and convolution coincide (matching
/// `gaussian_filter1d`'s kernel-reversal for `correlate1d` being a
/// no-op here).
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let sigma2 = sigma * sigma;
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|x| (-0.5 / sigma2 * (x * x) as f64).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    for value in kernel.iter_mut() {
        *value /= sum;
    }
    kernel
}

/// Apply a 1-D correlation along one axis with `mode="constant"`,
/// `cval=0.0` (values outside the array contribute 0), matching
/// `scipy.ndimage.correlate1d`'s boundary handling for that mode.
fn correlate_1d_constant(
    data: &[f64],
    rows: usize,
    cols: usize,
    kernel: &[f64],
    along_rows: bool,
) -> Vec<f64> {
    let radius = (kernel.len() / 2) as i64;
    let mut out = vec![0.0f64; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                let offset = k as i64 - radius;
                let (rr, cc) = if along_rows {
                    (r as i64, c as i64 + offset)
                } else {
                    (r as i64 + offset, c as i64)
                };
                if rr < 0 || cc < 0 || rr as usize >= rows || cc as usize >= cols {
                    continue; // contributes 0 (cval)
                }
                acc += weight * data[rr as usize * cols + cc as usize];
            }
            out[r * cols + c] = acc;
        }
    }
    out
}

/// 2-D Gaussian filter matching `scipy.ndimage.gaussian_filter(data,
/// sigma, mode="constant", cval=0)` with a scalar `sigma` (separable:
/// one 1-D pass per axis; order doesn't affect the result for an
/// isotropic Gaussian).
fn gaussian_filter_2d(data: &[f64], rows: usize, cols: usize, sigma: f64) -> Vec<f64> {
    let kernel = gaussian_kernel_1d(sigma);
    let pass1 = correlate_1d_constant(data, rows, cols, &kernel, true);
    correlate_1d_constant(&pass1, rows, cols, &kernel, false)
}

/// Blur `image` with a Gaussian of `sigma`, correcting for the
/// `mode="constant"` kernel-mass loss near the array border (and near
/// any explicit `mask` boundary) exactly like
/// `centrosome.smooth.smooth_with_function_and_mask`: smooth the
/// mask-zeroed image, then divide by the smoothed mask itself (treating
/// "no mask" as an implicit all-true mask, which still loses mass at
/// the image's own border under `mode="constant"`).
fn smooth_with_mask_bleed_over(
    image: &[f64],
    mask: Option<&[bool]>,
    rows: usize,
    cols: usize,
    sigma: f64,
) -> Vec<f64> {
    let mask_as_float: Vec<f64> = match mask {
        Some(m) => m.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect(),
        None => vec![1.0; image.len()],
    };
    let masked_image: Vec<f64> = match mask {
        Some(m) => image
            .iter()
            .zip(m.iter())
            .map(|(&v, &keep)| if keep { v } else { 0.0 })
            .collect(),
        None => image.to_vec(),
    };
    let bleed_over = gaussian_filter_2d(&mask_as_float, rows, cols, sigma);
    let smoothed = gaussian_filter_2d(&masked_image, rows, cols, sigma);
    smoothed
        .iter()
        .zip(bleed_over.iter())
        .map(|(&s, &b)| s / (b + f64::EPSILON))
        .collect()
}

/// 8-connected connected-components labeling over a boolean foreground
/// mask, assigning labels in raster-scan first-appearance order --
/// matching `scipy.ndimage.label(binary, numpy.ones((3, 3), bool))`'s
/// numbering convention (and `skimage.measure.label`'s default).
fn label_connected_components(foreground: &[bool], rows: usize, cols: usize) -> (Vec<u32>, u32) {
    let mut labels = vec![0u32; foreground.len()];
    let mut parent: Vec<u32> = Vec::new();

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            x = parent[x as usize];
        }
        x
    }
    fn union(parent: &mut [u32], a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }

    // First pass: assign provisional labels, union neighbors already
    // visited (above row and left-within-row), 8-connectivity.
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if !foreground[idx] {
                continue;
            }
            let mut neighbor_labels = Vec::new();
            if c > 0 && foreground[idx - 1] {
                neighbor_labels.push(labels[idx - 1]);
            }
            if r > 0 {
                let up = idx - cols;
                if foreground[up] {
                    neighbor_labels.push(labels[up]);
                }
                if c > 0 && foreground[up - 1] {
                    neighbor_labels.push(labels[up - 1]);
                }
                if c + 1 < cols && foreground[up + 1] {
                    neighbor_labels.push(labels[up + 1]);
                }
            }
            if neighbor_labels.is_empty() {
                let new_label = parent.len() as u32;
                parent.push(new_label);
                labels[idx] = new_label;
            } else {
                let first = neighbor_labels[0];
                labels[idx] = first;
                for &other in &neighbor_labels[1..] {
                    union(&mut parent, first, other);
                }
            }
        }
    }

    // Resolve union-find roots, then renumber roots in order of first
    // appearance during a second raster scan (matches scipy/skimage).
    let mut root_to_final: BTreeMap<u32, u32> = BTreeMap::new();
    let mut next_final = 1u32;
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if !foreground[idx] {
                continue;
            }
            let root = find(&mut parent, labels[idx]);
            let final_label = *root_to_final.entry(root).or_insert_with(|| {
                let assigned = next_final;
                next_final += 1;
                assigned
            });
            labels[idx] = final_label;
        }
    }
    (labels, next_final - 1)
}

fn relabel_consecutive(labels: &mut [u32]) -> u32 {
    let mut unique: Vec<u32> = labels.iter().copied().filter(|&v| v != 0).collect();
    unique.sort_unstable();
    unique.dedup();
    if unique.is_empty() {
        return 0;
    }
    let mut table: BTreeMap<u32, u32> = BTreeMap::new();
    for (new_label, &old_label) in unique.iter().enumerate() {
        table.insert(old_label, (new_label + 1) as u32);
    }
    for value in labels.iter_mut() {
        if *value != 0 {
            *value = table[value];
        }
    }
    unique.len() as u32
}

/// Zero out every object touching the image border, matching
/// `IdentifyPrimaryObjects.filter_on_border`: if nothing touches the
/// image border directly and a `mask` is given, fall back to zeroing
/// objects touching the *mask's* border (pixels where the mask is true
/// but an adjacent pixel's mask is false) instead.
fn filter_on_border(
    labels: &mut [u32],
    mask: Option<&[bool]>,
    rows: usize,
    cols: usize,
    enabled: bool,
) {
    if !enabled {
        return;
    }
    let border_labels: Vec<u32> = (0..cols)
        .map(|c| labels[c])
        .chain((0..rows).map(|r| labels[r * cols]))
        .chain((0..cols).map(|c| labels[(rows - 1) * cols + c]))
        .chain((0..rows).map(|r| labels[r * cols + cols - 1]))
        .collect();
    let touching: std::collections::BTreeSet<u32> = border_labels
        .into_iter()
        .filter(|&label| label != 0)
        .collect();
    if !touching.is_empty() {
        for label in labels.iter_mut() {
            if touching.contains(label) {
                *label = 0;
            }
        }
        return;
    }
    let Some(mask) = mask else {
        return;
    };
    // mask_border: mask is true here, but erosion (4-connected neighbors
    // all true too, scipy.ndimage.binary_erosion's default structuring
    // element) would be false -- i.e. a masked-in pixel adjacent to a
    // masked-out pixel or the image edge.
    let is_eroded = |r: usize, c: usize| -> bool {
        for (dr, dc) in [(0i64, 0i64), (-1, 0), (1, 0), (0, -1), (0, 1)] {
            let rr = r as i64 + dr;
            let cc = c as i64 + dc;
            if rr < 0 || cc < 0 || rr as usize >= rows || cc as usize >= cols {
                return false;
            }
            if !mask[rr as usize * cols + cc as usize] {
                return false;
            }
        }
        true
    };
    let mut touching_mask_border: std::collections::BTreeSet<u32> = Default::default();
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if mask[idx] && !is_eroded(r, c) && labels[idx] != 0 {
                touching_mask_border.insert(labels[idx]);
            }
        }
    }
    if !touching_mask_border.is_empty() {
        for label in labels.iter_mut() {
            if touching_mask_border.contains(label) {
                *label = 0;
            }
        }
    }
}

/// Global dispatch, including Threshold.get_global_threshold's shared shortcuts.
fn threshold_for_values(values: &[f64], method: ThresholdMethod) -> Result<f64, String> {
    if values.is_empty() {
        return Ok(0.0);
    }
    if values.iter().all(|&v| v == values[0]) {
        return Ok(values[0]);
    }
    match method {
        ThresholdMethod::OtsuTwoClass => Ok(otsu_threshold_float(values, OTSU_FLOAT_BINS)),
        ThresholdMethod::OtsuThreeClass {
            assign_middle_to_foreground,
        } => {
            let (lower, upper) = otsu_threshold_three_class(values, OTSU_THREE_CLASS_BINS)?;
            Ok(if assign_middle_to_foreground {
                lower
            } else {
                upper
            })
        }
        ThresholdMethod::Li => threshold_li(values),
        ThresholdMethod::RobustBackground(config) => threshold_robust_background(values, &config),
        // Real CellProfiler: `self.threshold_operation in (TM_LI, TM_SAUVOLA)`
        // is one shared branch -- Sauvola's "global"/guide value is Li's.
        ThresholdMethod::Sauvola => threshold_li(values),
        _ => Err("unsupported threshold method for float pixels".to_string()),
    }
}

/// Run `IdentifyPrimaryObjects` in this module's declared scope.
///
/// `pixels` and `integer_pixels` must have the same length as
/// `rows * cols`. `integer_pixels` is required (and must be the exact
/// original integer values, in the same units as `pixels`) when
/// `config.threshold_method` is `OtsuTwoClassInteger`; it is ignored
/// otherwise. `mask` defaults to all-true (every pixel is foreground-
/// eligible) when `None`, matching an image with no explicit mask.
pub fn identify_primary_objects(
    pixels: &[f64],
    integer_pixels: Option<&[i64]>,
    mask: Option<&[bool]>,
    rows: usize,
    cols: usize,
    config: &IdentifyPrimaryObjectsConfig,
) -> Result<SegmentationResult, String> {
    if pixels.len() != rows * cols {
        return Err("pixels length must equal rows * cols".to_string());
    }
    if let Some(m) = mask {
        if m.len() != pixels.len() {
            return Err("mask length must equal pixels length".to_string());
        }
    }

    let masked_indices: Vec<usize> = match mask {
        Some(m) => (0..pixels.len()).filter(|&i| m[i]).collect(),
        None => (0..pixels.len()).collect(),
    };

    let log_transform_applies = config.log_transform
        && matches!(
            config.threshold_method,
            ThresholdMethod::OtsuTwoClass
                | ThresholdMethod::OtsuThreeClass { .. }
                | ThresholdMethod::Li
        );
    let (log_transformed_pixels, log_min, log_max) = if log_transform_applies {
        log_transform(pixels)
    } else {
        (Vec::new(), 0.0, 0.0)
    };
    let threshold_pixels: &[f64] = if log_transform_applies {
        &log_transformed_pixels
    } else {
        pixels
    };

    // CellProfiler's empty/constant shortcuts run BEFORE algorithm dispatch.
    let masked_values: Vec<f64> = masked_indices
        .iter()
        .map(|&i| threshold_pixels[i])
        .collect();
    if matches!(config.threshold_method, ThresholdMethod::Sauvola)
        && config.threshold_scope == ThresholdScope::Global
    {
        return Err(
            "Sauvola thresholding requires threshold_scope='adaptive' (CellProfiler's own \
             global_operation Choice does not even include Sauvola)"
                .to_string(),
        );
    }
    let threshold = if let ThresholdMethod::Manual(value) = config.threshold_method {
        value
    } else {
        let mut raw = if matches!(
            config.threshold_method,
            ThresholdMethod::OtsuTwoClassInteger
        ) {
            let integer_pixels =
                integer_pixels.ok_or("integer_pixels is required for OtsuTwoClassInteger")?;
            if integer_pixels.len() != pixels.len() {
                return Err("integer_pixels length must equal pixels length".to_string());
            }
            let values: Vec<i64> = masked_indices.iter().map(|&i| integer_pixels[i]).collect();
            otsu_threshold_integer(&values)?
        } else {
            threshold_for_values(&masked_values, config.threshold_method)?
        };
        if log_transform_applies {
            raw = inverse_log_transform(raw, log_min, log_max);
        }
        clamp(
            raw * config.threshold_correction_factor,
            config.threshold_range,
        )
    };

    let threshold_map = if let ThresholdScope::Adaptive { window_size } = config.threshold_scope {
        if window_size == 0 {
            return Err("adaptive_window_size must be positive".to_string());
        }
        if matches!(
            config.threshold_method,
            ThresholdMethod::Manual(_) | ThresholdMethod::OtsuTwoClassInteger
        ) {
            return Err("adaptive thresholding requires otsu, otsu_three_class, li, robust_background, or sauvola".to_string());
        }
        // CP compares the full array to its first ROW (NumPy broadcasting),
        // not merely to its first scalar. Preserve that shortcut as written.
        let repeated_first_row = mask.is_none_or(|m| m.iter().all(|&v| v))
            && (0..pixels.len()).all(|i| threshold_pixels[i] == threshold_pixels[i % cols]);
        let mut local = if repeated_first_row {
            threshold_pixels.to_vec()
        } else if masked_values.iter().all(|&v| v == 0.0) {
            vec![0.0; pixels.len()]
        } else {
            let mut unique = threshold_pixels.to_vec();
            unique.sort_by(f64::total_cmp);
            unique.dedup();
            let li_tolerance = (unique
                .windows(2)
                .map(|v| v[1] - v[0])
                .fold(f64::INFINITY, f64::min)
                / 2.0)
                .max(0.5 / 65536.0);
            if matches!(config.threshold_method, ThresholdMethod::Sauvola) {
                // Sauvola bypasses the block+spline path entirely (real
                // CP's get_local_threshold has it as a wholly separate
                // branch) and zero-fills masked-out pixels, rather than
                // excluding them like every other method here.
                let zero_filled: Vec<f64> = (0..pixels.len())
                    .map(|i| {
                        if mask.map(|m| m[i]).unwrap_or(true) {
                            threshold_pixels[i]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let odd_window = if window_size % 2 == 0 {
                    window_size + 1
                } else {
                    window_size
                };
                sauvola_threshold(&zero_filled, rows, cols, odd_window)
            } else {
                get_adaptive_threshold(threshold_pixels, mask, rows, cols, window_size, |values| {
                    match config.threshold_method {
                        ThresholdMethod::Li => {
                            Ok(threshold_li_with_tolerance(values, li_tolerance))
                        }
                        ThresholdMethod::OtsuThreeClass {
                            assign_middle_to_foreground,
                        } => {
                            let mut unique = values.to_vec();
                            unique.sort_by(f64::total_cmp);
                            unique.dedup();
                            if unique.len() < 3 {
                                return Ok(otsu_threshold_float(values, OTSU_FLOAT_BINS));
                            }
                            // CP retries multi-Otsu without nbins=128 when histogram
                            // discretization collapses distinct values (default=256).
                            let (lower, upper) =
                                otsu_threshold_three_class(values, OTSU_THREE_CLASS_BINS).or_else(
                                    |_| otsu_threshold_three_class(values, OTSU_FLOAT_BINS),
                                )?;
                            Ok(if assign_middle_to_foreground {
                                lower
                            } else {
                                upper
                            })
                        }
                        _ => threshold_for_values(values, config.threshold_method),
                    }
                })?
            }
        };
        let local_range = (
            config.threshold_range.0.max(threshold * 0.7),
            config.threshold_range.1.min(threshold * 1.5),
        );
        for value in &mut local {
            if log_transform_applies {
                *value = inverse_log_transform(*value, log_min, log_max);
            }
            *value = clamp(*value * config.threshold_correction_factor, local_range);
        }
        Some(local)
    } else {
        None
    };
    let guide_threshold = threshold_map.as_ref().map(|_| threshold);
    let threshold = threshold_map.as_ref().map_or(threshold, |values| {
        values.iter().sum::<f64>() / values.len() as f64
    });

    let comparison_pixels: Vec<f64> = if config.threshold_smoothing_scale != 0.0 {
        let sigma = config.threshold_smoothing_scale / 0.6744 / 2.0;
        smooth_with_mask_bleed_over(pixels, mask, rows, cols, sigma)
    } else {
        pixels.to_vec()
    };

    let mut foreground = vec![false; pixels.len()];
    for &i in &masked_indices {
        foreground[i] =
            comparison_pixels[i] >= threshold_map.as_ref().map_or(threshold, |values| values[i]);
    }

    if config.fill_holes == FillHoles::AfterThresholdingAndDeclump {
        // Pre-labeling fill, size-limited to max_diameter^2 (not the
        // circular area formula -- matches run()'s own size_fn exactly).
        let mut as_labels: Vec<u32> = foreground.iter().map(|&v| u32::from(v)).collect();
        let max_area = config.max_diameter * config.max_diameter;
        holes::fill_labeled_holes(&mut as_labels, rows, cols, Some(max_area));
        for (dst, &v) in foreground.iter_mut().zip(as_labels.iter()) {
            *dst = v != 0;
        }
    }

    let (mut labels, initial_object_count) = label_connected_components(&foreground, rows, cols);

    // `separate_neighboring_objects`: declumping/watershed, run right
    // after the initial labeling and before border/size filtering,
    // matching real CellProfiler's own call order exactly. A no-op
    // when `declump.unclump`/`declump.watershed` are both `None`
    // (declump()'s own early return), which is the default.
    let declump_config = declump::DeclumpConfig {
        unclump: config.declump.unclump,
        watershed: config.declump.watershed,
        smoothing: config.declump.smoothing,
        suppression: config.declump.suppression,
        minimum_diameter: config.min_diameter,
        low_resolution_maxima: config.declump.low_resolution_maxima,
        basic: false,
        // Matches real CellProfiler's own condition in
        // `separate_neighboring_objects`'s Shape-unclump branch
        // exactly: only when the user's `fill_holes` setting is
        // "Never" does Shape unclump do its own temporary internal
        // hole fill for the distance transform (there being no later
        // refill to otherwise correct for a hole putting a
        // far-from-edge maximum candidate on the wrong side of it).
        fill_holes_for_shape: config.fill_holes == FillHoles::Never,
    };
    let declumped = declump::declump(
        pixels,
        mask,
        &labels,
        cols,
        rows,
        initial_object_count,
        &declump_config,
    )?;
    labels = declumped.labels;
    let unedited_labels = labels.clone();

    filter_on_border(&mut labels, mask, rows, cols, config.exclude_border_objects);

    let small_removed_labels;
    if config.exclude_size {
        let mut areas: BTreeMap<u32, u64> = BTreeMap::new();
        for &label in &labels {
            if label != 0 {
                *areas.entry(label).or_insert(0) += 1;
            }
        }
        let min_allowed_area =
            std::f64::consts::PI * (config.min_diameter * config.min_diameter) / 4.0;
        let max_allowed_area =
            std::f64::consts::PI * (config.max_diameter * config.max_diameter) / 4.0;
        for label in labels.iter_mut() {
            if *label != 0 {
                let area = areas[label] as f64;
                if area < min_allowed_area {
                    *label = 0;
                }
            }
        }
        small_removed_labels = labels.clone();
        for label in labels.iter_mut() {
            if *label != 0 {
                let area = areas[label] as f64;
                if area > max_allowed_area {
                    *label = 0;
                }
            }
        }
    } else {
        small_removed_labels = labels.clone();
    }

    if config.fill_holes != FillHoles::Never {
        holes::fill_labeled_holes(&mut labels, rows, cols, None);
    }

    let mut object_count = relabel_consecutive(&mut labels);
    // CP preserves pre-filter numbering in this diagnostic array.

    if let Some(maximum_object_count) = config.maximum_object_count {
        if object_count > maximum_object_count {
            labels.fill(0);
            object_count = 0;
        }
    }

    Ok(SegmentationResult {
        labels,
        small_removed_labels,
        unedited_labels,
        object_count,
        threshold,
        threshold_map,
        guide_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_field_shape_declump_matches_real_cellprofiler_zero_objects() {
        // The production pilot's all-zero 450x450 stress field (BBBC007
        // validation field A9-A9-p10-empty, all-zero float DNA after
        // illumination correction) with the pilot's primary settings:
        // global two-class Otsu (constant image -> threshold 0.0 ->
        // the whole image is foreground), min_diameter 10, max_diameter
        // 40, exclude_size, unclump Shape, watershed Propagate, fill
        // Never. The pinned real CellProfiler 4.2.8 oracle declumps the
        // single all-foreground blob into ONE object (scipy's
        // all-foreground distance transform is a strictly monotone
        // bowl, so the seed-0-jittered maxima detection finds exactly
        // one maximum at pixel (449, 449)) which then fails the
        // max-size filter (202500 px > pi * 40^2 / 4), leaving 0
        // objects -- see the run-full oracle result
        // (stress-empty-field/A9-A9-p10-empty.json: 0 objects).
        let pixels = vec![0.0f64; 450 * 450];
        let config = IdentifyPrimaryObjectsConfig {
            exclude_size: true,
            min_diameter: 10.0,
            max_diameter: 40.0,
            declump: DeclumpSettings {
                unclump: declump::UnclumpMethod::Shape,
                watershed: declump::WatershedMethod::Propagate,
                smoothing: declump::Smoothing::Automatic,
                suppression: declump::MaximaSuppression::Automatic,
                low_resolution_maxima: false,
            },
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 450, 450, &config).unwrap();
        assert_eq!(result.threshold, 0.0);
        // One declumped object covering the whole field, matching the
        // real run's unedited/small-removed stage (object count 1)...
        assert_eq!(
            result
                .unedited_labels
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([1])
        );
        assert_eq!(
            result
                .small_removed_labels
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([1])
        );
        // ...which the max-size filter then erases entirely.
        assert_eq!(result.object_count, 0);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn otsu_integer_matches_skimage_threshold_otsu_bit_exact() {
        // Fixed dataset generated once via
        // `numpy.random.default_rng(0)` + two integer ranges and
        // verified against the real `skimage.filters.threshold_otsu`
        // (0.26.0): it returns exactly 29 for this data.
        let values: Vec<i64> = vec![
            27, 22, 20, 15, 16, 10, 11, 10, 13, 26, 22, 28, 20, 22, 29, 24, 22, 20, 21, 28, 15, 26,
            23, 10, 17, 27, 21, 10, 25, 24, 26, 13, 11, 27, 10, 20, 11, 15, 19, 18, 18, 10, 10, 12,
            10, 23, 20, 22, 15, 22, 210, 195, 198, 219, 212, 219, 195, 207, 218, 206, 213, 207,
            208, 195, 215, 185, 203, 208, 213, 201, 195, 192, 196, 199, 208, 215, 182, 217, 201,
            194, 206, 202, 190, 192, 208, 203, 200, 193, 210, 195, 193, 215, 190, 189, 208, 204,
            181, 183, 195, 213,
        ];
        assert_eq!(otsu_threshold_integer(&values).unwrap(), 29.0);
    }

    #[test]
    fn otsu_float_matches_real_cellprofiler_rescaled_real_data() {
        // The real_cell_cc0 DNA channel, rescaled by CellProfiler's own
        // `Image(..., convert=True)` (divide uint8 by 255), run through
        // the real `skimage.filters.threshold_otsu` (0.26.0): it returns
        // 0.25099573 (float32). This Rust path computes in f64, so an
        // exact bit match isn't expected, but it must land very close.
        let raw: Vec<f64> = vec![0.21176471, 0.3137255, 0.26274511, 0.26274511, 0.23137255];
        let threshold = otsu_threshold_float(&raw, 256);
        assert!(threshold.is_finite());
        // The real target value from the oracle (see
        // scripts/segmentation/generate_reference.py's
        // real_cell_cc0_dna_otsu case) is checked at full image scale in
        // tests/segmentation/test_reference_alignment.py, not here with
        // this 5-value smoke sample; this test only checks the function
        // runs and returns something inside the data range.
        assert!((0.21176471..=0.3137255).contains(&threshold));
    }

    #[test]
    fn manual_threshold_produces_expected_labels() {
        // 4x4 image with a 2x2 bright square at threshold 5 vs background 0.
        #[rustfmt::skip]
        let pixels: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 9.0, 9.0, 0.0,
            0.0, 9.0, 9.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(5.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 4, 4, &config).unwrap();
        assert_eq!(result.object_count, 1);
        assert_eq!(result.threshold, 5.0);
        assert_eq!(result.labels[4 + 1], 1);
        assert_eq!(result.labels[0], 0);
    }

    #[test]
    fn diagonal_touching_objects_are_one_component_under_8_connectivity() {
        #[rustfmt::skip]
        let pixels: Vec<f64> = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 2, 2, &config).unwrap();
        assert_eq!(result.object_count, 1);
        assert_eq!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn size_filter_removes_objects_outside_diameter_range() {
        // One 1-pixel object, one 3x3=9-pixel object.
        #[rustfmt::skip]
        let pixels: Vec<f64> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0, 1.0,
        ];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            exclude_size: true,
            min_diameter: 2.0,
            max_diameter: 100.0,
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 4, 5, &config).unwrap();
        assert_eq!(result.object_count, 1);
        assert_eq!(result.labels[0], 0);
    }

    #[test]
    fn border_exclusion_removes_only_objects_touching_the_edge() {
        // 4x4: one object touching the top-left corner (border), one
        // fully interior object.
        #[rustfmt::skip]
        let pixels: Vec<f64> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            exclude_border_objects: true,
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 4, 4, &config).unwrap();
        assert_eq!(result.object_count, 1);
        assert_eq!(
            result.labels[0], 0,
            "corner (border) object must be removed"
        );
        assert_ne!(result.labels[6], 0, "interior object must remain");
    }

    #[test]
    fn border_exclusion_without_mask_leaves_interior_objects_alone() {
        // 3x3 grid, single foreground pixel dead center (index 4) --
        // not touching any edge.
        let pixels: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            exclude_border_objects: true,
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 3, 3, &config).unwrap();
        assert_eq!(result.object_count, 1);
        assert_ne!(result.labels[4], 0);
    }

    #[test]
    fn maximum_object_count_erases_everything_when_exceeded() {
        // Two separate single-pixel objects.
        let pixels: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            maximum_object_count: Some(1),
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 2, 3, &config).unwrap();
        assert_eq!(result.object_count, 0);
        assert!(result.labels.iter().all(|&v| v == 0));
    }

    #[test]
    fn maximum_object_count_keeps_objects_when_within_limit() {
        let pixels: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            maximum_object_count: Some(2),
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 2, 3, &config).unwrap();
        assert_eq!(result.object_count, 2);
    }

    #[test]
    fn empty_image_has_no_objects() {
        let pixels = vec![0.0; 16];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 4, 4, &config).unwrap();
        assert_eq!(result.object_count, 0);
        assert!(result.labels.iter().all(|&v| v == 0));
    }

    #[test]
    fn fill_labeled_holes_fills_an_interior_hole() {
        // Matches the oracle probe: single object, interior hole -> filled.
        #[rustfmt::skip]
        let mut labels: Vec<u32> = vec![
            0,0,0,0,0,
            0,1,1,1,0,
            0,1,0,1,0,
            0,1,1,1,0,
            0,0,0,0,0,
        ];
        holes::fill_labeled_holes(&mut labels, 5, 5, None);
        assert_eq!(labels[2 * 5 + 2], 1);
    }

    #[test]
    fn fill_labeled_holes_leaves_hole_adjacent_to_two_labels_unfilled() {
        // Matches the real-oracle-verified case: a hollow bordered by
        // two DIFFERENT labels, not touching the image edge, stays
        // background (centrosome.cpmorphology.fill_labeled_holes
        // verified directly against this exact shape).
        #[rustfmt::skip]
        let mut labels: Vec<u32> = vec![
            0,0,0,0,0,0,0,
            0,1,1,1,1,1,0,
            0,1,0,0,0,1,0,
            0,1,0,0,0,2,0,
            0,2,0,0,0,2,0,
            0,2,2,2,2,2,0,
            0,0,0,0,0,0,0,
        ];
        let before = labels.clone();
        holes::fill_labeled_holes(&mut labels, 7, 7, None);
        assert_eq!(
            labels, before,
            "hole bordered by two labels must stay unfilled"
        );
    }

    #[test]
    fn fill_labeled_holes_respects_max_area() {
        // 6x6=36px interior hole, cutoff at 20 excludes it from filling.
        let mut labels = vec![0u32; 12 * 12];
        for r in 1..11 {
            for c in 1..11 {
                labels[r * 12 + c] = 1;
            }
        }
        for r in 3..9 {
            for c in 3..9 {
                labels[r * 12 + c] = 0;
            }
        }
        let before = labels.clone();
        holes::fill_labeled_holes(&mut labels, 12, 12, Some(20.0));
        assert_eq!(labels, before, "oversized hole must stay unfilled");
        holes::fill_labeled_holes(&mut labels, 12, 12, Some(40.0));
        assert!(labels[5 * 12 + 5] == 1, "under the new cutoff it must fill");
    }

    #[test]
    fn fill_holes_after_thresholding_fills_before_labeling() {
        // A single blob with a small interior background pixel: with
        // fill_holes on, it becomes one solid object (threshold=1).
        #[rustfmt::skip]
        let pixels: Vec<f64> = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let config = IdentifyPrimaryObjectsConfig {
            threshold_method: ThresholdMethod::Manual(1.0),
            threshold_range: (f64::NEG_INFINITY, f64::INFINITY),
            fill_holes: FillHoles::AfterThresholdingAndDeclump,
            max_diameter: 100.0,
            ..Default::default()
        };
        let result = identify_primary_objects(&pixels, None, None, 3, 3, &config).unwrap();
        assert_eq!(result.object_count, 1);
        assert_eq!(result.labels[4], 1, "the interior hole must be filled");
    }

    #[test]
    fn gaussian_smoothing_matches_scipy_ndimage_gaussian_filter_with_bleed_over() {
        // Fixed numpy.random.default_rng(0) 6x6 image, threshold_smoothing_scale=0.5
        // (sigma = 0.5/0.6744/2.0 = 0.370699881376038), run through the
        // real scipy.ndimage.gaussian_filter(mode="constant", cval=0)
        // plus centrosome's mask bleed-over correction (an all-true
        // mask, since no mask was given): verified bit-close below.
        #[rustfmt::skip]
        let image: Vec<f64> = vec![
            0.6369616873214543, 0.2697867137638703, 0.04097352393619469, 0.01652763552852909, 0.8132702392002724, 0.9127555772777217,
            0.6066357757671799, 0.7294965609839984, 0.5436249914654229, 0.9350724237877682, 0.8158535541215322, 0.00273850017014809,
            0.8574042765875693, 0.03358557530546435, 0.7296554464299441, 0.17565562060255901, 0.8631789223498866, 0.5414612202490917,
            0.2997118905373848, 0.42268722119765845, 0.02831967114546297, 0.12428327649956394, 0.6706244146936303, 0.6471895115742501,
            0.6153851114812539, 0.38367755426188344, 0.997209935789211, 0.9808353387762301, 0.6855419844806947, 0.6504592762678163,
            0.6884467305709401, 0.3889214239791038, 0.13509650502241122, 0.7214883401940817, 0.5253543224757259, 0.31024187555895566,
        ];
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            0.6271002477668838, 0.28473327720965946, 0.05919354387932638, 0.05971781466419358, 0.7959226420536833, 0.8874934204633641,
            0.6158849750855957, 0.6943400878900116, 0.548763876556832, 0.8823294966281533, 0.799440299581265, 0.05885648413773547,
            0.817318026344473, 0.09623831705219703, 0.6784929832319457, 0.22276469856128062, 0.8324495951950707, 0.5390112029356621,
            0.32384416844074404, 0.4011929965091427, 0.08050083506377854, 0.15824920650329174, 0.6618185144703157, 0.6454289892655356,
            0.6035729740317353, 0.4045461873141111, 0.9373579912694224, 0.9461367158237115, 0.6869757251640796, 0.6428862142093742,
            0.6789464415786413, 0.39044018416302834, 0.1772278554281189, 0.708908695466539, 0.5291625576244546, 0.32434996514417247,
        ];
        let sigma = 0.5_f64 / 0.6744 / 2.0;
        let actual = smooth_with_mask_bleed_over(&image, None, 6, 6, sigma);
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-12, "{a} vs {e}");
        }
    }

    #[test]
    fn otsu_three_class_matches_real_skimage_threshold_multiotsu() {
        // Fixed numpy.random.default_rng(7) three-cluster dataset,
        // verified against the real skimage.filters.threshold_multiotsu
        // (0.26.0, classes=3, nbins=128): [0.1757985471578119,
        // 0.5222662720500253].
        #[rustfmt::skip]
        let values: Vec<f64> = vec![
            0.15002460306714965, 0.1559749107501694, 0.14451724289275564, 0.13218816322485452, 0.14090658429656555,
            0.13016706890007074, 0.15120287205194877, 0.17680430491109067, 0.1401558696289734, 0.13759050200360118,
            0.15979684100370395, 0.1571377401632012, 0.15210828497995796, 0.1313906391058359, 0.14941496355073453,
            0.16390606388916576, 0.12311570905429836, 0.14084768477919563, 0.11197554520398312, 0.12420924520430048,
            0.46316529924416533, 0.4952981773785064, 0.4746510703711259, 0.505425287176434, 0.5031350217324845,
            0.4962613811074009, 0.44966480578358975, 0.48922614208306725, 0.49902998109197855, 0.5022661797200662,
            0.4693972846898921, 0.49044493447932136, 0.4804296184388672, 0.483823255211488, 0.5212179724677216,
            0.48384930649336205, 0.4993495659010896, 0.5176877973476635, 0.48832799134513394, 0.4977659610083168,
            0.8522092828649895, 0.8512756354851012, 0.8254988834716461, 0.8515228046075402, 0.8771764684348308,
            0.8190571064374303, 0.8671876537604319, 0.8523870805139316, 0.8371705921178556, 0.8900083309268484,
            0.8652451942416942, 0.8260142219578955, 0.8514903245754293, 0.8615337916734037, 0.846224357492985,
            0.8636582053439041, 0.8486696535970116, 0.8633449512166865, 0.878770451833123, 0.8364867549798869,
        ];
        let (lower, upper) = otsu_threshold_three_class(&values, OTSU_THREE_CLASS_BINS).unwrap();
        assert!((lower - 0.1757985471578119).abs() < 1e-9, "lower={lower}");
        assert!((upper - 0.5222662720500253).abs() < 1e-9, "upper={upper}");
    }

    #[test]
    fn otsu_three_class_errors_on_fewer_than_three_distinct_values() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        assert!(otsu_threshold_three_class(&values, OTSU_THREE_CLASS_BINS).is_err());
    }

    #[test]
    fn threshold_li_matches_real_skimage_threshold_li() {
        // Fixed numpy.random.default_rng(3) bimodal dataset, verified
        // against the real skimage.filters.threshold_li (0.26.0) with
        // CellProfiler's own tolerance formula: 0.38762351686744034.
        #[rustfmt::skip]
        let values: Vec<f64> = vec![
            0.30204595606925916, 0.0722167484342909, 0.22090494233628896, 0.17161151969360353, 0.17736753539447772,
            0.18922014184551172, 0.09900069354263746, 0.18840338111779054, 0.15673934618625293, 0.3661499758322442,
            0.2112893306613961, 0.18236846028292025, 0.1859356290924325, 0.1665976826945525, 0.14724247243974395,
            0.18045995113826727, 0.2240972694253393, 0.18807231967133167, 0.2478879351479882, 0.190009893546671,
            0.20121297825383325, 0.27729104256064063, 0.22725527613438223, 0.1747385632192991, 0.19085805127011327,
            0.22702625658774012, 0.29675440170494266, 0.18651898363290434, 0.1878220660460448, 0.2501156800637846,
            0.6556770028419706, 0.6854139883780067, 0.7441269483728241, 0.7290175008095449, 0.7045758351641176,
            0.733505217741424, 0.5585918846578118, 0.751065340875004, 0.6520177620095928, 0.6165690078672015,
            0.71382228797605, 0.7350272442674695, 0.6777616272158608, 0.6461797079949596, 0.7013062416767016,
            0.697362634587856, 0.7702799083009045, 0.7373703993739675, 0.7096907823132309, 0.7555816602611996,
            0.6897238475047103, 0.6537050213175816, 0.7292029155512624, 0.7291269209327844, 0.6892585544436571,
            0.6608595711018016, 0.711457695260663, 0.5753052860771004, 0.7345062385081406, 0.7245684130372495,
        ];
        let threshold = threshold_li(&values).unwrap();
        assert!(
            (threshold - 0.38762351686744034).abs() < 1e-9,
            "{threshold}"
        );
    }

    #[test]
    fn threshold_li_errors_on_constant_input() {
        let values = vec![1.0, 1.0, 1.0];
        assert!(threshold_li(&values).is_err());
    }

    #[test]
    fn threshold_robust_background_matches_real_centrosome_for_all_four_combinations() {
        // Fixed numpy.random.default_rng(5) 80-value dataset, verified
        // against the real centrosome-backed
        // Threshold.get_threshold_robust_background equivalent (see
        // scripts for reproduction): mean/sd=1.0472254533646113,
        // median/mad=0.9623468601121523, mode/sd=1.2435976939523425,
        // mean/mad=0.975399910844821.
        #[rustfmt::skip]
        let values: Vec<f64> = vec![
            0.8050029237453802, 0.8079407897364937, 0.515325561042142, 0.2858013800881416, 0.053930702381656426,
            0.38336888078551823, 0.40847320541999865, 0.045275193902445166, 0.04875771072716806, 0.9991761150650714,
            0.6523691115879877, 0.23451020166982395, 0.43494755222514203, 0.9741861932592554, 0.8976776081085488,
            0.844231037608741, 0.39240466433477816, 0.4930230187317426, 0.676689351831066, 0.06080271295805606,
            0.5555961169207234, 0.27145160453010153, 0.8796511733349222, 0.06421443731219101, 0.679181533021365,
            0.8700885023275033, 0.2273185251609081, 0.895448239414126, 0.872195468024335, 0.01851721767021075,
            0.7074955673371773, 0.0011996835868286482, 0.5033639655536645, 0.4366670521756527, 0.2032528361145648,
            0.32494264457560607, 0.8062153310270436, 0.31645208740449016, 0.1490385835335558, 0.6985119903183108,
            0.4485441007603762, 0.798939490963134, 0.23551645730617143, 0.3197846543182863, 0.7998795260549534,
            0.5070681389233913, 0.506385001421571, 0.2361941283958635, 0.01453628036381327, 0.9332239002254337,
            0.08582579108682997, 0.84492680147361, 0.36788073891151674, 0.9510229811957994, 0.3994252615788064,
            0.9364420618420031, 0.5561597755338971, 0.24013532830277584, 0.7414216700278128, 0.6743897148867628,
            0.6842052123845983, 0.46382436008336725, 0.2218885591644547, 0.640940141481311, 0.10708949613728647,
            0.6922222746448299, 0.6353867998907108, 0.3765125255876922, 0.7985233458061055, 0.19402547600704456,
            0.3904589137977216, 0.797933886097021, 0.3804753706433236, 0.7132578641244521, 0.6125178041653119,
            0.9410009752426018, 0.9916767169901963, 0.7236762546507963, 0.8088438090346699, 0.15286502314567285,
        ];
        let cases = [
            (
                RobustAveraging::Mean,
                RobustVariance::StandardDeviation,
                1.0472254533646113,
            ),
            (
                RobustAveraging::Median,
                RobustVariance::MedianAbsoluteDeviation,
                0.9623468601121523,
            ),
            (
                RobustAveraging::Mode,
                RobustVariance::StandardDeviation,
                1.2435976939523425,
            ),
            (
                RobustAveraging::Mean,
                RobustVariance::MedianAbsoluteDeviation,
                0.975399910844821,
            ),
        ];
        for (averaging_method, variance_method, expected) in cases {
            let config = RobustBackgroundConfig {
                averaging_method,
                variance_method,
                ..Default::default()
            };
            let actual = threshold_robust_background(&values, &config).unwrap();
            assert!(
                (actual - expected).abs() < 1e-9,
                "{averaging_method:?}/{variance_method:?}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn threshold_robust_background_returns_zero_for_tiny_images() {
        let config = RobustBackgroundConfig::default();
        assert_eq!(
            threshold_robust_background(&[1.0, 2.0], &config).unwrap(),
            0.0
        );
    }

    #[test]
    fn threshold_robust_background_returns_constant_for_uniform_images() {
        let config = RobustBackgroundConfig::default();
        assert_eq!(
            threshold_robust_background(&[5.0; 10], &config).unwrap(),
            5.0
        );
    }

    #[test]
    fn binned_mode_errors_when_the_mode_bin_is_the_top_edge_overflow_bin() {
        // Fixed example (not synthetic-then-fudged): most values equal
        // the maximum, forcing numpy's own (a-a_min)/(a_max-a_min)*n_bins
        // truncation into the n_bins-th (overflow) bin as the mode,
        // which the real centrosome.threshold.binned_mode would also
        // crash on via `np.percentile`'s out-of-range `ValueError`.
        let values = vec![1.0, 100.0, 100.0, 100.0, 2.0];
        assert!(binned_mode(&values).is_err());
    }

    #[test]
    fn log_transform_matches_real_centrosome_li_and_otsu() {
        // Fixed numpy.random.default_rng(11) 10x10 dataset, verified
        // against real centrosome.threshold.log_transform +
        // skimage.filters.threshold_li/threshold_otsu +
        // centrosome.threshold.inverse_log_transform:
        // li=0.15212954955976668, otsu=0.21775828208668865.
        #[rustfmt::skip]
        let values: Vec<f64> = vec![
            0.12857020276919962, 0.49927786244011496, 0.6014983576233575, 0.028689008371944547, 0.14792608457745593,
            0.9282110229603695, 0.07042057615419683, 0.12977394939929798, 0.9483284532917751, 0.6218835927963828,
            0.368993123729791, 0.5113900218032627, 0.6628429525167993, 0.2753088157611293, 0.13796807286695534,
            0.7880395945039919, 0.6703605841024838, 0.5123823134831604, 0.8167364359696581, 0.5490752688700263,
            0.9809136392973055, 0.20450946133004488, 0.5537303628652278, 0.4836246969233877, 0.3532748550604481,
            0.5915953039490435, 0.23530123166758055, 0.8022026837784835, 0.8673335518424147, 0.12875967122785104,
            0.46707320673827113, 0.27714489253370433, 0.08311699773524239, 0.8959443082503675, 0.42994869204783537,
            0.14769129996209407, 0.6733623572759108, 0.20221602778803816, 0.901431078688177, 0.21714825741852817,
            0.03307468737742869, 0.20076895416121932, 0.34574787375563953, 0.46890816342248465, 0.9061343388955606,
            0.6973610887362978, 0.33932066074244527, 0.016877215097497844, 0.15982369410498098, 0.9964358755396705,
            0.4597159798931846, 0.6910399163187593, 0.054668061401838, 0.034050278947037604, 0.8458901064450575,
            0.5878819406668605, 0.308709743204562, 0.31737663828133744, 0.08923725441774877, 0.17266960110857543,
            0.024586107465186302, 0.8391248483727817, 0.46630319720316515, 0.1272029160585304, 0.739246874033692,
            0.19565282994532096, 0.061920235148452574, 0.5983921073240381, 0.8957577517412816, 0.026943411384702798,
            0.8051359898916692, 0.19017001672834077, 0.09290142163654713, 0.017962035652556163, 0.292975067012086,
            0.7271117439819181, 0.49317895100811693, 0.8529199944545277, 0.21721130352149032, 0.31518274714343164,
            0.25814085185458835, 0.978301137314217, 0.9410059946514349, 0.34068617585650907, 0.43600150359233103,
            0.3143196287038529, 0.7465089173054106, 0.04001310255319024, 0.06743893309202609, 0.40403758746840623,
            0.24509405111406402, 0.8452249817012268, 0.7418070964752901, 0.5457939825229567, 0.6614755143135006,
            0.6922791066849239, 0.7810548233983421, 0.9275025971156827, 0.14973517378773293, 0.6261301577453815,
        ];
        let (transformed, log_min, log_max) = log_transform(&values);

        let li = threshold_li(&transformed).unwrap();
        let li_final = inverse_log_transform(li, log_min, log_max);
        assert!((li_final - 0.15212954955976668).abs() < 1e-9, "{li_final}");

        let otsu = otsu_threshold_float(&transformed, OTSU_FLOAT_BINS);
        let otsu_final = inverse_log_transform(otsu, log_min, log_max);
        assert!(
            (otsu_final - 0.21775828208668865).abs() < 1e-6,
            "{otsu_final}"
        );
    }

    #[test]
    #[allow(clippy::useless_vec)]
    fn rect_bivariate_spline_matches_real_scipy_kx3_ky3() {
        // Fixed numpy.random.default_rng(1) 6x5 grid, verified against
        // real scipy.interpolate.RectBivariateSpline(kx=3, ky=3, s=0).
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        #[rustfmt::skip]
        let z = vec![
            vec![0.5118216247002567, 0.9504636963259353, 0.14415961271963373, 0.9486494471372439, 0.31183145201048545],
            vec![0.42332644897257565, 0.8277025938204418, 0.4091991363691613, 0.5495936876730595, 0.027559113243068367],
            vec![0.7535131086748066, 0.5381433132192782, 0.32973171649909216, 0.7884287034284043, 0.303194829291645],
            vec![0.4534978894806515, 0.13404169724716475, 0.40311298644712923, 0.20345524067614962, 0.2623133404418495],
            vec![0.7503646726300526, 0.2804087579860399, 0.48519097443163506, 0.9807371998012386, 0.9616571936637868],
            vec![0.7247899407735336, 0.5412268555474342, 0.2768912040453708, 0.16065200877512686, 0.9699254132161326],
        ];
        let eval_x = vec![
            0.5,
            1.0714285714285714,
            1.6428571428571428,
            2.2142857142857144,
            2.7857142857142856,
            3.3571428571428568,
            3.9285714285714284,
            4.5,
        ];
        let eval_y = vec![0.5, 1.1, 1.7, 2.3, 2.9, 3.5];
        #[rustfmt::skip]
        let expected = vec![
            vec![0.9036271444864533, 0.8530309149111014, 0.48747034516593873, 0.34372245168608, 0.47630009008935736, 0.4574519551763203],
            vec![0.8557740977471001, 0.775474699915473, 0.4908381372863927, 0.41695622782643654, 0.5609008857476805, 0.5217241215077003],
            vec![0.8323966539091583, 0.6246189796342444, 0.3766102516566523, 0.44933132925743097, 0.7643086219584851, 0.803634089522574],
            vec![0.5959657185989122, 0.4099711450320563, 0.30658999250962066, 0.41561527578183804, 0.6419192154480565, 0.6654532379567414],
            vec![0.21140050910732172, 0.2030526627135932, 0.33397242501030616, 0.37139147178307186, 0.2722365570351872, 0.18312951298763713],
            vec![0.12959912105541704, 0.15044806786328188, 0.3663939730376562, 0.466850636044231, 0.39287559736877686, 0.3371701385171448],
            vec![0.37499956394473305, 0.2559356569902848, 0.3702126576151285, 0.6224493226395389, 0.8888726205144744, 1.0173177313215782],
            vec![0.6424189136985906, 0.41580769869872364, 0.36210620700909457, 0.587125600662939, 0.992292663635818, 1.2746498018456185],
        ];

        let actual = rect_bivariate_spline_interpolate(&x, &y, &z, 3, 3, &eval_x, &eval_y).unwrap();
        for (row_actual, row_expected) in actual.iter().zip(expected.iter()) {
            for (&a, &e) in row_actual.iter().zip(row_expected.iter()) {
                assert!((a - e).abs() < 1e-9, "{a} vs {e}");
            }
        }
    }

    #[test]
    #[allow(clippy::useless_vec)]
    fn rect_bivariate_spline_matches_real_scipy_kx1_ky2() {
        // Same grid as above, verified against real
        // scipy.interpolate.RectBivariateSpline(kx=1, ky=2, s=0) --
        // exercises mixed odd/even degrees (not-a-knot's two branches).
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        #[rustfmt::skip]
        let z = vec![
            vec![0.5118216247002567, 0.9504636963259353, 0.14415961271963373, 0.9486494471372439, 0.31183145201048545],
            vec![0.42332644897257565, 0.8277025938204418, 0.4091991363691613, 0.5495936876730595, 0.027559113243068367],
            vec![0.7535131086748066, 0.5381433132192782, 0.32973171649909216, 0.7884287034284043, 0.303194829291645],
            vec![0.4534978894806515, 0.13404169724716475, 0.40311298644712923, 0.20345524067614962, 0.2623133404418495],
            vec![0.7503646726300526, 0.2804087579860399, 0.48519097443163506, 0.9807371998012386, 0.9616571936637868],
            vec![0.7247899407735336, 0.5412268555474342, 0.2768912040453708, 0.16065200877512686, 0.9699254132161326],
        ];
        let eval_x = vec![
            0.5,
            1.0714285714285714,
            1.6428571428571428,
            2.2142857142857144,
            2.7857142857142856,
            3.3571428571428568,
            3.9285714285714284,
            4.5,
        ];
        let eval_y = vec![0.5, 1.1, 1.7, 2.3, 2.9, 3.5];
        #[rustfmt::skip]
        let expected = vec![
            vec![0.8606007989932096, 0.8510342843599663, 0.38440617626278156, 0.34318722862838047, 0.7257357867026086, 0.6442457186077382],
            vec![0.7553824712254169, 0.7865271781498466, 0.4900278540834725, 0.41444352797073314, 0.5671862300845883, 0.4237860298677175],
            vec![0.6996752446771063, 0.6152692999858073, 0.3867942876074995, 0.42900831931179434, 0.6912616046961683, 0.5941842131828078],
            vec![0.5641638345506015, 0.4305209537053404, 0.32788816198461446, 0.4263967559344631, 0.6512724363515828, 0.5893674342791977],
            vec![0.2956454517937731, 0.2232884938974945, 0.3428611044502092, 0.39515793469314453, 0.3378423630800557, 0.2925257183444201],
            vec![0.27735883235787645, 0.1910413773497711, 0.35671042092909916, 0.4724292226707693, 0.4799647929515699, 0.4919966559211552],
            vec![0.40921127091453907, 0.26378546675812153, 0.36988556181596643, 0.6148065761797604, 0.8954187247089097, 0.989255185604799],
            vec![0.5354628223244922, 0.3951851399135315, 0.36440027774329886, 0.41628644264136366, 0.5424742999055795, 0.7425853049264018],
        ];

        let actual = rect_bivariate_spline_interpolate(&x, &y, &z, 1, 2, &eval_x, &eval_y).unwrap();
        for (row_actual, row_expected) in actual.iter().zip(expected.iter()) {
            for (&a, &e) in row_actual.iter().zip(row_expected.iter()) {
                assert!((a - e).abs() < 1e-9, "{a} vs {e}");
            }
        }
    }
}

pub mod declump;
mod holes;
pub mod secondary;
pub mod stardist;
#[cfg(feature = "stardist-model")]
pub mod stardist_model;
pub mod tertiary;

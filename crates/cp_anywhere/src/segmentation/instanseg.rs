//! In-memory InstanSeg ONNX inference plus the native port of upstream's
//! deterministic postprocessing. Requires the `instanseg-model` feature.
//!
//! The upstream reference is `instanseg/instanseg` at the
//! `instanseg_models_v0.1.2` release pin (Apache-2.0), specifically
//! `InstanSeg_Torchscript.forward` and the helpers it calls in
//! `instanseg/utils/loss/instanseg_loss.py`, `instanseg/utils/tiling.py`,
//! and `instanseg/utils/pytorch_utils.py`. The port below follows that
//! source operation by operation (same defaults, same order) for the
//! single-channel nuclei variant (`cells_and_nuclei=False`,
//! `dim_coords=2`, `n_sigma=2`, `dim_seeds=1`), verified label-exact
//! against upstream's own `test-input`/`test-output_instance_segmentation`
//! oracle pair (see `crates/cp_anywhere/tests/instanseg_fixtures/`).
//!
//! # Declared gaps (upstream features out of this port's scope)
//!
//! - `cells_and_nuclei=True` (the two-pass cells-and-nuclei loop with
//!   `resolve_cell_and_nucleus_boundaries`) is a documented upstream
//!   variant, not implemented here; the bundled single_channel_nuclei
//!   model does not use it.
//! - `mean_threshold` and `fg_threshold` are upstream parameters the
//!   in-scope code path never uses: `forward` resolves them into
//!   locals, but with `dim_seeds == 1` the `binary_map`/`fg_threshold`
//!   masking branch is skipped, and the `mean_threshold` filtering
//!   lives in upstream's separate `merge_sparse_predictions`/training
//!   path, not in `forward`. They are neither exposed here nor silently
//!   accepted. (`min_size` IS active in the serialized graph, unlike
//!   the wheel source -- see the faithfulness notes.)
//!
//! # Faithfulness notes
//!
//! - `torch_peak_local_max`'s documented quirk ("NOT IDENTICAL, AS USES
//!   *FIRST* MAX") is replicated: each pixel keeps the first (lowest,
//!   row-major) window position attaining the window maximum.
//! - `_instanseg_padding`'s `if padx > img.shape[-2] { padx -= extra_pad }`
//!   branch is inert with `extra_pad = 0`; it is ported as this
//!   comment rather than dead code.
//! - Upstream computes everywhere in float32 (the TorchScript model's
//!   dtype and the ONNX exports'); this port is float32 throughout.
//!   Integer labels are exact, so residual float32 rounding can only
//!   matter for a pixel sitting within ~1 ulp of a decision threshold.
//! - For images smaller than the 64 px gather windows
//!   (`min(h, w) < 64` at the default `window_size = 32`), upstream
//!   crashes (negative window indices reach `sparse_coo_tensor`, which
//!   rejects them); this port returns an explicit error instead.
//! - The overlap merge replicates the SERIALIZED `single_channel_nuclei`
//!   TorchScript graph, which differs from the installed v0.1.2 wheel
//!   source in two places verified against upstream's own oracle
//!   (`test-input` -> `test-output_instance_segmentation`):
//!   (1) the merge metric is `fast_sparse_intersection_over_minimum_area`
//!   (IoMA), not `fast_sparse_iou` (the wheel's `forward` shows IoU; the
//!   released graph shows `_8 = fast_sparse_intersection_over_minimum_area`);
//!   (2) the model's own saved defaults are `seed_threshold = 0.1`,
//!   `overlap_threshold = 0.5`, and `min_size = 20` -- not the wheel
//!   source's `params.get` fallbacks (0.7 / 0.3 / 10). The saved values
//!   are the defaults here because they are what upstream's released
//!   model actually runs (and what reproduces its published test output
//!   exactly).
//!   (3) the serialized graph also applies `min_size` ACTIVELY before
//!   cleanup (`valid = sum(x >= mask_threshold) > min_size` over each
//!   object's raw classifier output, filtering the probability maps,
//!   coordinates, and centroid lists); the wheel source resolves
//!   `min_size` into a local but never uses it in `forward`. Ported
//!   here because it changes the oracle result (it removes the two
//!   sub-threshold border peaks for upstream's own test image).

use std::io::Cursor;
use std::sync::Arc;

use tract_onnx::{prelude::*, tract_core::internal::ensure};

// generate_coordinate_map's width * 64 / 256 coordinate scaling.
const COORD_SCALE: f64 = 64.0 / 256.0;
// _instanseg_padding's default min_dim.
const PADDING_MIN_DIM: usize = 16;
// Feature vector width: dim_coords (2) + n_sigma (2).
const FEATURE_WIDTH: usize = 4;

/// Postprocessing parameters mirroring upstream's own tunable surface.
/// Defaults are the serialized `single_channel_nuclei` model's own saved
/// values (which override the wheel source's `params.get` fallbacks; see
/// the module-level faithfulness notes).
#[derive(Clone, Copy, Debug)]
pub struct InstanSegConfig {
    /// `mask_threshold` (model default 0.53, wheel fallback 0.53).
    pub mask_threshold: f32,
    /// `peak_distance`, the peak-local-maxima neighbourhood size (5).
    pub peak_distance: usize,
    /// `seed_threshold` (model default 0.1; wheel fallback 0.7).
    pub seed_threshold: f32,
    /// `overlap_threshold` for the IoMA overlap merge (model default 0.5;
    /// wheel fallback 0.3).
    pub overlap_threshold: f32,
    /// `window_size` (upstream default 32); centroid gather windows are
    /// `2 * min(window_size, height, width)` px.
    pub window_size: usize,
    /// `min_size` (serialized model default 20): objects whose RAW
    /// classifier output has at most `min_size` pixels at or above
    /// `mask_threshold` are discarded before cleanup (serialized-graph
    /// behaviour; the wheel source never applies `min_size` in
    /// `forward`). The comparison is strict (`count > min_size`).
    pub min_size: usize,
    /// `cleanup_fragments` (upstream default true).
    pub cleanup_fragments: bool,
}

impl Default for InstanSegConfig {
    fn default() -> Self {
        Self {
            mask_threshold: 0.53,
            peak_distance: 5,
            seed_threshold: 0.1,
            overlap_threshold: 0.5,
            window_size: 32,
            min_size: 20,
            cleanup_fragments: true,
        }
    }
}

/// Segmentation labels for one image, row-major, `0` = background.
/// Unlike StarDist decoding (this crate's `segmentation::stardist`),
/// labels are NOT relabelled consecutive: upstream's overlap merge
/// renumbers a merged object onto the smallest label of its connected
/// component (`find_connected_components` upstream), so surviving
/// labels can be non-contiguous.
#[derive(Debug)]
pub struct InstanSegResult {
    pub labels: Vec<u32>,
    pub object_count: u32,
    pub rows: usize,
    pub cols: usize,
}

/// The fcn network's raw float32 channel stack in NCHW order:
/// `data[ch * rows * cols + r * cols + c]`.
#[derive(Debug)]
pub struct FcnOutput {
    pub channels: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// The two exported ONNX graphs of the pinned `single_channel_nuclei`
/// model (see `src/cp_anywhere/data/instanseg/NOTICE.md`): the traced
/// fcn CNN with dynamic `(1, 1, H, W)` input, and the small linear
/// pixel classifier with `(N, 4)` feature input.
pub struct InstanSegOnnxModel {
    fcn_bytes: Vec<u8>,
    classifier_bytes: Vec<u8>,
}

impl InstanSegOnnxModel {
    pub fn from_bytes(fcn_bytes: Vec<u8>, classifier_bytes: Vec<u8>) -> Self {
        Self {
            fcn_bytes,
            classifier_bytes,
        }
    }

    fn build_plan(
        &self,
        bytes: &[u8],
        input: &Tensor,
        what: &str,
    ) -> TractResult<Arc<TypedSimplePlan>> {
        let mut reader = Cursor::new(bytes);
        let model = tract_onnx::onnx().model_for_read(&mut reader)?;
        ensure!(
            model.input_outlets()?.len() == 1,
            "expected one {what} model input"
        );
        let fact = model.input_fact(0)?;
        ensure!(
            fact.datum_type().is_none_or(|dt| dt == f32::datum_type()),
            "expected a float32 {what} model input"
        );
        // The exported input facts may pin a static shape (the pixel
        // classifier was exported as (3, 4)); the graphs themselves are
        // batch-agnostic (Gemm/Relu/conv chains), so the actual caller
        // shape is bound instead of matching the declared one.
        model
            .with_input_fact(0, InferenceFact::dt_shape_from_tensor(input))?
            .into_optimized()?
            .into_runnable()
    }

    /// Run the fcn network on one caller-normalized, already-padded
    /// row-major image (`rows * cols` -> float32 NCHW `(1, 1, rows, cols)`).
    pub fn run_fcn(&self, image: &[f32], rows: usize, cols: usize) -> Result<FcnOutput, String> {
        let len = rows.checked_mul(cols).ok_or("image shape overflows")?;
        if image.len() != len {
            return Err("image length must equal rows * cols".into());
        }
        if !image.iter().all(|&v| v.is_finite()) {
            return Err("image values must be finite".into());
        }
        let input = Tensor::from_shape(&[1, 1, rows, cols], image)
            .map_err(|e| format!("failed to shape fcn input tensor: {e:#}"))?;
        let plan = self
            .build_plan(&self.fcn_bytes, &input, "fcn")
            .map_err(|e| format!("failed to build InstanSeg fcn ONNX plan: {e:#}"))?;
        let result = plan
            .run(tvec!(input.into()))
            .map_err(|e| format!("InstanSeg fcn ONNX inference failed: {e:#}"))?;
        let [out] = &result[..] else {
            return Err(format!("expected 1 fcn output, got {}", result.len()));
        };
        let shape = out.shape();
        let &[batch, channels, out_rows, out_cols] = shape else {
            return Err(format!(
                "expected a 4-D fcn output (N, C, H, W), got {shape:?}"
            ));
        };
        if batch != 1 {
            return Err(format!("expected batch size 1, got {batch}"));
        }
        let view = out
            .to_plain_array_view::<f32>()
            .map_err(|e| format!("failed to read fcn output as float32: {e:#}"))?;
        if !view.iter().copied().all(f32::is_finite) {
            return Err("fcn output has non-finite values".into());
        }
        let count = channels
            .checked_mul(out_rows)
            .and_then(|v| v.checked_mul(out_cols))
            .ok_or("fcn output shape overflows")?;
        let mut data = Vec::new();
        data.try_reserve_exact(count)
            .map_err(|_| "fcn output is too large")?;
        for ch in 0..channels {
            for r in 0..out_rows {
                for c in 0..out_cols {
                    data.push(view[[0, ch, r, c]]);
                }
            }
        }
        Ok(FcnOutput {
            channels,
            rows: out_rows,
            cols: out_cols,
            data,
        })
    }

    /// Run the pixel classifier on a flattened `(rows, feature_width)`
    /// float32 batch (row-major), returning the raw (pre-sigmoid)
    /// logits, one per row.
    pub fn run_pixel_classifier(
        &self,
        features: &[f32],
        rows: usize,
        feature_width: usize,
    ) -> Result<Vec<f32>, String> {
        let len = rows
            .checked_mul(feature_width)
            .ok_or("feature shape overflows")?;
        if features.len() != len {
            return Err("features length must equal rows * feature_width".into());
        }
        let input = Tensor::from_shape(&[rows, feature_width], features)
            .map_err(|e| format!("failed to shape classifier input tensor: {e:#}"))?;
        let plan = self
            .build_plan(&self.classifier_bytes, &input, "pixel classifier")
            .map_err(|e| format!("failed to build InstanSeg classifier ONNX plan: {e:#}"))?;
        let result = plan
            .run(tvec!(input.into()))
            .map_err(|e| format!("InstanSeg classifier ONNX inference failed: {e:#}"))?;
        let [out] = &result[..] else {
            return Err(format!(
                "expected 1 classifier output, got {}",
                result.len()
            ));
        };
        let shape = out.shape();
        let &[out_rows, out_cols] = shape else {
            return Err(format!(
                "expected a 2-D classifier output (N, 1), got {shape:?}"
            ));
        };
        if out_rows != rows || out_cols != 1 {
            return Err(format!(
                "classifier output shape {out_rows}x{out_cols} does not match {rows}x1"
            ));
        }
        let view = out
            .to_plain_array_view::<f32>()
            .map_err(|e| format!("failed to read classifier output as float32: {e:#}"))?;
        if !view.iter().copied().all(f32::is_finite) {
            return Err("classifier output has non-finite values".into());
        }
        Ok(view.iter().copied().collect())
    }

    /// Full single-image path: clamp to [-2, 3], reflect-pad bottom and
    /// right to a 16 px multiple (`_instanseg_padding`, `extra_pad = 0`,
    /// `min_dim = 16`), run the fcn, recover the padding, then run the
    /// postprocessing port. `image` is the caller-normalized float32
    /// field (`rows * cols`), row-major.
    pub fn segment(
        &self,
        image: &[f32],
        rows: usize,
        cols: usize,
        config: &InstanSegConfig,
    ) -> Result<InstanSegResult, String> {
        if rows == 0 || cols == 0 {
            return Err("image dimensions must be positive".into());
        }
        // Upstream clamps before padding; clamping commutes with the
        // reflect gather, so both orders agree.
        let (padded_rows, padded_cols) = instanseg_padding_dims(rows, cols)?;
        let mut padded = vec![0.0f32; padded_rows * padded_cols];
        for r in 0..padded_rows {
            let source_row = reflect_index(r, rows);
            for c in 0..padded_cols {
                padded[r * padded_cols + c] =
                    image[source_row * cols + reflect_index(c, cols)].clamp(-2.0, 3.0);
            }
        }
        let fcn = self.run_fcn(&padded, padded_rows, padded_cols)?;
        if fcn.rows < rows || fcn.cols < cols {
            return Err(format!(
                "fcn output {}x{} is smaller than the padded input {}x{}",
                fcn.rows, fcn.cols, padded_rows, padded_cols
            ));
        }
        self.postprocess(&fcn, rows, cols, config)
    }

    /// `_recover_padding` + the full downstream postprocessing port on a
    /// raw fcn output. `rows`/`cols` are the padded-back (original)
    /// image dimensions; the fcn output must cover them.
    pub fn postprocess(
        &self,
        fcn: &FcnOutput,
        rows: usize,
        cols: usize,
        config: &InstanSegConfig,
    ) -> Result<InstanSegResult, String> {
        const MIN_CHANNELS: usize = 5; // 2 coords + 2 sigmas + 1 seed
        if fcn.channels < MIN_CHANNELS {
            return Err(format!(
                "expected at least {MIN_CHANNELS} fcn channels (2 coords + 2 sigmas + \
                 1 seed), got {}",
                fcn.channels
            ));
        }
        let plane = rows.checked_mul(cols).ok_or("image shape overflows")?;
        // `_recover_padding`: the fcn output is row-major over the PADDED
        // grid (fcn.rows x fcn.cols); crop each channel to the original
        // (rows, cols) top-left plane before everything downstream.
        if fcn.rows < rows || fcn.cols < cols {
            return Err(format!(
                "fcn output {}x{} is smaller than the padded-back image {}x{}",
                fcn.rows, fcn.cols, rows, cols
            ));
        }
        let cropped_channel = |ch: usize| -> Vec<f32> {
            let stride = fcn.cols;
            let base = ch * fcn.rows * stride;
            let mut out = vec![0.0f32; plane];
            for r in 0..rows {
                let start = base + r * stride;
                out[r * cols..(r + 1) * cols].copy_from_slice(&fcn.data[start..start + cols]);
            }
            out
        };
        let coords_x = cropped_channel(0);
        let coords_y = cropped_channel(1);
        let sigma_x = cropped_channel(2);
        let sigma_y = cropped_channel(3);
        let seed_raw = cropped_channel(fcn.channels - 1);

        // (torch.sigmoid(x[0:2]) - 0.5) * 8, then fields = fields + xxyy.
        let (xx, yy) = coordinate_maps(rows, cols);
        let mut fields_x = vec![0.0f32; plane];
        let mut fields_y = vec![0.0f32; plane];
        for (index, value) in coords_x.iter().enumerate() {
            let r = index / cols;
            let c = index % cols;
            fields_x[index] = (sigmoid(*value) - 0.5) * 8.0 + xx[c];
            fields_y[index] = (sigmoid(coords_y[index]) - 0.5) * 8.0 + yy[r];
        }

        // mask_map = x[-1] / 15 + 0.5 (the dim_seeds == 1 branch).
        let mask_map: Vec<f32> = seed_raw.iter().map(|&v| v / 15.0 + 0.5).collect();

        let mut centroids = peak_local_max(
            &mask_map,
            rows,
            cols,
            config.peak_distance,
            config.seed_threshold,
        );
        let mut centroid_count = centroids.len();
        if centroid_count == 0 {
            return Ok(InstanSegResult {
                labels: vec![0; plane],
                object_count: 0,
                rows,
                cols,
            });
        }

        let window = config.window_size.min(rows).min(cols);
        let slice_size = window * 2;
        // Upstream crashes for images smaller than the gather window; fail
        // explicitly instead of wrapping indices that sparse_coo_tensor
        // would itself reject.
        if slice_size > rows || slice_size > cols {
            return Err(format!(
                "image {rows}x{cols} is smaller than InstanSeg's {slice_size} px centroid \
                 gather window (upstream itself cannot segment an image this small with \
                 window_size {})",
                config.window_size
            ));
        }

        // Clamp centroids into the image with torch's clamp_ semantics:
        // `clamp_(min=a, max=b)` applies max first, so `a > b` collapses
        // every value onto `b`.
        let clamp_centroid = |value: usize, max: usize| -> usize {
            let low = value.max(window);
            low.min(max)
        };
        let mut top_left_row: Vec<usize> = centroids
            .iter()
            .map(|&(r, _)| clamp_centroid(r, rows - window) - window)
            .collect();
        let mut top_left_col: Vec<usize> = centroids
            .iter()
            .map(|&(_, c)| clamp_centroid(c, cols - window) - window)
            .collect();

        // fields_at_centroids = fields[:, y, x]; c = fields_at_centroids.T.
        let centroid_x: Vec<f32> = centroids
            .iter()
            .map(|&(r, c)| fields_x[r * cols + c])
            .collect();
        let centroid_y: Vec<f32> = centroids
            .iter()
            .map(|&(r, c)| fields_y[r * cols + c])
            .collect();

        // Window gather: for centroid ci and in-window offset (a, b)
        // (row-major), the pixel is (top_left_row[ci] + a,
        // top_left_col[ci] + b). Clamp guarantees non-negative top-left
        // coordinates; torch wraps Python-style negative indices.
        let window_pixel = |top_left_row: &[usize],
                            top_left_col: &[usize],
                            ci: usize,
                            a: usize,
                            b: usize|
         -> Result<usize, String> {
            let r = wrap_index(top_left_row[ci] + a, rows)?;
            let c = wrap_index(top_left_col[ci] + b, cols)?;
            Ok(r * cols + c)
        };

        // feature_engineering_slow: row ci * slice^2 + a * slice + b of
        // the (N, 4) feature batch is
        // [fields_x - c_x, fields_y - c_y, sigma_x, sigma_y] at that
        // centroid's window pixel.
        let area = slice_size
            .checked_mul(slice_size)
            .ok_or("gather window overflows")?;
        let feature_count = centroid_count
            .checked_mul(area)
            .ok_or("feature batch overflows")?;
        let mut features = vec![0.0f32; feature_count * FEATURE_WIDTH];
        for ci in 0..centroid_count {
            for a in 0..slice_size {
                for b in 0..slice_size {
                    let index = ci * area + a * slice_size + b;
                    let pixel = window_pixel(&top_left_row, &top_left_col, ci, a, b)?;
                    features[index * FEATURE_WIDTH] = fields_x[pixel] - centroid_x[ci];
                    features[index * FEATURE_WIDTH + 1] = fields_y[pixel] - centroid_y[ci];
                    features[index * FEATURE_WIDTH + 2] = sigma_x[pixel];
                    features[index * FEATURE_WIDTH + 3] = sigma_y[pixel];
                }
            }
        }

        // x = torch.sigmoid(self.pixel_classifier(x)), reshaped to
        // (C, 1, slice_size, slice_size).
        let logits = self.run_pixel_classifier(&features, feature_count, FEATURE_WIDTH)?;
        let mut prob = vec![0.0f32; feature_count];
        for (value, &logit) in prob.iter_mut().zip(logits.iter()) {
            *value = sigmoid(logit);
        }

        // Serialized-graph `min_size` filter (absent from the wheel
        // source): objects whose RAW classifier output (before cleanup)
        // does not have STRICTLY MORE than `min_size` pixels at or above
        // mask_threshold are dropped, together with their coordinates,
        // windows, and original centroids. Skipped entirely when
        // min_size == 0, matching the graph's `if min_size > 0` gate.
        if config.min_size > 0 {
            let mut valid: Vec<usize> = Vec::new();
            for ci in 0..centroid_count {
                let count = prob[ci * area..(ci + 1) * area]
                    .iter()
                    .filter(|&&p| p >= config.mask_threshold)
                    .count();
                if count > config.min_size {
                    valid.push(ci);
                }
            }
            let kept = valid.len();
            if kept < centroid_count {
                if kept == 0 {
                    return Ok(InstanSegResult {
                        labels: vec![0; plane],
                        object_count: 0,
                        rows,
                        cols,
                    });
                }
                let compact_prob: Vec<f32> = valid
                    .iter()
                    .flat_map(|&ci| prob[ci * area..(ci + 1) * area].to_vec())
                    .collect();
                prob = compact_prob;
                let new_top_left_row: Vec<usize> =
                    valid.iter().map(|&ci| top_left_row[ci]).collect();
                let new_top_left_col: Vec<usize> =
                    valid.iter().map(|&ci| top_left_col[ci]).collect();
                let new_centroids: Vec<(usize, usize)> =
                    valid.iter().map(|&ci| centroids[ci]).collect();
                top_left_row = new_top_left_row;
                top_left_col = new_top_left_col;
                centroids = new_centroids;
                centroid_count = kept;
            }
        }

        // cleanup_fragments: flood-fill each object's own threshold mask
        // from its shifted centroid seed, zero unreached fragments, then
        // fill enclosed holes with exactly mask_threshold.
        if config.cleanup_fragments {
            // shifted_centroid = centroids_idx - top_left, which lands in
            // [0, 2 * window) under every clamp outcome.
            let seed_pixel = |ci: usize| -> (usize, usize) {
                (
                    centroids[ci].0 + window - clamp_centroid(centroids[ci].0, rows - window),
                    centroids[ci].1 + window - clamp_centroid(centroids[ci].1, cols - window),
                )
            };
            let in_mask: Vec<bool> = prob.iter().map(|&p| p >= config.mask_threshold).collect();
            let filled = flood_fill_from_seeds(&in_mask, seed_pixel, centroid_count, slice_size);
            // holes = torch.bitwise_xor(fill_holes(filled), filled): the
            // raw fill_holes result is also true on filled pixels, so the
            // XOR removes them before the hole-fill step below.
            let mut holes = fill_holes(&filled, centroid_count, slice_size);
            for (hole, &is_filled) in holes.iter_mut().zip(filled.iter()) {
                *hole ^= is_filled;
            }
            for ci in 0..centroid_count {
                let base = ci * area;
                for index in 0..area {
                    if !filled[base + index] {
                        prob[base + index] = 0.0;
                    } else if holes[base + index] {
                        prob[base + index] = config.mask_threshold;
                    }
                }
            }
        }

        // convert(): per pixel, the highest-probability candidate at or
        // above mask_threshold wins; ties keep the earliest candidate in
        // the original (centroid-major) scan order, i.e. the lowest
        // centroid id (torch's stable double argsort does the same).
        let mut labels = vec![0u32; plane];
        let mut best_prob = vec![-1.0f32; plane];
        for ci in 0..centroid_count {
            for a in 0..slice_size {
                for b in 0..slice_size {
                    let value = prob[ci * area + a * slice_size + b];
                    if value < config.mask_threshold {
                        continue;
                    }
                    let pixel = window_pixel(&top_left_row, &top_left_col, ci, a, b)?;
                    if value > best_prob[pixel] {
                        best_prob[pixel] = value;
                        labels[pixel] = ci as u32 + 1;
                    }
                }
            }
        }

        // The overlap merge: object i's foreground set is the (necessarily
        // unique) window pixels surviving at or above mask_threshold.
        // The serialized model merges on
        // fast_sparse_intersection_over_minimum_area (IoMA): inter /
        // min(area_i, area_j), with NaN (0/0) comparing false exactly
        // like torch.
        let mut foreground: Vec<Vec<u32>> = vec![Vec::new(); centroid_count];
        for ci in 0..centroid_count {
            for a in 0..slice_size {
                for b in 0..slice_size {
                    if prob[ci * area + a * slice_size + b] >= config.mask_threshold {
                        foreground[ci]
                            .push(window_pixel(&top_left_row, &top_left_col, ci, a, b)? as u32);
                    }
                }
            }
        }
        for pixels in foreground.iter_mut() {
            pixels.sort_unstable();
            pixels.dedup();
        }
        let areas: Vec<usize> = foreground.iter().map(Vec::len).collect();
        let mut adjacency = vec![false; centroid_count * centroid_count];
        for i in 0..centroid_count {
            for j in i..centroid_count {
                let intersection = if i == j {
                    areas[i]
                } else {
                    count_intersection(&foreground[i], &foreground[j])
                };
                // IoMA: intersection over the minimum area of the pair.
                let minimum = areas[i].min(areas[j]);
                let ioma = if minimum == 0 {
                    f32::NAN
                } else {
                    intersection as f32 / minimum as f32
                };
                let connected = ioma > config.overlap_threshold; // NaN > x is false
                adjacency[i * centroid_count + j] = connected;
                adjacency[j * centroid_count + i] = connected;
            }
        }
        // find_connected_components: min-label propagation iterated to
        // convergence (upstream's cap is 100 iterations; it proceeds
        // with partially converged labels and only warns, so does this
        // port).
        let mut component: Vec<u32> = (1..=centroid_count as u32).collect();
        for iteration in 0..100 {
            let previous = component.clone();
            for i in 0..centroid_count {
                let mut min_label = component[i];
                for j in 0..centroid_count {
                    if adjacency[i * centroid_count + j] {
                        min_label = min_label.min(component[j]);
                    }
                }
                component[i] = min_label;
            }
            if component == previous {
                break;
            }
            if iteration == 99 {
                // Upstream warns and continues; the port continues too.
            }
        }

        // remap_values: label v (1-based) maps onto the component minimum
        // of object v; background 0 maps to 0.
        for label in labels.iter_mut() {
            if *label != 0 {
                *label = component[*label as usize - 1];
            }
        }

        let distinct: std::collections::BTreeSet<u32> =
            labels.iter().copied().filter(|&v| v != 0).collect();
        Ok(InstanSegResult {
            labels,
            object_count: distinct.len() as u32,
            rows,
            cols,
        })
    }
}

/// `_instanseg_padding(img, extra_pad = 0, min_dim = 16)`'s bottom/right
/// padding sizes: each dimension is padded up to the next `min_dim`
/// multiple. Upstream's odd `if padx > img.shape[-2] { padx -=
/// extra_pad }` branch is inert at `extra_pad = 0` and is intentionally
/// not carried as dead code.
fn instanseg_padding_dims(rows: usize, cols: usize) -> Result<(usize, usize), String> {
    let padded = |dim: usize| -> Result<usize, String> {
        dim.checked_add((PADDING_MIN_DIM - dim % PADDING_MIN_DIM) % PADDING_MIN_DIM)
            .ok_or_else(|| "padded image dimension overflows".into())
    };
    let padded_rows = padded(rows)?;
    let padded_cols = padded(cols)?;
    // F.pad(mode="reflect") requires the pad to be strictly smaller than
    // the dimension being padded; matching torch's own failure mode.
    if padded_rows - rows >= rows || padded_cols - cols >= cols {
        return Err(
            "image is too small for InstanSeg's 16 px reflect padding (torch's own \
             F.pad(mode='reflect') fails in this case)"
                .into(),
        );
    }
    Ok((padded_rows, padded_cols))
}

/// `F.pad(..., mode="reflect")` boundary index mapping: reflects without
/// repeating the edge value (period `2 * (n - 1)`). Size-1 dimensions
/// are rejected before this runs.
fn reflect_index(idx: usize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    if idx < len {
        return idx;
    }
    let edge = len - 1;
    let pos = idx % (2 * edge);
    pos.min(2 * edge - pos)
}

/// Python-style negative index wrap, matching torch advanced indexing.
fn wrap_index(idx: usize, len: usize) -> Result<usize, String> {
    if idx >= len {
        return Err(format!(
            "coordinate index {idx} is out of range for dimension of length {len}"
        ));
    }
    Ok(idx)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `generate_coordinate_map(mode="linear", spatial_dim=2)`:
/// `xx = linspace(0, w * 64/256, w)` broadcast across rows and
/// `yy = linspace(0, h * 64/256, h)` broadcast across columns.
/// Computed in f64 with per-element f32 rounding, matching torch's
/// higher-precision linspace interpolation.
fn coordinate_maps(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let linspace = |n: usize, end: f64| -> Vec<f32> {
        if n == 1 {
            return vec![0.0];
        }
        let step = end / (n as f64 - 1.0);
        (0..n).map(|i| (step * i as f64) as f32).collect()
    };
    (
        linspace(cols, cols as f64 * COORD_SCALE),
        linspace(rows, rows as f64 * COORD_SCALE),
    )
}

/// `torch_peak_local_max` on a single-channel `rows * cols` map: a peak
/// is a pixel whose own global index equals the argmax of its
/// `(2r+1) x (2r+1)` window with a value strictly above `min_value`.
/// Ties keep the FIRST row-major window position attaining the max,
/// replicating max-pool-with-indices' documented first-max behaviour.
/// Returns `(row, col)` pairs in row-major order (`nonzero`'s order).
fn peak_local_max(
    image: &[f32],
    rows: usize,
    cols: usize,
    radius: usize,
    min_value: f32,
) -> Vec<(usize, usize)> {
    let mut peaks = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let row_start = r.saturating_sub(radius);
            let row_end = (r + radius).min(rows - 1);
            let col_start = c.saturating_sub(radius);
            let col_end = (c + radius).min(cols - 1);
            let mut best_value = f32::NEG_INFINITY;
            let mut best_index = usize::MAX;
            for wr in row_start..=row_end {
                for wc in col_start..=col_end {
                    let value = image[wr * cols + wc];
                    if value > best_value {
                        best_value = value;
                        best_index = wr * cols + wc;
                    }
                }
            }
            if best_index == r * cols + c && best_value > min_value {
                peaks.push((r, c));
            }
        }
    }
    peaks
}

/// flood_fill's per-object 8-connected dilation loop (3x3 max-pool with
/// the object's threshold mask as the flood barrier, converged or capped
/// at max(seed height, seed width) iterations).
fn flood_fill_from_seeds(
    mask: &[bool],
    seed_pixel: impl Fn(usize) -> (usize, usize),
    object_count: usize,
    size: usize,
) -> Vec<bool> {
    let mut result = vec![false; object_count * size * size];
    for ci in 0..object_count {
        let base = ci * size * size;
        let mut current = vec![false; size * size];
        let (seed_row, seed_col) = seed_pixel(ci);
        current[seed_row * size + seed_col] = true;
        let max_iterations = size; // max(bw_seed.shape[-1], bw_seed.shape[-2])
        for _ in 0..max_iterations {
            let mut next = current.clone();
            for r in 0..size {
                for c in 0..size {
                    let index = r * size + c;
                    if next[index] {
                        continue;
                    }
                    let mut dilated = false;
                    for dr in r.saturating_sub(1)..=(r + 1).min(size - 1) {
                        for dc in c.saturating_sub(1)..=(c + 1).min(size - 1) {
                            if current[dr * size + dc] {
                                dilated = true;
                            }
                        }
                    }
                    if dilated && mask[base + index] {
                        next[index] = true;
                    }
                }
            }
            if next == current {
                break;
            }
            current = next;
        }
        result[base..base + size * size].copy_from_slice(&current);
    }
    result
}

/// fill_holes: dilate the filled region by 3 iterations, flood-fill the
/// complement from every non-dilated pixel simultaneously (the seeded
/// max-pool loop is additive over seed pixels), and invert the result.
fn fill_holes(filled: &[bool], object_count: usize, size: usize) -> Vec<bool> {
    let mut result = vec![false; object_count * size * size];
    for ci in 0..object_count {
        let base = ci * size * size;
        let region = &filled[base..base + size * size];
        let mut dilated = region.to_vec();
        for _ in 0..3 {
            let mut next = dilated.clone();
            for r in 0..size {
                for c in 0..size {
                    let index = r * size + c;
                    if next[index] {
                        continue;
                    }
                    for dr in r.saturating_sub(1)..=(r + 1).min(size - 1) {
                        for dc in c.saturating_sub(1)..=(c + 1).min(size - 1) {
                            if dilated[dr * size + dc] {
                                next[index] = true;
                            }
                        }
                    }
                }
            }
            dilated = next;
        }
        let mut reached = vec![false; size * size];
        for (index, &seed) in dilated.iter().enumerate() {
            reached[index] = !seed;
        }
        for _ in 0..size {
            let mut next = reached.clone();
            for r in 0..size {
                for c in 0..size {
                    let index = r * size + c;
                    if next[index] {
                        continue;
                    }
                    let mut dilated = false;
                    for dr in r.saturating_sub(1)..=(r + 1).min(size - 1) {
                        for dc in c.saturating_sub(1)..=(c + 1).min(size - 1) {
                            if reached[dr * size + dc] {
                                dilated = true;
                            }
                        }
                    }
                    if dilated && !region[index] {
                        next[index] = true;
                    }
                }
            }
            if next == reached {
                break;
            }
            reached = next;
        }
        for (index, &reached) in reached.iter().enumerate() {
            result[base + index] = !reached;
        }
    }
    result
}

/// Intersection size of two sorted, unique pixel lists.
fn count_intersection(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut count) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

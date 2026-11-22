//! In-memory StarDist 2D ONNX inference via `tract`.
//! This module requires the `stardist-model` feature. Decoding does not.

use std::io::Cursor;
use std::sync::Arc;

use tract_onnx::{prelude::*, tract_core::internal::ensure};

const FLUO_INPUT_DIVISOR: (usize, usize) = (16, 16);

/// A single-channel StarDist 2D ONNX model with NHWC input and two NHWC heads.
/// Each prediction builds a `tract` plan for the concrete padded input shape.
pub struct StarDistOnnxModel {
    bytes: Vec<u8>,
    n_rays: usize,
    grid: (usize, usize),
    input_divisor: (usize, usize),
}

/// Network heads cropped to the original image's output grid, flattened in
/// row-major order: `prob` has one value per cell, `dist` has `n_rays`.
#[derive(Debug)]
pub struct StarDistRawOutput {
    pub prob: Vec<f32>,
    pub dist: Vec<f32>,
    pub out_rows: usize,
    pub out_cols: usize,
    pub n_rays: usize,
}

impl StarDistOnnxModel {
    /// Use the packaged `2D_versatile_fluo` input divisor (16 on both axes).
    pub fn from_bytes(model_bytes: Vec<u8>, n_rays: usize, grid: (usize, usize)) -> Self {
        Self::from_bytes_with_input_divisor(model_bytes, n_rays, grid, FLUO_INPUT_DIVISOR)
    }

    /// Set the model's input divisor explicitly for a different StarDist export.
    /// Both divisors must be positive multiples of the corresponding grid axes.
    /// `predict` validates this configuration and the model's output shapes.
    pub fn from_bytes_with_input_divisor(
        model_bytes: Vec<u8>,
        n_rays: usize,
        grid: (usize, usize),
        input_divisor: (usize, usize),
    ) -> Self {
        Self {
            bytes: model_bytes,
            n_rays,
            grid,
            input_divisor,
        }
    }

    fn build_plan(&self, input: &Tensor) -> TractResult<Arc<TypedSimplePlan>> {
        let mut reader = Cursor::new(&self.bytes);
        let model = tract_onnx::onnx().model_for_read(&mut reader)?;
        ensure!(
            model.input_outlets()?.len() == 1,
            "expected one StarDist model input"
        );
        let fact = model.input_fact(0)?;
        ensure!(
            fact.datum_type().is_none_or(|dt| dt == f32::datum_type())
                && fact.shape.matches(input, None)?,
            "expected a compatible float32 NHWC StarDist model input"
        );
        model
            .with_input_fact(0, InferenceFact::dt_shape_from_tensor(input))?
            .into_optimized()?
            .into_runnable()
    }

    /// Run the model on one caller-normalized row-major image (`rows * cols`).
    /// Reflect-pad the bottom and right to the model's input divisor, then crop
    /// both heads to `(ceil(rows / grid.0), ceil(cols / grid.1))`.
    pub fn predict(
        &self,
        image: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<StarDistRawOutput, String> {
        if rows == 0 || cols == 0 {
            return Err("image dimensions must be positive".into());
        }
        let len = rows.checked_mul(cols).ok_or("image shape overflows")?;
        if image.len() != len {
            return Err("image length must equal rows * cols".into());
        }
        if !image.iter().all(|x| x.is_finite()) {
            return Err("image values must be finite".into());
        }
        if self.n_rays < 3 {
            return Err("n_rays must be at least 3".into());
        }
        let (grid_rows, grid_cols) = self.grid;
        let (div_rows, div_cols) = self.input_divisor;
        if [grid_rows, grid_cols, div_rows, div_cols].contains(&0)
            || div_rows % grid_rows != 0
            || div_cols % grid_cols != 0
        {
            return Err(
                "input divisors must be positive multiples of positive grid dimensions".into(),
            );
        }
        let padded_rows = padded_dim(rows, div_rows)?;
        let padded_cols = padded_dim(cols, div_cols)?;
        let padded_len = padded_rows
            .checked_mul(padded_cols)
            .ok_or("padded image shape overflows")?;
        if padded_len > isize::MAX as usize / std::mem::size_of::<f32>() {
            return Err("padded image is too large".into());
        }
        let head_shape = (padded_rows / grid_rows, padded_cols / grid_cols);
        head_shape
            .0
            .checked_mul(head_shape.1)
            .and_then(|v| v.checked_mul(self.n_rays))
            .filter(|&v| v <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or("distance head shape is too large")?;

        let mut padded = Vec::new();
        let samples = if (rows, cols) == (padded_rows, padded_cols) {
            image
        } else {
            padded
                .try_reserve_exact(padded_len)
                .map_err(|_| "padded image is too large")?;
            for r in 0..padded_rows {
                let offset = reflected_index(r, rows) * cols;
                padded.extend_from_slice(&image[offset..offset + cols]);
                for c in cols..padded_cols {
                    padded.push(image[offset + reflected_index(c, cols)]);
                }
            }
            &padded
        };
        let input = Tensor::from_shape(&[1, padded_rows, padded_cols, 1], samples)
            .map_err(|e| format!("failed to shape input tensor: {e:#}"))?;
        drop(padded);
        let plan = self
            .build_plan(&input)
            .map_err(|e| format!("failed to build StarDist ONNX plan: {e:#}"))?;
        let result = plan
            .run(tvec!(input.into()))
            .map_err(|e| format!("StarDist ONNX inference failed: {e:#}"))?;
        unpack_heads(
            &result,
            head_shape,
            (rows.div_ceil(grid_rows), cols.div_ceil(grid_cols)),
            self.n_rays,
        )
    }

    pub fn n_rays(&self) -> usize {
        self.n_rays
    }

    pub fn grid(&self) -> (usize, usize) {
        self.grid
    }
}

fn padded_dim(dim: usize, divisor: usize) -> Result<usize, String> {
    dim.checked_add((divisor - dim % divisor) % divisor)
        .ok_or_else(|| "padded image dimension overflows".into())
}

fn reflected_index(idx: usize, len: usize) -> usize {
    if idx < len || len == 1 {
        return idx.min(len - 1);
    }
    // The edge is not repeated; tiny axes may need several reflections.
    let edge = len - 1;
    let pos = idx % (2 * edge);
    pos.min(2 * edge - pos)
}

fn unpack_heads(
    outputs: &[TValue],
    (rows, cols): (usize, usize),
    (out_rows, out_cols): (usize, usize),
    n_rays: usize,
) -> Result<StarDistRawOutput, String> {
    let [a, b] = outputs else {
        return Err(format!(
            "expected 2 StarDist model outputs, got {}",
            outputs.len()
        ));
    };
    let prob_shape = [1, rows, cols, 1];
    let dist_shape = [1, rows, cols, n_rays];
    // Identify the heads by their entire shape, not just channel index or order.
    let (prob, dist) = if a.shape() == prob_shape && b.shape() == dist_shape {
        (a, b)
    } else if b.shape() == prob_shape && a.shape() == dist_shape {
        (b, a)
    } else {
        return Err(format!(
            "expected NHWC output shapes {prob_shape:?} and {dist_shape:?}, got {:?} and {:?}",
            a.shape(),
            b.shape()
        ));
    };
    let prob = read_head(prob, "prob", out_rows, out_cols, 1, |p| {
        (0.0..=1.0).contains(&p)
    })?;
    let dist = read_head(dist, "dist", out_rows, out_cols, n_rays, f32::is_finite)?;
    Ok(StarDistRawOutput {
        prob,
        dist,
        out_rows,
        out_cols,
        n_rays,
    })
}

fn read_head(
    tensor: &Tensor,
    name: &str,
    rows: usize,
    cols: usize,
    channels: usize,
    is_valid: impl Fn(f32) -> bool,
) -> Result<Vec<f32>, String> {
    let view = tensor
        .to_plain_array_view::<f32>()
        .map_err(|e| format!("failed to read {name} output as float32: {e:#}"))?;
    if !view.iter().copied().all(is_valid) {
        return Err(format!("{name} output has invalid values"));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(rows * cols * channels)
        .map_err(|_| format!("{name} output is too large"))?;
    for r in 0..rows {
        for c in 0..cols {
            for k in 0..channels {
                result.push(view[[0, r, c, k]]);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn model_bytes() -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/cp_anywhere/data/stardist/2D_versatile_fluo.onnx");
        std::fs::read(path).expect(
            "src/cp_anywhere/data/stardist/2D_versatile_fluo.onnx must be present \
             (run scripts/stardist/convert.py or restore it from the repo)",
        )
    }

    #[test]
    fn predicts_two_heads_with_the_expected_grid_downsample() {
        let model = StarDistOnnxModel::from_bytes(model_bytes(), 32, (2, 2));
        let image = vec![0.3f32; 64 * 64];
        let out = model.predict(&image, 64, 64).unwrap();
        assert_eq!(out.out_rows, 32);
        assert_eq!(out.out_cols, 32);
        assert_eq!(out.n_rays, 32);
        assert_eq!(out.prob.len(), 32 * 32);
        assert_eq!(out.dist.len(), 32 * 32 * 32);
        assert!(out.prob.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn predicts_a_rectangular_odd_image_like_explicit_reflection_and_crop() {
        let model = StarDistOnnxModel::from_bytes(model_bytes(), 32, (2, 2));
        let (rows, cols) = (31, 17);
        let image: Vec<f32> = (0..rows * cols)
            .map(|i| (i * 37 % 101) as f32 / 100.0)
            .collect();
        // Pad bottom by one row and right by 15 columns, excluding the edge pixel.
        let padded: Vec<f32> = (0..32)
            .flat_map(|r| {
                let image = &image;
                (0..32).map(move |c| {
                    image[(if r == 31 { 29 } else { r }) * cols + if c >= 17 { 32 - c } else { c }]
                })
            })
            .collect();
        let aligned = model.predict(&padded, 32, 32).unwrap();
        let odd = model.predict(&image, rows, cols).unwrap();
        assert_eq!((odd.out_rows, odd.out_cols, odd.n_rays), (16, 9, 32));
        assert_eq!(odd.prob.len(), 16 * 9);
        assert_eq!(odd.dist.len(), 16 * 9 * 32);
        for r in 0..16 {
            assert_eq!(
                odd.prob[r * 9..(r + 1) * 9],
                aligned.prob[r * 16..r * 16 + 9]
            );
            assert_eq!(
                odd.dist[r * 9 * 32..(r + 1) * 9 * 32],
                aligned.dist[r * 16 * 32..(r * 16 + 9) * 32]
            );
        }
    }

    #[test]
    fn rejects_invalid_image_buffers_before_loading_onnx() {
        let model = StarDistOnnxModel::from_bytes(vec![], 32, (2, 2));
        for (image, rows, cols) in [
            (&[][..], usize::MAX, 2),
            (&[], 0, 10),
            (&[f32::NAN], 1, 1),
            (&[f32::INFINITY], 1, 1),
        ] {
            assert!(model
                .predict(image, rows, cols)
                .err()
                .unwrap()
                .starts_with("image"));
        }
    }

    #[test]
    fn output_heads_require_nhwc_rank_and_finite_float32_samples() {
        let pack = |shape: &[usize], values: &[f32]| -> TValue {
            Tensor::from_shape(shape, values).unwrap().into()
        };
        let prob = || pack(&[1, 1, 2, 1], &[0.8; 2]);
        let dist = || pack(&[1, 1, 2, 4], &[2.0; 8]);
        let unpack = |heads: &[TValue]| unpack_heads(heads, (1, 2), (1, 1), 4);
        assert_eq!(unpack(&[dist(), prob()]).unwrap().prob, [0.8]);
        assert!(unpack(&[dist()]).is_err());
        assert!(unpack(&[prob(), pack(&[1, 2, 4], &[2.0; 8])]).is_err());
        assert!(unpack(&[prob(), pack(&[1, 1, 2, 5], &[2.0; 10])]).is_err());
        assert!(unpack(&[prob(), pack(&[1, 1, 2, 4], &[f32::NAN; 8])]).is_err());
        assert!(unpack(&[pack(&[1, 1, 2, 1], &[0.8, f32::INFINITY]), dist()]).is_err());
        let integer_prob = Tensor::from_shape(&[1, 1, 2, 1], &[1_i32; 2]).unwrap();
        assert!(unpack(&[integer_prob.into(), dist()]).is_err());
    }

    #[test]
    fn validates_model_config_before_inference() {
        for (n_rays, grid, div, error) in [
            (2, (2, 2), (16, 16), "n_rays"),
            (4, (0, 2), (16, 16), "input divisors"),
            (4, (3, 2), (16, 16), "input divisors"),
            (4, (2, 2), (0, 16), "input divisors"),
            (4, (1, 1), (usize::MAX, 16), "padded image"),
            (usize::MAX, (2, 2), (16, 16), "distance head"),
        ] {
            let model = StarDistOnnxModel::from_bytes_with_input_divisor(vec![], n_rays, grid, div);
            assert!(model.predict(&[0.5], 1, 1).unwrap_err().starts_with(error));
        }
    }

    #[test]
    fn detects_five_separated_synthetic_nuclei_blobs() {
        use crate::segmentation::stardist::{decode, StarDistDecodeConfig, StarDistShapes};

        let model = StarDistOnnxModel::from_bytes(model_bytes(), 32, (2, 2));
        let size = 128usize;
        let centers = [(24i32, 24i32), (24, 100), (100, 24), (64, 64), (100, 100)];
        let mut image = vec![0.05f32; size * size];
        for row in 0..size {
            for col in 0..size {
                let mut v = image[row * size + col];
                for &(cy, cx) in &centers {
                    let dy = row as f32 - cy as f32;
                    let dx = col as f32 - cx as f32;
                    v += 0.9 * (-(dy * dy + dx * dx) / (2.0 * 8.0 * 8.0)).exp();
                }
                image[row * size + col] = v.clamp(0.0, 1.0);
            }
        }
        let raw = model.predict(&image, size, size).unwrap();
        let shapes = StarDistShapes {
            out_rows: raw.out_rows,
            out_cols: raw.out_cols,
            n_rays: raw.n_rays,
            image_rows: size,
            image_cols: size,
        };
        let config = StarDistDecodeConfig {
            prob_thresh: 0.479_071_46,
            nms_thresh: 0.3,
            grid: model.grid(),
            border: 2,
        };
        let result = decode(&raw.prob, &raw.dist, &shapes, &config).unwrap();
        assert_eq!(result.detections.len(), centers.len());
    }
}

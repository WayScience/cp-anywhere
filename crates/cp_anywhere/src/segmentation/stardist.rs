//! Decode StarDist 2D probability and ray-distance heads into a label image.
//!
//! Based on StarDist's `geom2d.py` and `nms.py` (BSD-3-Clause, copyright
//! 2018–2025 Uwe Schmidt and Martin Weigert). This is **approximate NMS**:
//! it counts shared pixel centers inside the image and divides by the smaller
//! rasterized area. Upstream intersects polygons in geometry space instead.
//! Small or border-clipped polygons and overlaps near the threshold can give
//! different survivors. See `tests/stardist_decode.rs` for comparison against
//! upstream on one real-network fixture.

use std::f64::consts::TAU;

const MIN_RAY: f32 = 1e-3;
// Allow roundoff at an integer boundary (e.g. from sin/cos and interpolation).
const EDGE_EPSILON: f64 = 1e-9;

type Point = (f64, f64);

#[derive(Clone, Copy)]
struct Span {
    row: usize,
    start: usize,
    end: usize, // exclusive
}

#[derive(Clone, Copy)]
struct Bounds {
    top: usize,
    bottom: usize, // inclusive
    left: usize,
    right: usize, // inclusive
}

impl Bounds {
    fn clipped(vertices: &[Point], rows: usize, cols: usize) -> Self {
        let ((mut top, mut left), (mut bottom, mut right)) = (vertices[0], vertices[0]);
        for &(row, col) in &vertices[1..] {
            top = top.min(row);
            bottom = bottom.max(row);
            left = left.min(col);
            right = right.max(col);
        }
        // Float-to-unsigned conversion saturates for negative and enormous rays.
        Self {
            top: (top - EDGE_EPSILON).ceil() as usize,
            bottom: ((bottom + EDGE_EPSILON).floor() as usize).min(rows - 1),
            left: (left - EDGE_EPSILON).ceil() as usize,
            right: ((right + EDGE_EPSILON).floor() as usize).min(cols - 1),
        }
    }

    fn overlaps(self, other: Self) -> bool {
        self.top <= other.bottom
            && other.top <= self.bottom
            && self.left <= other.right
            && other.left <= self.right
    }
}

#[derive(Default)]
struct Rasterizer {
    vertices: Vec<Point>,
    crossings: Vec<f64>,
    intervals: Vec<(f64, f64)>,
}

impl Rasterizer {
    /// Fill at integer pixel centers, including polygon edges and vertices.
    /// The output spans are sorted and disjoint, even on a concave row.
    fn rasterize(&mut self, bounds: Bounds, spans: &mut Vec<Span>) -> usize {
        spans.clear();
        let mut area = 0;
        for row in bounds.top..=bounds.bottom {
            let y = row as f64;
            self.crossings.clear();
            self.intervals.clear();
            for (&(y0, x0), &(y1, x1)) in self
                .vertices
                .iter()
                .zip(self.vertices.iter().cycle().skip(1))
            {
                // Count each vertex only once for parity. Add the actual
                // boundary separately, including local maxima and flat edges.
                if (y0 > y) != (y1 > y) {
                    let t = (y - y0) / (y1 - y0);
                    let x = if t < 0.5 {
                        x0 + t * (x1 - x0)
                    } else {
                        x1 + ((y - y1) / (y0 - y1)) * (x0 - x1)
                    };
                    self.crossings.push(x);
                }
                if (y0 - y).abs() <= EDGE_EPSILON {
                    if (y1 - y).abs() <= EDGE_EPSILON {
                        self.intervals.push((x0.min(x1), x0.max(x1)));
                    } else {
                        self.intervals.push((x0, x0));
                    }
                }
            }
            self.crossings.sort_unstable_by(f64::total_cmp);
            self.intervals
                .extend(self.crossings.chunks_exact(2).map(|p| (p[0], p[1])));
            self.intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for &(left, right) in &self.intervals {
                let first = (left - EDGE_EPSILON).ceil();
                let last = (right + EDGE_EPSILON).floor();
                if last < 0.0 {
                    continue;
                }
                let start = (first as usize).max(bounds.left);
                let last = (last as usize).min(bounds.right);
                if start > last {
                    continue;
                }
                let end = last + 1;
                if let Some(prev) = spans.last_mut().filter(|s| s.row == row && s.end >= start) {
                    area += end.saturating_sub(prev.end);
                    prev.end = prev.end.max(end);
                } else {
                    area += end - start;
                    spans.push(Span { row, start, end });
                }
            }
        }
        area
    }
}

struct Survivor {
    idx: usize,
    bounds: Bounds,
    spans: Vec<Span>,
    area: usize,
}

/// Count the intersection without expanding spans into individual pixels.
fn shared_pixels(a: &[Span], b: &[Span], stop_above: f64) -> usize {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while let (Some(a), Some(b)) = (a.get(i), b.get(j)) {
        match a.row.cmp(&b.row) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                total += a.end.min(b.end).saturating_sub(a.start.max(b.start));
                if total as f64 > stop_above {
                    break;
                }
                if a.end <= b.end {
                    i += 1;
                }
                if b.end <= a.end {
                    j += 1;
                }
            }
        }
    }
    total
}

/// Settings mirroring `non_maximum_suppression`'s own keyword
/// arguments (`prob_thresh`, `nms_thresh`, `grid`, border exclusion
/// `b`).
#[derive(Clone, Copy, Debug)]
pub struct StarDistDecodeConfig {
    pub prob_thresh: f32,
    pub nms_thresh: f32,
    pub grid: (usize, usize),
    /// Candidates within `border` pixels of the *network output*
    /// array edge are excluded, matching `_ind_prob_thresh`'s default
    /// `b=2`.
    pub border: usize,
}

impl Default for StarDistDecodeConfig {
    fn default() -> Self {
        Self {
            prob_thresh: 0.5,
            nms_thresh: 0.5,
            grid: (1, 1),
            border: 2,
        }
    }
}

/// One surviving, post-NMS detection, in input-image pixel
/// coordinates.
#[derive(Clone, Debug)]
pub struct StarDistDetection {
    pub row: f64,
    pub col: f64,
    pub prob: f32,
}

#[derive(Debug)]
pub struct StarDistDecodeResult {
    pub labels: Vec<u32>,
    pub detections: Vec<StarDistDetection>,
}

/// Static shape parameters for one `decode` call: the network output
/// grid (`out_rows x out_cols`, `n_rays` per pixel) and the
/// corresponding full-resolution image it should be upscaled to.
#[derive(Clone, Copy, Debug)]
pub struct StarDistShapes {
    pub out_rows: usize,
    pub out_cols: usize,
    pub n_rays: usize,
    pub image_rows: usize,
    pub image_cols: usize,
}

/// Decode flattened row-major probability and distance tensors into an image.
/// The output grid shape must be `ceil(image_shape / grid)` per axis.
/// Probabilities and thresholds must be in [0, 1]. Ray distances must be
/// finite; values below 1e-3 are floored before constructing polygons.
pub fn decode(
    prob: &[f32],
    dist: &[f32],
    shapes: &StarDistShapes,
    config: &StarDistDecodeConfig,
) -> Result<StarDistDecodeResult, String> {
    let StarDistShapes {
        out_rows,
        out_cols,
        n_rays,
        image_rows,
        image_cols,
    } = *shapes;
    let (grid_rows, grid_cols) = config.grid;
    if [
        out_rows, out_cols, image_rows, image_cols, grid_rows, grid_cols,
    ]
    .contains(&0)
    {
        return Err("image, output and grid dimensions must be positive".into());
    }
    if n_rays < 3 {
        return Err("n_rays must be at least 3".into());
    }
    if ![config.prob_thresh, config.nms_thresh]
        .iter()
        .all(|p| (0.0..=1.0).contains(p))
    {
        return Err("prob_thresh and nms_thresh must be finite and in [0, 1]".into());
    }
    let prob_len = out_rows
        .checked_mul(out_cols)
        .ok_or("prob shape overflows")?;
    let dist_len = prob_len.checked_mul(n_rays).ok_or("dist shape overflows")?;
    let image_len = image_rows
        .checked_mul(image_cols)
        .ok_or("label image shape overflows")?;
    if out_rows != image_rows.div_ceil(grid_rows) || out_cols != image_cols.div_ceil(grid_cols) {
        return Err("output dimensions must equal ceil(image dimensions / grid)".into());
    }
    if prob.len() != prob_len {
        return Err("prob length must equal out_rows * out_cols".into());
    }
    if dist.len() != dist_len {
        return Err("dist length must equal out_rows * out_cols * n_rays".into());
    }
    if !prob.iter().all(|p| (0.0..=1.0).contains(p)) {
        return Err("prob values must be finite and in [0, 1]".into());
    }
    if !dist.iter().all(|d| d.is_finite()) {
        return Err("dist values must be finite".into());
    }
    let mut labels = Vec::new();
    labels
        .try_reserve_exact(image_len)
        .map_err(|_| "label image is too large")?;
    labels.resize(image_len, 0);

    // Keep only flat indices for unprocessed candidates; rays stay in `dist`.
    let b = config.border;
    let mut candidates = Vec::new();
    for r in b..out_rows.saturating_sub(b) {
        for c in b..out_cols.saturating_sub(b) {
            let idx = r * out_cols + c;
            if prob[idx] > config.prob_thresh {
                candidates.push(idx);
            }
        }
    }
    // Higher probability first; ties in row-major order.
    candidates.sort_unstable_by(|&a, &b| prob[b].total_cmp(&prob[a]).then(a.cmp(&b)));
    let center = |idx: usize| {
        (
            (idx / out_cols * grid_rows) as f64,
            (idx % out_cols * grid_cols) as f64,
        )
    };
    let rays: Vec<_> = (0..n_rays)
        .map(|k| {
            // Exact cardinal directions avoid magnifying trig roundoff for long rays.
            if k == 0 {
                (0.0, 1.0)
            } else if n_rays % 4 == 0 && k == n_rays / 4 {
                (1.0, 0.0)
            } else if n_rays % 2 == 0 && k == n_rays / 2 {
                (0.0, -1.0)
            } else if n_rays % 4 == 0 && k == n_rays / 4 * 3 {
                (-1.0, 0.0)
            } else {
                (k as f64 * TAU / n_rays as f64).sin_cos()
            }
        })
        .collect();

    let mut survivors: Vec<Survivor> = Vec::new();
    let mut rasterizer = Rasterizer::default();
    let mut spans = Vec::new();
    for idx in candidates {
        let (row, col) = center(idx);
        rasterizer.vertices.clear();
        rasterizer.vertices.extend(
            rays.iter()
                .zip(&dist[idx * n_rays..(idx + 1) * n_rays])
                .map(|(&(dy, dx), &d)| {
                    let d = d.max(MIN_RAY) as f64;
                    (row + d * dy, col + d * dx)
                }),
        );
        let bounds = Bounds::clipped(&rasterizer.vertices, image_rows, image_cols);
        let area = rasterizer.rasterize(bounds, &mut spans);
        if area == 0 {
            continue;
        }
        let suppressed = config.nms_thresh < 1.0
            && survivors.iter().any(|s| {
                if !bounds.overlaps(s.bounds) {
                    return false;
                }
                let limit = area.min(s.area) as f64 * f64::from(config.nms_thresh);
                shared_pixels(&spans, &s.spans, limit) as f64 > limit
            });
        if !suppressed {
            // Only keep spans for survivors, and reuse scratch for rejected candidates.
            survivors.push(Survivor {
                idx,
                bounds,
                area,
                spans: std::mem::take(&mut spans),
            });
        }
    }

    // Lowest score paints first so higher scores win any residual overlap.
    // Detection i corresponds to label i + 1, even if fully overwritten.
    survivors.sort_unstable_by(|a, b| prob[a.idx].total_cmp(&prob[b.idx]).then(a.idx.cmp(&b.idx)));
    let mut detections = Vec::new();
    for s in survivors {
        let label = u32::try_from(detections.len() + 1).map_err(|_| "too many StarDist labels")?;
        for span in s.spans {
            let offset = span.row * image_cols;
            labels[offset + span.start..offset + span.end].fill(label);
        }
        let (row, col) = center(s.idx);
        detections.push(StarDistDetection {
            row,
            col,
            prob: prob[s.idx],
        });
    }
    Ok(StarDistDecodeResult { labels, detections })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_shapes(size: usize, n_rays: usize) -> StarDistShapes {
        StarDistShapes {
            out_rows: size,
            out_cols: size,
            n_rays,
            image_rows: size,
            image_cols: size,
        }
    }

    /// A single candidate with all-equal ray distances traces a
    /// regular `n_rays`-gon of circumradius `r`; its rasterized area
    /// should approximate `pi * r^2` to within the polygon's own
    /// inscribed/circumscribed error for that `n_rays` (a hand-
    /// verifiable geometric oracle, independent of any Python/NumPy
    /// comparison).
    #[test]
    fn single_regular_polygon_rasterizes_to_approximately_its_circle_area() {
        let n_rays = 32;
        let radius = 10.0f32;
        let size = 40;
        let mut prob = vec![0.0f32; size * size];
        let mut dist = vec![0.0f32; size * size * n_rays];
        let center = (20usize, 20usize);
        prob[center.0 * size + center.1] = 0.9;
        for k in 0..n_rays {
            dist[(center.0 * size + center.1) * n_rays + k] = radius;
        }
        let config = StarDistDecodeConfig {
            prob_thresh: 0.5,
            nms_thresh: 0.5,
            grid: (1, 1),
            border: 2,
        };
        let shapes = square_shapes(size, n_rays);
        let result = decode(&prob, &dist, &shapes, &config).unwrap();
        assert_eq!(result.detections.len(), 1);
        let area = result.labels.iter().filter(|&&l| l != 0).count();
        let expected = std::f64::consts::PI * f64::from(radius) * f64::from(radius);
        let relative_error = (area as f64 - expected).abs() / expected;
        assert!(relative_error < 0.05, "area={area} expected~{expected}");
    }

    #[test]
    fn two_separated_candidates_both_survive_nms() {
        let n_rays = 16;
        let size = 40;
        let mut prob = vec![0.0f32; size * size];
        let mut dist = vec![0.0f32; size * size * n_rays];
        for &(r, c, p) in &[(10usize, 10usize, 0.9f32), (30usize, 30usize, 0.8f32)] {
            prob[r * size + c] = p;
            for k in 0..n_rays {
                dist[(r * size + c) * n_rays + k] = 5.0;
            }
        }
        let config = StarDistDecodeConfig::default();
        let shapes = square_shapes(size, n_rays);
        let result = decode(&prob, &dist, &shapes, &config).unwrap();
        assert_eq!(result.detections.len(), 2);
    }

    #[test]
    fn two_heavily_overlapping_candidates_reduce_to_one_via_nms() {
        let n_rays = 16;
        let size = 40;
        let mut prob = vec![0.0f32; size * size];
        let mut dist = vec![0.0f32; size * size * n_rays];
        // Two centers only 2px apart, both radius 8 -- near-total overlap.
        for &(r, c, p) in &[(20usize, 20usize, 0.95f32), (20usize, 22usize, 0.7f32)] {
            prob[r * size + c] = p;
            for k in 0..n_rays {
                dist[(r * size + c) * n_rays + k] = 8.0;
            }
        }
        let config = StarDistDecodeConfig {
            nms_thresh: 0.5,
            ..StarDistDecodeConfig::default()
        };
        let shapes = square_shapes(size, n_rays);
        let result = decode(&prob, &dist, &shapes, &config).unwrap();
        assert_eq!(result.detections.len(), 1);
        // The higher-probability candidate (0.95, at col 20) must be
        // the one that survives, matching NMS's score-descending order.
        assert_eq!(result.detections[0].col, 20.0);
    }

    #[test]
    fn candidates_within_the_border_band_are_excluded() {
        let n_rays = 8;
        let size = 20;
        let mut prob = vec![0.0f32; size * size];
        let dist = vec![3.0f32; size * size * n_rays];
        prob[0] = 0.9; // row 0, col 0: inside the default border=2 band
        prob[10 * size + 10] = 0.9; // interior, should survive
        let config = StarDistDecodeConfig::default();
        let shapes = square_shapes(size, n_rays);
        let result = decode(&prob, &dist, &shapes, &config).unwrap();
        assert_eq!(result.detections.len(), 1);
        assert_eq!(
            (result.detections[0].row, result.detections[0].col),
            (10.0, 10.0)
        );
    }

    #[test]
    fn grid_upscales_candidate_centers_to_image_coordinates() {
        let n_rays = 8;
        let out_size = 10;
        let image_size = 20;
        let mut prob = vec![0.0f32; out_size * out_size];
        let dist = vec![3.0f32; out_size * out_size * n_rays];
        prob[5 * out_size + 5] = 0.9;
        let config = StarDistDecodeConfig {
            grid: (2, 2),
            border: 1,
            ..StarDistDecodeConfig::default()
        };
        let shapes = StarDistShapes {
            out_rows: out_size,
            out_cols: out_size,
            n_rays,
            image_rows: image_size,
            image_cols: image_size,
        };
        let result = decode(&prob, &dist, &shapes, &config).unwrap();
        assert_eq!(result.detections.len(), 1);
        assert_eq!(
            (result.detections[0].row, result.detections[0].col),
            (10.0, 10.0)
        );
    }

    #[test]
    fn rejects_mismatched_prob_length() {
        let shapes = square_shapes(10, 8);
        let err = decode(
            &[0.0; 3],
            &[0.0; 24],
            &shapes,
            &StarDistDecodeConfig::default(),
        )
        .unwrap_err();
        assert!(err.contains("prob length"));
    }

    #[test]
    fn rejects_invalid_decode_tensors_and_config() {
        let shapes = square_shapes(1, 4);
        let config = StarDistDecodeConfig {
            border: 0,
            ..Default::default()
        };
        for p in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(
                decode(&[p], &[1.0; 4], &shapes, &config).is_err(),
                "prob={p}"
            );
        }
        for d in [f32::NAN, f32::NEG_INFINITY] {
            assert!(
                decode(&[0.0], &[d; 4], &shapes, &config).is_err(),
                "dist={d}"
            );
        }
        for invalid in [
            StarDistDecodeConfig {
                grid: (0, 1),
                ..config
            },
            StarDistDecodeConfig {
                prob_thresh: f32::NAN,
                ..config
            },
            StarDistDecodeConfig {
                nms_thresh: 1.1,
                ..config
            },
        ] {
            assert!(
                decode(&[0.9], &[1.0; 4], &shapes, &invalid).is_err(),
                "{invalid:?}"
            );
        }
        for invalid in [
            StarDistShapes {
                n_rays: 2,
                ..shapes
            },
            StarDistShapes {
                n_rays: usize::MAX,
                ..square_shapes(2, 4)
            },
            StarDistShapes {
                out_rows: 0,
                ..shapes
            },
            StarDistShapes {
                image_cols: 2,
                ..shapes
            },
            StarDistShapes {
                out_rows: usize::MAX,
                out_cols: 2,
                ..shapes
            },
        ] {
            assert!(
                decode(&[0.9; 4], &[1.0; 16], &invalid, &config).is_err(),
                "{invalid:?}"
            );
        }
        let huge_grid = StarDistDecodeConfig {
            grid: (usize::MAX, 1),
            ..config
        };
        let huge_image = StarDistShapes {
            image_rows: usize::MAX,
            ..shapes
        };
        assert!(decode(&[0.9], &[1.0; 4], &huge_image, &huge_grid).is_err());
    }

    #[test]
    fn rasterizes_integer_pixel_centers_and_includes_the_polygon_boundary() {
        let mut prob = [0.0; 25];
        prob[12] = 0.9;
        let config = StarDistDecodeConfig {
            border: 0,
            ..Default::default()
        };
        let out = decode(&prob, &[2.0; 100], &square_shapes(5, 4), &config).unwrap();
        assert_eq!(
            out.labels,
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,]
        );
    }

    #[test]
    fn wide_polygons_store_one_span_per_row() {
        let width = 1_000_000_000;
        let edge = (width - 1) as f64;
        let mut rasterizer = Rasterizer {
            vertices: vec![(0.0, 0.0), (0.0, edge), (3.0, edge), (3.0, 0.0)],
            ..Rasterizer::default()
        };
        let bounds = Bounds::clipped(&rasterizer.vertices, 4, width);
        let mut spans = Vec::new();
        assert_eq!(rasterizer.rasterize(bounds, &mut spans), 4 * width);
        assert_eq!(spans.len(), 4);
        let subset = [
            Span {
                row: 0,
                start: 3,
                end: 5,
            },
            Span {
                row: 0,
                start: 7,
                end: 9,
            },
            Span {
                row: 2,
                start: width - 1,
                end: width + 1,
            },
        ];
        assert_eq!(shared_pixels(&spans, &subset, f64::INFINITY), 5);
    }

    #[test]
    fn floors_small_and_negative_rays_and_preserves_partial_grid_cells() {
        let shapes = StarDistShapes {
            out_rows: 2,
            out_cols: 3,
            n_rays: 4,
            image_rows: 3,
            image_cols: 5,
        };
        let config = StarDistDecodeConfig {
            grid: (2, 2),
            border: 0,
            ..Default::default()
        };
        let prob = [0.0, 0.0, 0.0, 0.0, 0.0, 0.9];
        let out = decode(&prob, &[-10.0; 24], &shapes, &config).unwrap();
        assert_eq!(out.labels, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(out.detections.len(), 1);
        assert_eq!((out.detections[0].row, out.detections[0].col), (2.0, 4.0));
    }
}

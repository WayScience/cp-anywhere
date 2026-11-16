//! Standard 2D MeasureObjectSizeShape measurements (CellProfiler 4.2.8).
//!
//! Units are pixels (orientation is degrees). Bounding-box maxima are exclusive.
//! Disconnected pixels carrying one label are measured as a single object.
//! The skimage perimeter estimator, diamond-offset raster convex hull, 8-connected
//! Euler number, pixel-center Feret diameters and padded Euclidean radii match
//! the compiled CP 4.2.8 / scikit-image 0.21.0 / centrosome 1.2.3 reference.
//!
//! Sparse labels deliberately retain CP's inconsistent vector lengths:
//! regionprops columns contain present labels in ascending order, radius columns
//! pack those objects followed by zeros up to the maximum label, and Feret
//! columns are indexed by label minus one. Callers requiring aligned columns
//! should reject or explicitly relabel sparse inputs before calling this module.
//!
//! The perimeter stencil/weights and inertia/orientation conventions below are
//! adapted from scikit-image 0.21.0, skimage/measure/_regionprops_utils.py,
//! _moments.py and _regionprops.py. Its BSD-3-Clause notice follows:
//!
//! Copyright: 2009-2022 the scikit-image team
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//! 1. Redistributions of source code must retain the above copyright notice,
//!    this list of conditions and the following disclaimer.
//! 2. Redistributions in binary form must reproduce the above copyright notice,
//!    this list of conditions and the following disclaimer in the documentation
//!    and/or other materials provided with the distribution.
//! 3. Neither the name of the copyright holder nor the names of its contributors
//!    may be used to endorse or promote products derived from this software
//!    without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//! ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//! LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//! CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//! SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//! INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//! CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//! ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//! POSSIBILITY OF SUCH DAMAGE.

use std::collections::BTreeMap;
use std::f64::consts::PI;

const FEATURES: [&str; 25] = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "Orientation",
    "Center_X",
    "Center_Y",
    "BoundingBoxArea",
    "BoundingBoxMinimum_X",
    "BoundingBoxMaximum_X",
    "BoundingBoxMinimum_Y",
    "BoundingBoxMaximum_Y",
    "FormFactor",
    "Extent",
    "Solidity",
    "Compactness",
    "EulerNumber",
    "MaximumRadius",
    "MeanRadius",
    "MedianRadius",
    "ConvexArea",
    "MinFeretDiameter",
    "MaxFeretDiameter",
    "EquivalentDiameter",
];

#[derive(Default)]
struct Object {
    pixels: Vec<(usize, usize)>,
    min_y: usize,
    max_y: usize,
    min_x: usize,
    max_x: usize,
}

/// Measure the 25 standard 2D size/shape columns, without Zernike or moments.
///
/// Returns an error for mismatched/overflowing dimensions or an unallocatable
/// sparse-label result. Empty input returns all 25 columns with empty vectors.
/// See the module documentation for CP's sparse-label vector-length behavior.
pub fn measure(
    labels: &[u32],
    rows: usize,
    cols: usize,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    if rows.checked_mul(cols) != Some(labels.len()) {
        return Err("size/shape dimensions do not match the label buffer".into());
    }
    let mut objects = BTreeMap::<u32, Object>::new();
    for (index, &label) in labels.iter().enumerate() {
        if label == 0 {
            continue;
        }
        let y = index / cols;
        let x = index % cols;
        let object = objects.entry(label).or_insert_with(|| Object {
            min_y: y,
            max_y: y,
            min_x: x,
            max_x: x,
            ..Object::default()
        });
        object.pixels.push((y, x));
        object.min_y = object.min_y.min(y);
        object.max_y = object.max_y.max(y);
        object.min_x = object.min_x.min(x);
        object.max_x = object.max_x.max(x);
    }
    let max_label = objects
        .last_key_value()
        .map_or(0, |(&label, _)| label as usize);
    let mut columns: Vec<Vec<f64>> = FEATURES.iter().map(|_| Vec::new()).collect();
    // CP allocates these five columns to max(label), not count(present labels).
    for &column in &[18, 19, 20, 22, 23] {
        columns[column]
            .try_reserve_exact(max_label)
            .map_err(|_| "size/shape cannot allocate sparse-label output".to_string())?;
        columns[column].resize(max_label, 0.0);
    }
    for (object_index, (&label, object)) in objects.iter().enumerate() {
        let height = object.max_y - object.min_y + 1;
        let width = object.max_x - object.min_x + 1;
        let mut mask = vec![false; height * width];
        let mut points = Vec::with_capacity(object.pixels.len());
        let mut sum_y = 0.0;
        let mut sum_x = 0.0;
        for &(y, x) in &object.pixels {
            let local_y = y - object.min_y;
            let local_x = x - object.min_x;
            mask[local_y * width + local_x] = true;
            points.push((local_x as i64, local_y as i64));
            sum_y += local_y as f64;
            sum_x += local_x as f64;
        }
        let area = points.len() as f64;
        let cy = sum_y / area;
        let cx = sum_x / area;
        // Separable central-moment accumulation follows skimage's axis order.
        let (mut mu20, mut mu02, mut mu11) = (0.0, 0.0, 0.0);
        for x in 0..width {
            let (mut count, mut first_y, mut second_y) = (0.0, 0.0, 0.0);
            for y in 0..height {
                if mask[y * width + x] {
                    let dy = y as f64 - cy;
                    count += 1.0;
                    first_y += dy;
                    second_y += dy * dy;
                }
            }
            let dx = x as f64 - cx;
            mu20 += second_y;
            mu02 += count * dx * dx;
            mu11 += first_y * dx;
        }
        // Match skimage's sum-then-subtract inertia diagonal, including ties.
        let a = ((mu20 + mu02) - mu20) / area;
        let b = -mu11 / area;
        let c = ((mu20 + mu02) - mu02) / area;
        let (major, minor) = inertia_eigenvalues(a, b, c);
        let orientation = if a - c == 0.0 {
            if b < 0.0 {
                45.0
            } else {
                -45.0
            }
        } else {
            0.5 * (-2.0 * b).atan2(c - a) * (180.0 / PI)
        };
        let perimeter = perimeter(&mask, height, width);
        let hull = convex_hull(points);
        let convex_area = convex_image_area(&hull, height) as f64;
        let (min_feret, max_feret) = feret_diameters(&hull);
        let (maximum_radius, mean_radius, median_radius) = radii(&mask, height, width);
        let bbox_area = (height * width) as f64;
        let values = [
            area,
            perimeter,
            4.0 * major.sqrt(),
            4.0 * minor.sqrt(),
            if major == 0.0 {
                0.0
            } else {
                (1.0 - minor / major).sqrt()
            },
            orientation,
            cx + object.min_x as f64,
            cy + object.min_y as f64,
            bbox_area,
            object.min_x as f64,
            (object.max_x + 1) as f64,
            object.min_y as f64,
            (object.max_y + 1) as f64,
            4.0 * PI * area / perimeter.powi(2),
            area / bbox_area,
            area / convex_area,
            perimeter.powi(2) / (4.0 * PI * area).max(1.0),
            euler_number(&mask, height, width) as f64,
            maximum_radius,
            mean_radius,
            median_radius,
            convex_area,
            min_feret,
            max_feret,
            (4.0 * area / PI).sqrt(),
        ];
        for (column, &value) in values.iter().enumerate() {
            match column {
                18..=20 => columns[column][object_index] = value,
                22 | 23 => columns[column][label as usize - 1] = value,
                _ => columns[column].push(value),
            }
        }
    }
    Ok(FEATURES
        .iter()
        .zip(columns)
        .map(|(name, values)| (format!("AreaShape_{name}"), values))
        .collect())
}

// LAPACK DLAE2 (v3.10.1, SRC/dlae2.f), adapted for a positive-semidefinite
// inertia tensor. Explicit FMA matches the compiled reference OpenBLAS 0.3.20
// armv8 routine: a trace/discriminant shortcut loses sub-ulp minor eigenvalues.
//
// Copyright (c) 1992-2022 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2022 The University of California Berkeley.
// All rights reserved.
// Copyright (c) 2006-2022 The University of Colorado Denver.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer listed in this license
//   in the documentation and/or other materials provided with the distribution.
// - Neither the name of the copyright holders nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// The copyright holders provide no reassurances that the source code provided
// does not infringe any patent, copyright, or any other intellectual property
// rights of third parties. The copyright holders disclaim any liability to any
// recipient for claims brought against recipient by any third party for
// infringement of that parties intellectual property rights.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
fn inertia_eigenvalues(a: f64, b: f64, c: f64) -> (f64, f64) {
    let sum = a + c;
    let difference = (a - c).abs();
    let twice_b = (b + b).abs();
    let root = if difference > twice_b {
        let ratio = twice_b / difference;
        difference * ratio.mul_add(ratio, 1.0).sqrt()
    } else if difference < twice_b {
        let ratio = difference / twice_b;
        twice_b * ratio.mul_add(ratio, 1.0).sqrt()
    } else {
        twice_b * 2.0f64.sqrt()
    };
    if sum == 0.0 {
        return (0.5 * root, 0.0);
    }
    let major = 0.5 * (sum + root);
    let (large, small) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let minor = (large / major).mul_add(small, -((b / major) * b));
    (major.max(0.0), minor.max(0.0))
}

fn perimeter(mask: &[bool], height: usize, width: usize) -> f64 {
    let mut border = vec![false; mask.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let interior = y > 0
                && x > 0
                && y + 1 < height
                && x + 1 < width
                && mask[index - width]
                && mask[index + width]
                && mask[index - 1]
                && mask[index + 1];
            border[index] = mask[index] && !interior;
        }
    }
    let mut histogram = [0usize; 50];
    for y in 0..height {
        for x in 0..width {
            let mut code = 0;
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if border[yy * width + xx] {
                        code += if yy == y && xx == x {
                            1
                        } else if yy == y || xx == x {
                            2
                        } else {
                            10
                        };
                    }
                }
            }
            histogram[code] += 1;
        }
    }
    let mut result = 0.0;
    for (code, &count) in histogram.iter().enumerate() {
        let weight = match code {
            5 | 7 | 15 | 17 | 25 | 27 => 1.0,
            21 | 33 => 2.0f64.sqrt(),
            13 | 23 => (1.0 + 2.0f64.sqrt()) / 2.0,
            _ => 0.0,
        };
        result += count as f64 * weight;
    }
    result
}

// Exact integer monotone-chain hull of pixel centers; collinear interior points
// are removed, as in centrosome's compiled convex_hull_ijv.
type Point = (i64, i64);

fn cross(a: Point, b: Point, c: Point) -> i128 {
    (b.0 - a.0) as i128 * (c.1 - a.1) as i128 - (b.1 - a.1) as i128 * (c.0 - a.0) as i128
}

fn convex_hull(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_unstable();
    points.dedup();
    if points.len() <= 1 {
        return points;
    }
    let mut hull = Vec::with_capacity(points.len() * 2);
    for &p in &points {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_length = hull.len();
    for &p in points.iter().rev().skip(1) {
        while hull.len() > lower_length && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

fn convex_image_area(hull: &[Point], height: usize) -> usize {
    // Minkowski sum with the four pixel-edge midpoints, not pixel corners and
    // not polygon area. Only raster centers in this offset hull are counted.
    let mut expanded = Vec::with_capacity(hull.len() * 4);
    for &(x, y) in hull {
        expanded.extend([
            (2 * x - 1, 2 * y),
            (2 * x + 1, 2 * y),
            (2 * x, 2 * y - 1),
            (2 * x, 2 * y + 1),
        ]);
    }
    let expanded = convex_hull(expanded);
    let mut area = 0;
    for row in 0..height {
        let y = 2 * row as i64;
        let mut low: Option<(i128, i128)> = None;
        let mut high: Option<(i128, i128)> = None;
        for i in 0..expanded.len() {
            let (mut a, mut b) = (expanded[i], expanded[(i + 1) % expanded.len()]);
            if a.1 > b.1 {
                std::mem::swap(&mut a, &mut b);
            }
            if a.1 > y || y > b.1 {
                continue;
            }
            let candidates = if a.1 == b.1 {
                [(a.0 as i128, 1), (b.0 as i128, 1)]
            } else {
                let denominator = (b.1 - a.1) as i128;
                let numerator = a.0 as i128 * denominator + (y - a.1) as i128 * (b.0 - a.0) as i128;
                [(numerator, denominator); 2]
            };
            for (n, d) in candidates {
                if low.is_none_or(|(ln, ld)| n * ld < ln * d) {
                    low = Some((n, d));
                }
                if high.is_none_or(|(hn, hd)| n * hd > hn * d) {
                    high = Some((n, d));
                }
            }
        }
        if let (Some((ln, ld)), Some((hn, hd))) = (low, high) {
            let first = -(-ln).div_euclid(2 * ld);
            let last = hn.div_euclid(2 * hd);
            area += (last - first + 1).max(0) as usize;
        }
    }
    area
}

fn feret_diameters(hull: &[Point]) -> (f64, f64) {
    if hull.len() < 2 {
        return (0.0, 0.0);
    }
    let mut max_squared: f64 = 0.0;
    let mut min_width = f64::INFINITY;
    for (i, &a) in hull.iter().enumerate() {
        let b = hull[(i + 1) % hull.len()];
        let edge_squared =
            (b.0 - a.0) as f64 * (b.0 - a.0) as f64 + (b.1 - a.1) as f64 * (b.1 - a.1) as f64;
        let mut max_cross = 0i128;
        for &p in hull {
            let dx = (p.0 - a.0) as f64;
            let dy = (p.1 - a.1) as f64;
            max_squared = max_squared.max(dx * dx + dy * dy);
            max_cross = max_cross.max(cross(a, b, p).abs());
        }
        min_width = min_width.min(max_cross as f64 / edge_squared.sqrt());
    }
    (min_width, max_squared.sqrt())
}

fn euler_number(mask: &[bool], height: usize, width: usize) -> i64 {
    let mut seen = vec![false; mask.len()];
    let mut stack = Vec::new();
    let mut result = 0;
    for start in 0..mask.len() {
        if seen[start] {
            continue;
        }
        let foreground = mask[start];
        let mut touches_edge = false;
        seen[start] = true;
        stack.push(start);
        while let Some(index) = stack.pop() {
            let y = index / width;
            let x = index % width;
            touches_edge |= y == 0 || x == 0 || y + 1 == height || x + 1 == width;
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if !foreground && yy != y && xx != x {
                        continue;
                    }
                    let neighbor = yy * width + xx;
                    if !seen[neighbor] && mask[neighbor] == foreground {
                        seen[neighbor] = true;
                        stack.push(neighbor);
                    }
                }
            }
        }
        if foreground {
            result += 1;
        } else if !touches_edge {
            result -= 1;
        }
    }
    result
}

fn radii(mask: &[bool], height: usize, width: usize) -> (f64, f64, f64) {
    // Exact separable squared Euclidean distance transform. Padding guarantees
    // finite horizontal distances; the vertical pass uses a lower envelope of
    // parabolas. This is O(bounding-box area), not a foreground/background scan.
    let h = height + 2;
    let w = width + 2;
    let mut squared = vec![0.0; h * w];
    for y in 1..=height {
        let mut previous_zero = 0;
        for x in 1..=width {
            if !mask[(y - 1) * width + x - 1] {
                previous_zero = x;
            }
            squared[y * w + x] = ((x - previous_zero) as f64).powi(2);
        }
        previous_zero = width + 1;
        for x in (1..=width).rev() {
            if !mask[(y - 1) * width + x - 1] {
                previous_zero = x;
            }
            squared[y * w + x] = squared[y * w + x].min(((previous_zero - x) as f64).powi(2));
        }
    }
    let mut f = vec![0.0; h];
    let mut sites = vec![0; h];
    let mut boundaries = vec![0.0; h + 1];
    for x in 1..=width {
        for y in 0..h {
            f[y] = squared[y * w + x];
        }
        let mut k = 0;
        sites[0] = 0;
        boundaries[0] = f64::NEG_INFINITY;
        boundaries[1] = f64::INFINITY;
        for q in 1..h {
            let mut s;
            loop {
                let v = sites[k];
                s = ((f[q] + (q as f64).powi(2)) - (f[v] + (v as f64).powi(2)))
                    / (2.0 * (q - v) as f64);
                if s > boundaries[k] {
                    break;
                }
                k -= 1;
            }
            k += 1;
            sites[k] = q;
            boundaries[k] = s;
            boundaries[k + 1] = f64::INFINITY;
        }
        k = 0;
        for q in 0..h {
            while boundaries[k + 1] < q as f64 {
                k += 1;
            }
            let delta = q as f64 - sites[k] as f64;
            squared[q * w + x] = delta * delta + f[sites[k]];
        }
    }
    let mut distances = Vec::new();
    let mut sum = 0.0;
    let mut maximum: f64 = 0.0;
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] {
                let value = squared[(y + 1) * w + x + 1].sqrt();
                distances.push(value);
                sum += value;
                maximum = maximum.max(value);
            }
        }
    }
    let mean = sum / distances.len() as f64;
    distances.sort_unstable_by(f64::total_cmp);
    let middle = distances.len() / 2;
    let median = if distances.len() % 2 == 0 {
        (distances[middle - 1] + distances[middle]) / 2.0
    } else {
        distances[middle]
    };
    (maximum, mean, median)
}

#[cfg(test)]
// Oracle values are literals, including values equal to mathematical constants.
#[allow(clippy::approx_constant)]
mod tests {
    use super::measure;

    fn close(actual: f64, expected: f64) {
        assert!(
            actual == expected
                || (actual.is_nan() && expected.is_nan())
                || (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn rectangle_matches_cellprofiler_428_literals() {
        // fixtures/size_shape/rectangle.json: real CP 4.2.8 + skimage 0.21.0.
        let mut labels = vec![0; 12 * 14];
        for y in 2..8 {
            for x in 3..11 {
                labels[y * 14 + x] = 1;
            }
        }
        let actual = measure(&labels, 12, 14).unwrap();
        let expected = [
            ("Area", 48.0),
            ("BoundingBoxArea", 48.0),
            ("BoundingBoxMaximum_X", 11.0),
            ("BoundingBoxMaximum_Y", 8.0),
            ("BoundingBoxMinimum_X", 3.0),
            ("BoundingBoxMinimum_Y", 2.0),
            ("Center_X", 6.5),
            ("Center_Y", 4.5),
            ("Compactness", 0.954929658551372),
            ("ConvexArea", 48.0),
            ("Eccentricity", 0.6666666666666666),
            ("EquivalentDiameter", 7.817640190446719),
            ("EulerNumber", 1.0),
            ("Extent", 1.0),
            ("FormFactor", 1.0471975511965976),
            ("MajorAxisLength", 9.16515138991168),
            ("MaxFeretDiameter", 8.602325267042627),
            ("MaximumRadius", 3.0),
            ("MeanRadius", 1.6666666666666667),
            ("MedianRadius", 1.5),
            ("MinFeretDiameter", 5.0),
            ("MinorAxisLength", 6.831300510639732),
            ("Orientation", 90.0),
            ("Perimeter", 24.0),
            ("Solidity", 1.0),
        ];
        assert_eq!(actual.len(), expected.len());
        for (name, value) in expected {
            let values = &actual[&format!("AreaShape_{name}")];
            assert_eq!(values.len(), 1);
            close(values[0], value);
        }
    }

    #[test]
    fn singleton_preserves_reference_infinity_and_orientation() {
        let actual = measure(&[0, 0, 0, 1], 2, 2).unwrap();
        for (name, value) in [
            ("Area", 1.0),
            ("Perimeter", 0.0),
            ("FormFactor", f64::INFINITY),
            ("Compactness", 0.0),
            ("Orientation", -45.0),
            ("ConvexArea", 1.0),
            ("MinFeretDiameter", 0.0),
            ("MaxFeretDiameter", 0.0),
            ("MaximumRadius", 1.0),
            ("MeanRadius", 1.0),
            ("MedianRadius", 1.0),
            ("EulerNumber", 1.0),
        ] {
            close(actual[&format!("AreaShape_{name}")][0], value);
        }
    }

    #[test]
    fn sparse_labels_preserve_reference_misalignment() {
        // Actual CP returns compressed regionprops, packed radius vectors with
        // trailing zeros, and label-indexed Feret vectors. Do not relabel silently.
        let actual = measure(&[0, 1, 0, 0, 0, 3], 2, 3).unwrap();
        assert_eq!(actual["AreaShape_Area"], vec![1.0, 1.0]);
        assert_eq!(actual["AreaShape_Center_X"], vec![1.0, 2.0]);
        assert_eq!(actual["AreaShape_MaximumRadius"], vec![1.0, 1.0, 0.0]);
        assert_eq!(actual["AreaShape_MeanRadius"], vec![1.0, 1.0, 0.0]);
        assert_eq!(actual["AreaShape_MedianRadius"], vec![1.0, 1.0, 0.0]);
        assert_eq!(actual["AreaShape_MaxFeretDiameter"], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_images_return_all_25_empty_columns() {
        for (labels, rows, cols) in [(vec![0; 6], 2, 3), (vec![], 0, 3), (vec![], 3, 0)] {
            let actual = measure(&labels, rows, cols).unwrap();
            assert_eq!(actual.len(), 25);
            assert!(actual.values().all(Vec::is_empty));
        }
    }

    fn expect_one(labels: &[u32], rows: usize, cols: usize, expected: [f64; 25]) {
        let actual = measure(labels, rows, cols).unwrap();
        assert_eq!(actual.len(), 25);
        for (name, value) in super::FEATURES.iter().zip(expected) {
            let values = &actual[&format!("AreaShape_{name}")];
            assert_eq!(values.len(), 1, "{name}");
            close(values[0], value);
        }
    }

    #[test]
    fn ring_matches_cellprofiler_428_literals() {
        // Full standard columns from fixtures/size_shape/ring.json.
        let mut labels = vec![0; 12 * 14];
        for y in 2..8 {
            for x in 3..11 {
                if !(4..6).contains(&y) || !(5..9).contains(&x) {
                    labels[y * 14 + x] = 1;
                }
            }
        }
        expect_one(
            &labels,
            12,
            14,
            [
                40.0,
                28.0,
                9.838699100999074,
                7.429670248402684,
                0.6555547773570889,
                90.0,
                6.5,
                4.5,
                48.0,
                3.0,
                11.0,
                2.0,
                8.0,
                0.641141357875468,
                0.8333333333333334,
                0.8333333333333334,
                1.5597184423005743,
                0.0,
                1.4142135623730951,
                1.0414213562373096,
                1.0,
                48.0,
                5.0,
                8.602325267042627,
                7.136496464611085,
            ],
        );
    }

    #[test]
    fn diagonal_matches_cellprofiler_428_literals() {
        // Full standard columns from fixtures/size_shape/diagonal.json.
        let mut labels = vec![0; 12 * 14];
        for y in 2..9 {
            labels[y * 14 + y] = 1;
            labels[y * 14 + y + 1] = 1;
        }
        expect_one(
            &labels,
            12,
            14,
            [
                14.0,
                14.414213562373096,
                11.403124237432849,
                1.4031242374328479,
                0.9924008007166892,
                45.89495530412304,
                5.5,
                5.0,
                56.0,
                2.0,
                10.0,
                2.0,
                9.0,
                0.8467515397761582,
                0.25,
                1.0,
                1.1809839758476892,
                1.0,
                1.0,
                1.0,
                1.0,
                14.0,
                0.7071067811865476,
                9.219544457292887,
                4.222008245644752,
            ],
        );
    }

    #[test]
    fn diamond_raster_hull_and_noninteger_radii_match_oracle() {
        // Real CP run: raster convex area is 13, center-hull polygon area is 8.
        let labels = [
            0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
        ];
        expect_one(
            &labels,
            5,
            5,
            [
                13.0,
                11.313708498984761,
                4.1509961733021665,
                4.1509961733021665,
                0.0,
                -45.0,
                2.0,
                2.0,
                25.0,
                0.0,
                5.0,
                0.0,
                5.0,
                1.2762720155208533,
                0.52,
                1.0,
                0.783532027529331,
                1.0,
                2.23606797749979,
                1.2225324789993977,
                1.0,
                13.0,
                2.8284271247461903,
                4.0,
                4.068428945128219,
            ],
        );
    }

    #[test]
    fn nested_objects_and_border_padding_are_independent() {
        let mut labels = vec![0; 12 * 14];
        for y in 2..8 {
            for x in 3..11 {
                if !(4..6).contains(&y) || !(5..9).contains(&x) {
                    labels[y * 14 + x] = 1;
                } else if (6..8).contains(&x) {
                    labels[y * 14 + x] = 2;
                }
            }
        }
        let actual = measure(&labels, 12, 14).unwrap();
        assert_eq!(actual["AreaShape_Area"], vec![40.0, 4.0]);
        assert_eq!(actual["AreaShape_EulerNumber"], vec![0.0, 1.0]);
        assert_eq!(actual["AreaShape_ConvexArea"], vec![48.0, 4.0]);
        assert_eq!(actual["AreaShape_MaximumRadius"][1], 1.0);
        close(actual["AreaShape_MaximumRadius"][0], 1.4142135623730951);
        // Same 5x4 border rectangle as the saved CP fixture: exterior is zero,
        // even when the entire input buffer is foreground.
        let border = measure(&[1; 20], 5, 4).unwrap();
        assert_eq!(border["AreaShape_MaximumRadius"], vec![2.0]);
        assert_eq!(border["AreaShape_MeanRadius"], vec![1.3]);
        assert_eq!(border["AreaShape_MinFeretDiameter"], vec![3.0]);
    }

    #[test]
    fn sparse_feret_columns_are_label_indexed_not_packed() {
        let labels = [1, 1, 0, 0, 0, 1, 1, 0, 3, 3, 0, 0, 0, 3, 3];
        let actual = measure(&labels, 3, 5).unwrap();
        assert_eq!(actual["AreaShape_Area"], vec![4.0, 4.0]);
        assert_eq!(actual["AreaShape_MaximumRadius"], vec![1.0, 1.0, 0.0]);
        assert_eq!(actual["AreaShape_MinFeretDiameter"], vec![1.0, 0.0, 1.0]);
        close(actual["AreaShape_MaxFeretDiameter"][0], 1.4142135623730951);
        assert_eq!(actual["AreaShape_MaxFeretDiameter"][1], 0.0);
        close(actual["AreaShape_MaxFeretDiameter"][2], 1.4142135623730951);
    }

    #[test]
    fn rank_one_axis_keeps_compiled_reference_roundoff() {
        // CP/NumPy's compiled LAPACK keeps a positive sub-ulp eigenvalue here.
        // The mathematically exact zero minor axis is NOT the oracle output.
        let labels = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1];
        let actual = measure(&labels, 3, 5).unwrap();
        close(actual["AreaShape_MinorAxisLength"][0], 1.088226949185463e-8);
    }

    #[test]
    fn invalid_dimensions_are_errors() {
        assert!(measure(&[1], 2, 2).is_err());
        assert!(measure(&[], usize::MAX, 2).is_err());
    }
}

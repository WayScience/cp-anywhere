//! CellProfiler 4.2.8 optional two-dimensional size/shape measurements.
//!
//! Behavioral reference: centrosome 1.2.3 and scikit-image 0.21.0. The hull,
//! minimum-circle, radial-polynomial and Hu arithmetic below adapt those
//! upstream BSD implementations. Their redistribution notices follow.

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

If, by operation of law or otherwise, any of the aforementioned warranty
disclaimers are determined inapplicable, your sole remedy, regardless of the
form of action, including, but not limited to, negligence and strict
liability, shall be replacement of the software with an updated version if one
exists.

scikit-image: Copyright 2009-2022 the scikit-image team. BSD-3-Clause.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

use std::collections::BTreeMap;

/// Return only the optional CP 4.2.8 2D features selected by the two flags.
///
/// Entries follow label IDs 1..=max(labels); absent IDs contain NaNs. Pixel
/// coordinates for moments are local to each object's bounding box. Zernike
/// coordinates instead use the pixel-center minimum enclosing circle.
/// Undefined values deliberately remain IEEE NaN/inf, including singleton
/// Zernikes and the three normalized moments of total order less than two.
pub fn measure(
    labels: &[u32],
    rows: usize,
    cols: usize,
    calculate_zernikes: bool,
    calculate_advanced: bool,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    if rows == 0 || cols == 0 {
        return Err("size/shape requires two nonzero dimensions".into());
    }
    if rows.checked_mul(cols) != Some(labels.len()) {
        return Err("label buffer length does not match size/shape dimensions".into());
    }
    let mut result = BTreeMap::new();
    if !calculate_zernikes && !calculate_advanced {
        return Ok(result);
    }
    let count = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut points: BTreeMap<u32, Vec<Point>> = BTreeMap::new();
    for (index, &label) in labels.iter().enumerate() {
        if label != 0 {
            points.entry(label).or_default().push(Point {
                row: (index / cols) as f64,
                col: (index % cols) as f64,
            });
        }
    }
    // Build empty columns as well: an empty image still has the selected schema.
    let mut add = |name: String| -> Result<(), String> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| "cannot allocate size/shape output")?;
        values.resize(count, f64::NAN);
        result.insert(format!("AreaShape_{name}"), values);
        Ok(())
    };
    if calculate_advanced {
        for family in ["SpatialMoment", "CentralMoment", "NormalizedMoment"] {
            let row_orders = if family == "NormalizedMoment" { 4 } else { 3 };
            for i in 0..row_orders {
                for j in 0..4 {
                    add(format!("{family}_{i}_{j}"))?;
                }
            }
        }
        for i in 0..7 {
            add(format!("HuMoment_{i}"))?;
        }
        for i in 0..2 {
            for j in 0..2 {
                add(format!("InertiaTensor_{i}_{j}"))?;
            }
            add(format!("InertiaTensorEigenvalues_{i}"))?;
        }
    }
    if calculate_zernikes {
        for n in 0..10 {
            for m in (n % 2..=n).step_by(2) {
                add(format!("Zernike_{n}_{m}"))?;
            }
        }
    }
    // Centrosome selects candidate vertices from all objects in one global
    // minimum_position sort on every iteration, not one object at a time.
    let circles = if calculate_zernikes {
        let hulls = points
            .iter()
            .map(|(&id, pixels)| (id, convex_hull(pixels)))
            .collect();
        enclosing_circles(hulls)
    } else {
        BTreeMap::new()
    };
    for (&label, pixels) in &points {
        let mut put = |name: String, value: f64| {
            result.get_mut(&format!("AreaShape_{name}")).unwrap()[label as usize - 1] = value;
        };
        if calculate_advanced {
            let min_row = pixels.iter().map(|p| p.row).fold(f64::INFINITY, f64::min);
            let min_col = pixels.iter().map(|p| p.col).fold(f64::INFINITY, f64::min);
            let raw = moments(
                pixels,
                Point {
                    row: min_row,
                    col: min_col,
                },
            );
            let center = Point {
                row: raw[1][0] / raw[0][0],
                col: raw[0][1] / raw[0][0],
            };
            // Subtract the bbox origin before subtracting the local centroid,
            // just as regionprops does; translating the centroid first loses bits.
            let local: Vec<Point> = pixels
                .iter()
                .map(|p| Point {
                    row: p.row - min_row,
                    col: p.col - min_col,
                })
                .collect();
            let central = moments(&local, center);
            let mut normalized = [[f64::NAN; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    if i + j >= 2 {
                        normalized[i][j] =
                            central[i][j] / central[0][0].powf((i + j) as f64 / 2.0 + 1.0);
                    }
                    if i < 3 {
                        put(format!("SpatialMoment_{i}_{j}"), raw[i][j]);
                        put(format!("CentralMoment_{i}_{j}"), central[i][j]);
                    }
                    put(format!("NormalizedMoment_{i}_{j}"), normalized[i][j]);
                }
            }
            for (i, value) in hu_moments(&normalized).into_iter().enumerate() {
                put(format!("HuMoment_{i}"), value);
            }
            let sum = central[2][0] + central[0][2];
            let a = (sum - central[2][0]) / central[0][0];
            let c = (sum - central[0][2]) / central[0][0];
            let b = -central[1][1] / central[0][0];
            for (i, j, value) in [(0, 0, a), (0, 1, b), (1, 0, b), (1, 1, c)] {
                put(format!("InertiaTensor_{i}_{j}"), value);
            }
            let spread = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
            put(
                "InertiaTensorEigenvalues_0".into(),
                ((a + c + spread) / 2.0).max(0.0),
            );
            put(
                "InertiaTensorEigenvalues_1".into(),
                ((a + c - spread) / 2.0).max(0.0),
            );
        }
        if calculate_zernikes {
            let (center, radius) = circles[&label];
            for n in 0..10 {
                for m in (n % 2..=n).step_by(2) {
                    put(
                        format!("Zernike_{n}_{m}"),
                        zernike(pixels, center, radius, n, m),
                    );
                }
            }
        }
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
struct Point {
    row: f64,
    col: f64,
}

// The pinned oracle's NumPy uses OpenBLAS 0.3.20 ARMv8: its reduction axis
// is split into 128-element blocks, with a balanced, four-aligned tail when
// 128 < remaining < 256. Scalar summation across the entire axis disagrees
// on the tiny, nonzero first central moments of larger dense objects.
// This is an explicit reproducibility convention, not a new dependency.
fn dot_blocks(length: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    let mut start = 0;
    std::iter::from_fn(move || {
        let remaining = length - start;
        if remaining == 0 {
            return None;
        }
        let width = if remaining >= 256 {
            128
        } else if remaining > 128 {
            (remaining / 2).div_ceil(4) * 4
        } else {
            remaining
        };
        let block = start..start + width;
        start += width;
        Some(block)
    })
}

// Sum rows first, then columns, as skimage's two matrix products do. Points
// are row-major, and block boundaries are local to the object's actual bbox.
fn moments(points: &[Point], center: Point) -> [[f64; 4]; 4] {
    let min_row = points.first().unwrap().row as usize;
    let rows = points.last().unwrap().row as usize - min_row + 1;
    let min_col = points.iter().map(|p| p.col as usize).min().unwrap();
    let cols = points.iter().map(|p| p.col as usize).max().unwrap() - min_col + 1;
    let mut columns: BTreeMap<usize, [f64; 4]> = BTreeMap::new();
    let mut next = 0;
    for block in dot_blocks(rows) {
        let mut partial: BTreeMap<usize, [f64; 4]> = BTreeMap::new();
        while next < points.len() && (points[next].row as usize) < min_row + block.end {
            let p = points[next];
            let delta = p.row - center.row;
            let powers = [1.0, delta, delta * delta, delta.powf(3.0)];
            let column = partial.entry(p.col as usize).or_default();
            for i in 0..4 {
                column[i] += powers[i];
            }
            next += 1;
        }
        for (col, values) in partial {
            let column = columns.entry(col).or_default();
            for i in 0..4 {
                column[i] += values[i];
            }
        }
    }
    let mut output = [[0.0; 4]; 4];
    for block in dot_blocks(cols) {
        let mut partial = [[0.0; 4]; 4];
        for (&col, values) in columns.range(min_col + block.start..min_col + block.end) {
            let delta = col as f64 - center.col;
            let powers = [1.0, delta, delta * delta, delta.powf(3.0)];
            for i in 0..4 {
                for (j, &power) in powers.iter().enumerate() {
                    // Match the reference matrix kernel's fused multiply-add;
                    // even power-one products change cancellation residuals.
                    partial[i][j] = values[i].mul_add(power, partial[i][j]);
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                output[i][j] += partial[i][j];
            }
        }
    }
    output
}

// Arithmetic ordering follows skimage.measure._moments_cy (BSD-3-Clause;
// copyright 2009-2022 the scikit-image team; notice above).
fn hu_moments(nu: &[[f64; 4]; 4]) -> [f64; 7] {
    let mut t0 = nu[3][0] + nu[1][2];
    let mut t1 = nu[2][1] + nu[0][3];
    let mut q0 = t0 * t0;
    let mut q1 = t1 * t1;
    let n4 = 4.0 * nu[1][1];
    let s = nu[2][0] + nu[0][2];
    let d = nu[2][0] - nu[0][2];
    let mut hu = [0.0; 7];
    hu[0] = s;
    hu[1] = d * d + n4 * nu[1][1];
    hu[3] = q0 + q1;
    hu[5] = d * (q0 - q1) + n4 * t0 * t1;
    t0 *= q0 - 3.0 * q1;
    t1 *= 3.0 * q0 - q1;
    q0 = nu[3][0] - 3.0 * nu[1][2];
    q1 = 3.0 * nu[2][1] - nu[0][3];
    hu[2] = q0 * q0 + q1 * q1;
    hu[4] = q0 * t0 + q1 * t1;
    hu[6] = q1 * t0 - q0 * t1;
    hu
}

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b.col - a.col) * (c.row - b.row) - (c.col - b.col) * (b.row - a.row)
}

fn convex(a: Point, b: Point, c: Point) -> bool {
    let turn = cross(a, b, c);
    turn > 0.0 || (turn == 0.0 && b.col > a.col && b.col > c.col)
}

// The column envelopes, starting vertex, U-turn handling and ordering match
// centrosome._convex_hull, not a hull of pixel *corners*.
fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut envelopes: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for p in points {
        let bounds = envelopes.entry(p.col as usize).or_insert((p.row, p.row));
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

// Chrystal's hull algorithm, with centrosome's angle tests and circumcenter
// arithmetic. A numerically equivalent circle is insufficient: its rounding
// determines whether boundary pixels pass the strict Zernike r^2 <= 1 mask.
struct CircleState {
    label: u32,
    hull: Vec<Point>,
    // Centrosome swaps within_label_indexes when replacing an endpoint.
    // Only vertices numbered >= 2 participate in the next global sort.
    ranks: Vec<usize>,
    s0: usize,
    s1: usize,
    circle: Option<(Point, f64)>,
}

#[derive(Clone, Copy)]
struct CircleCandidate {
    object: usize,
    vertex: usize,
    angle: f64,
    at_a: f64,
    at_b: f64,
}

fn enclosing_circles(hulls: BTreeMap<u32, Vec<Point>>) -> BTreeMap<u32, (Point, f64)> {
    let mut states: Vec<_> = hulls
        .into_iter()
        .map(|(label, hull)| {
            let circle = match hull.len() {
                0 => None,
                1 => Some((hull[0], 0.0)),
                2 => Some(diameter(hull[0], hull[1])),
                _ => None,
            };
            CircleState {
                label,
                ranks: (0..hull.len()).collect(),
                hull,
                s0: 0,
                s1: 1,
                circle,
            }
        })
        .collect();
    let half_pi = std::f64::consts::FRAC_PI_2;
    while states.iter().any(|state| state.circle.is_none()) {
        // Flatten in label/hull order, exactly as centrosome's
        // keep_me_vertices mask; scipy.ndimage.minimum_position sorts this
        // entire vector (NumPy's unstable argsort) before grouping by label.
        let mut candidates = Vec::new();
        for (object, state) in states.iter().enumerate() {
            if state.circle.is_some() {
                continue;
            }
            let (a, b) = (state.hull[state.s0], state.hull[state.s1]);
            let ab2 = distance_squared(a, b);
            for (vertex, &v) in state.hull.iter().enumerate() {
                if state.ranks[vertex] < 2 {
                    continue;
                }
                let at_b = (((a.row - b.row) * (v.row - b.row)
                    + (a.col - b.col) * (v.col - b.col))
                    / (ab2 * distance_squared(v, b)).sqrt())
                .acos()
                .abs();
                let at_a = (((b.row - a.row) * (v.row - a.row)
                    + (b.col - a.col) * (v.col - a.col))
                    / (ab2 * distance_squared(v, a)).sqrt())
                .acos()
                .abs();
                candidates.push(CircleCandidate {
                    object,
                    vertex,
                    angle: std::f64::consts::PI - at_b - at_a,
                    at_a,
                    at_b,
                });
            }
        }
        let angles: Vec<_> = candidates.iter().map(|candidate| candidate.angle).collect();
        let mut selected = vec![None; states.len()];
        let sorted = crate::numpy_argsort(&angles);
        for index in sorted {
            let candidate = candidates[index];
            if selected[candidate.object].is_none() {
                selected[candidate.object] = Some(candidate);
            }
        }
        for (object, state) in states.iter_mut().enumerate() {
            if state.circle.is_some() {
                continue;
            }
            let (a, b) = (state.hull[state.s0], state.hull[state.s1]);
            let Some(candidate) = selected[object] else {
                state.circle = Some(diameter(a, b));
                continue;
            };
            let CircleCandidate {
                vertex,
                angle,
                at_a,
                at_b,
                ..
            } = candidate;
            if angle >= half_pi {
                state.circle = Some(diameter(a, b));
                continue;
            }
            let v = state.hull[vertex];
            if at_a <= half_pi && at_b <= half_pi && angle <= half_pi {
                let d = 2.0
                    * (a.col * (b.row - v.row) + b.col * (v.row - a.row) + v.col * (a.row - b.row));
                let aa = a.row * a.row + a.col * a.col;
                let bb = b.row * b.row + b.col * b.col;
                let vv = v.row * v.row + v.col * v.col;
                let center = Point {
                    col: (aa * (b.row - v.row) + bb * (v.row - a.row) + vv * (a.row - b.row)) / d,
                    row: (aa * (v.col - b.col) + bb * (a.col - v.col) + vv * (b.col - a.col)) / d,
                };
                state.circle = Some((center, distance_squared(a, center).sqrt()));
                continue;
            }
            if at_a > half_pi {
                state.ranks[state.s0] = state.ranks[vertex];
                state.ranks[vertex] = 0;
                state.s0 = vertex;
            } else {
                state.ranks[state.s1] = state.ranks[vertex];
                state.ranks[vertex] = 1;
                state.s1 = vertex;
            }
        }
    }
    states
        .into_iter()
        .filter_map(|state| state.circle.map(|circle| (state.label, circle)))
        .collect()
}

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, value| acc * value as f64)
}

// centrosome.zernike's radial Horner scheme, complex powers and area
// normalization. In particular there is no textbook (n+1) normalization factor.
fn zernike(points: &[Point], center: Point, radius: f64, n: usize, m: usize) -> f64 {
    let mut real = 0.0;
    let mut imag = 0.0;
    for p in points {
        let x = (p.col - center.col) / radius;
        let y = (p.row - center.row) / radius;
        let r2 = x * x + y * y;
        let mut radial = 0.0;
        for k in 0..=(n - m) / 2 {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            let coefficient = sign * factorial(n - k)
                / (factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k));
            radial = radial * r2 + coefficient;
        }
        if r2 > 1.0 {
            radial = 0.0;
        }
        let (mut re, mut im) = (1.0, 0.0);
        for _ in 0..m {
            (re, im) = (re * y - im * x, re * x + im * y);
        }
        real += radial * re;
        imag += radial * im;
    }
    (real * real + imag * imag).sqrt() / (radius * radius * std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured by running the actual CP 4.2.8 module with centrosome 1.2.3
    // and skimage 0.21.0. This asymmetric hull has a non-diameter circle,
    // nonzero signed Hu moments, and a boundary pixel rejected by r^2 > 1.
    #[test]
    fn asymmetric_shape_matches_all_83_oracle_columns() {
        let mut labels = vec![0; 90];
        for (row, col) in [(1, 1), (1, 4), (2, 2), (3, 1), (3, 4), (4, 2)] {
            labels[row * 10 + col] = 1;
        }
        let actual = measure(&labels, 9, 10, true, true).unwrap();
        let expected = [
            ("AreaShape_CentralMoment_0_0", 6.0),
            ("AreaShape_CentralMoment_0_1", 4.440892098500626e-16),
            ("AreaShape_CentralMoment_0_2", 9.333333333333334),
            ("AreaShape_CentralMoment_0_3", 4.444444444444447),
            ("AreaShape_CentralMoment_1_0", 4.440892098500626e-16),
            ("AreaShape_CentralMoment_1_1", -0.6666666666666667),
            ("AreaShape_CentralMoment_1_2", -2.8888888888888884),
            ("AreaShape_CentralMoment_1_3", -1.5555555555555562),
            ("AreaShape_CentralMoment_2_0", 7.333333333333334),
            ("AreaShape_CentralMoment_2_1", -0.22222222222222157),
            ("AreaShape_CentralMoment_2_2", 10.444444444444445),
            ("AreaShape_CentralMoment_2_3", 4.913580246913583),
            ("AreaShape_HuMoment_0", 0.462962962962963),
            ("AreaShape_HuMoment_1", 0.004458161865569276),
            ("AreaShape_HuMoment_2", 0.014034954021236605),
            ("AreaShape_HuMoment_3", 0.0030610171213737766),
            ("AreaShape_HuMoment_4", 1.7462776729911204e-05),
            ("AreaShape_HuMoment_5", 0.00018299303215220538),
            ("AreaShape_HuMoment_6", -9.878801221584306e-06),
            ("AreaShape_InertiaTensorEigenvalues_0", 1.5891972930813327),
            ("AreaShape_InertiaTensorEigenvalues_1", 1.1885804846964452),
            ("AreaShape_InertiaTensor_0_0", 1.5555555555555556),
            ("AreaShape_InertiaTensor_0_1", 0.11111111111111112),
            ("AreaShape_InertiaTensor_1_0", 0.11111111111111112),
            ("AreaShape_InertiaTensor_1_1", 1.2222222222222223),
            ("AreaShape_NormalizedMoment_0_0", f64::NAN),
            ("AreaShape_NormalizedMoment_0_1", f64::NAN),
            ("AreaShape_NormalizedMoment_0_2", 0.2592592592592593),
            ("AreaShape_NormalizedMoment_0_3", 0.050401023514057196),
            ("AreaShape_NormalizedMoment_1_0", f64::NAN),
            ("AreaShape_NormalizedMoment_1_1", -0.01851851851851852),
            ("AreaShape_NormalizedMoment_1_2", -0.03276066528413715),
            ("AreaShape_NormalizedMoment_1_3", -0.007201646090534982),
            ("AreaShape_NormalizedMoment_2_0", 0.20370370370370372),
            ("AreaShape_NormalizedMoment_2_1", -0.002520051175702851),
            ("AreaShape_NormalizedMoment_2_2", 0.04835390946502058),
            ("AreaShape_NormalizedMoment_2_3", 0.009286855258608687),
            ("AreaShape_NormalizedMoment_3_0", 0.005040102351405739),
            ("AreaShape_NormalizedMoment_3_1", -0.0102880658436214),
            ("AreaShape_NormalizedMoment_3_2", -0.01689367640008212),
            ("AreaShape_NormalizedMoment_3_3", -0.003746888177615202),
            ("AreaShape_SpatialMoment_0_0", 6.0),
            ("AreaShape_SpatialMoment_0_1", 8.0),
            ("AreaShape_SpatialMoment_0_2", 20.0),
            ("AreaShape_SpatialMoment_0_3", 56.0),
            ("AreaShape_SpatialMoment_1_0", 8.0),
            ("AreaShape_SpatialMoment_1_1", 10.0),
            ("AreaShape_SpatialMoment_1_2", 22.0),
            ("AreaShape_SpatialMoment_1_3", 58.0),
            ("AreaShape_SpatialMoment_2_0", 18.0),
            ("AreaShape_SpatialMoment_2_1", 22.0),
            ("AreaShape_SpatialMoment_2_2", 46.0),
            ("AreaShape_SpatialMoment_2_3", 118.0),
            ("AreaShape_Zernike_0_0", 0.44073676548524865),
            ("AreaShape_Zernike_1_1", 0.045079289050818545),
            ("AreaShape_Zernike_2_0", 0.21290976055748934),
            ("AreaShape_Zernike_2_2", 0.12482872145698055),
            ("AreaShape_Zernike_3_1", 0.07274264233585749),
            ("AreaShape_Zernike_3_3", 0.033438993995003774),
            ("AreaShape_Zernike_4_0", 0.24395431947072238),
            ("AreaShape_Zernike_4_2", 0.04987086970722689),
            ("AreaShape_Zernike_4_4", 0.2065679306598262),
            ("AreaShape_Zernike_5_1", 0.14405731457243298),
            ("AreaShape_Zernike_5_3", 0.14600773740919737),
            ("AreaShape_Zernike_5_5", 0.08886572333264058),
            ("AreaShape_Zernike_6_0", 0.09847628099022397),
            ("AreaShape_Zernike_6_2", 0.04648931677455873),
            ("AreaShape_Zernike_6_4", 0.15003413618028086),
            ("AreaShape_Zernike_6_6", 0.21478649601550334),
            ("AreaShape_Zernike_7_1", 0.1361726494626835),
            ("AreaShape_Zernike_7_3", 0.23852770162906936),
            ("AreaShape_Zernike_7_5", 0.011882816992200286),
            ("AreaShape_Zernike_7_7", 0.14139920276447504),
            ("AreaShape_Zernike_8_0", 0.09554426195028053),
            ("AreaShape_Zernike_8_2", 0.03797678814914931),
            ("AreaShape_Zernike_8_4", 0.12069733919596304),
            ("AreaShape_Zernike_8_6", 0.07422252145090656),
            ("AreaShape_Zernike_8_8", 0.04850680729483665),
            ("AreaShape_Zernike_9_1", 0.1547270658567515),
            ("AreaShape_Zernike_9_3", 0.21238656152789906),
            ("AreaShape_Zernike_9_5", 0.017551406619951887),
            ("AreaShape_Zernike_9_7", 0.1983771717591107),
            ("AreaShape_Zernike_9_9", 0.011833085764151785),
        ];
        assert_eq!(actual.len(), expected.len());
        for (name, reference) in expected {
            let value = actual[name][0];
            assert!(
                (value.is_nan() && reference.is_nan())
                    || (value - reference).abs() <= 1e-12 + 1e-11 * reference.abs(),
                "{name}: {value:?} != {reference:?}"
            );
        }
    }

    #[test]
    fn strict_circle_boundary_mask_matches_ties_and_diagonals() {
        // Oracle Zernike_0_0 values, not idealized geometric expectations.
        // For the square and right triangle all boundary pixels round outside!
        let cases: &[(&[(usize, usize)], f64)] = &[
            (&[(1, 1), (2, 2)], 1.2732395447351623),
            (&[(1, 1), (2, 2), (3, 3)], 0.4774648292756859),
            (&[(1, 1), (2, 2), (3, 3), (4, 4)], 0.14147106052612923),
            (&[(1, 1), (1, 4), (4, 1)], 0.0),
            (&[(1, 1), (1, 4), (4, 1), (4, 4)], 0.0),
        ];
        for (points, expected) in cases {
            let mut labels = [0; 60];
            for &(row, col) in *points {
                labels[row * 10 + col] = 1;
            }
            let actual = measure(&labels, 6, 10, true, false).unwrap();
            assert!((actual["AreaShape_Zernike_0_0"][0] - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn nonempty_flags_do_not_leak_other_columns() {
        let labels = [0, 1, 1, 0, 1, 0];
        let all = measure(&labels, 2, 3, true, true).unwrap();
        for (z, a, count) in [(false, false, 0), (true, false, 30), (false, true, 53)] {
            let selected = measure(&labels, 2, 3, z, a).unwrap();
            assert_eq!(selected.len(), count);
            for (name, values) in selected {
                assert_eq!(name.starts_with("AreaShape_Zernike_"), z);
                assert_eq!(values[0].to_bits(), all[&name][0].to_bits());
            }
        }
    }

    #[test]
    fn moments_use_local_bbox_coordinates_and_are_translation_invariant() {
        let mut a = [0; 100];
        let mut b = [0; 100];
        for (row, col) in [(0, 0), (0, 3), (1, 1), (2, 0), (2, 3), (3, 1)] {
            a[row * 10 + col] = 1;
            b[(row + 5) * 10 + col + 5] = 1;
        }
        let original = measure(&a, 10, 10, false, true).unwrap();
        let translated = measure(&b, 10, 10, false, true).unwrap();
        for (name, values) in original {
            assert_eq!(
                values[0].to_bits(),
                translated[&name][0].to_bits(),
                "{name}"
            );
        }
    }

    #[test]
    fn selected_columns_and_point_degeneracies() {
        let result = measure(&[0, 0, 0, 1], 2, 2, true, true).unwrap();
        assert_eq!(result.len(), 83);
        assert!(result["AreaShape_Zernike_0_0"][0].is_nan());
        assert!(result["AreaShape_NormalizedMoment_0_0"][0].is_nan());
        assert_eq!(result["AreaShape_SpatialMoment_0_0"], [1.0]);
        assert_eq!(result["AreaShape_SpatialMoment_0_1"], [0.0]);
        assert_eq!(result["AreaShape_HuMoment_0"], [0.0]);
        assert!(result["AreaShape_InertiaTensor_0_1"][0].is_sign_negative());
    }

    #[test]
    fn flags_control_exact_column_families_even_when_empty() {
        for (zernikes, advanced, count) in [
            (false, false, 0),
            (true, false, 30),
            (false, true, 53),
            (true, true, 83),
        ] {
            let result = measure(&[0; 6], 2, 3, zernikes, advanced).unwrap();
            assert_eq!(result.len(), count);
            assert!(result.values().all(Vec::is_empty));
        }
    }

    #[test]
    fn large_dense_moments_preserve_oracle_block_reduction_residuals() {
        // skimage 0.21.0 / NumPy OpenBLAS 0.3.20 ARMv8 oracle. Both bbox
        // dimensions cross the reference's 128-element dot-product block.
        let mut labels = vec![0; 153 * 179];
        for row in 0..153 {
            for col in 0..179 {
                labels[row * 179 + col] = u32::from((row * 37 + col * 43 + row * col) % 19 < 9);
            }
        }
        let actual = measure(&labels, 153, 179, false, true).unwrap();
        for (name, reference) in [
            ("AreaShape_CentralMoment_0_1", -2.9103830456733704e-10),
            ("AreaShape_CentralMoment_1_0", 5.479705578181893e-10),
        ] {
            assert!(
                (actual[name][0] - reference).abs() < 1e-12,
                "{name}: {} != {reference}",
                actual[name][0]
            );
        }
    }

    #[test]
    fn malformed_shapes_are_rejected() {
        assert!(measure(&[1], 2, 2, true, true).is_err());
        assert!(measure(&[], usize::MAX, 2, true, true).is_err());
        assert!(measure(&[], 0, 2, false, false).is_err());
    }
}

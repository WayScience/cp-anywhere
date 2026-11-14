//! CellProfiler 4.2.8 compatible measurements over borrowed, contiguous arrays.
//! The core has no Python, filesystem, or platform dependency.

use std::collections::{BTreeMap, BTreeSet};

pub const MAX_LABEL: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, Default)]
pub struct Families {
    pub image_intensity: bool,
    pub object_intensity: bool,
    pub neighbors: bool,
}

impl Families {
    pub const ALL: Self = Self {
        image_intensity: true,
        object_intensity: true,
        neighbors: true,
    };
}

#[derive(Debug)]
pub struct Input<'a> {
    pub shape: &'a [usize],
    pub images: &'a [(&'a str, &'a [f64])],
    pub objects: &'a [(&'a str, &'a [u32])],
    pub mask: Option<&'a [bool]>,
    pub families: Families,
}

#[derive(Debug)]
pub enum Column {
    Float(Vec<f64>),
    Int(Vec<i64>),
}

#[derive(Debug)]
pub struct ResultTable {
    pub features: BTreeMap<String, BTreeMap<String, Column>>,
    pub object_ids: BTreeMap<String, Vec<i64>>,
    pub relationships: Vec<(i64, i64)>,
}

fn put_float(table: &mut ResultTable, scope: &str, name: String, values: Vec<f64>) {
    table
        .features
        .entry(scope.to_owned())
        .or_default()
        .insert(name, Column::Float(values));
}
fn put_int(table: &mut ResultTable, scope: &str, name: String, values: Vec<i64>) {
    table
        .features
        .entry(scope.to_owned())
        .or_default()
        .insert(name, Column::Int(values));
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (sorted.len() - 1) as f64 * fraction;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] * (1.0 - pos.fract()) + sorted[hi] * pos.fract()
}
fn object_quantile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = sorted.len() as f64 * fraction;
    let lo = (pos.floor() as usize).min(sorted.len() - 1);
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] * (1.0 - pos.fract()) + sorted[hi] * pos.fract()
}
fn numpy_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

// Derived from NumPy 1.23.5 npysort/quicksort.cpp aquicksort_ and
// npysort_heapsort.h aheapsort_; see licenses/numpy.txt.
fn numpy_argsort(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    if order.len() > 1 {
        let depth = (usize::BITS - order.len().leading_zeros()) as i32 * 2;
        argsort_range(values, &mut order, 0, values.len() - 1, depth);
    }
    order
}

fn argsort_range(
    values: &[f64],
    order: &mut [usize],
    mut lo: usize,
    mut hi: usize,
    mut depth: i32,
) {
    while hi - lo > 15 {
        if depth < 0 {
            numpy_heapsort(values, &mut order[lo..=hi]);
            return;
        }
        let mid = lo + ((hi - lo) >> 1);
        if numpy_cmp(&values[order[mid]], &values[order[lo]]).is_lt() {
            order.swap(mid, lo);
        }
        if numpy_cmp(&values[order[hi]], &values[order[mid]]).is_lt() {
            order.swap(hi, mid);
        }
        if numpy_cmp(&values[order[mid]], &values[order[lo]]).is_lt() {
            order.swap(mid, lo);
        }
        let pivot = values[order[mid]];
        order.swap(mid, hi - 1);
        let mut i = lo;
        let mut j = hi - 1;
        loop {
            loop {
                i += 1;
                if !numpy_cmp(&values[order[i]], &pivot).is_lt() {
                    break;
                }
            }
            loop {
                j -= 1;
                if !numpy_cmp(&pivot, &values[order[j]]).is_lt() {
                    break;
                }
            }
            if i >= j {
                break;
            }
            order.swap(i, j);
        }
        order.swap(i, hi - 1);
        depth -= 1;
        if i - lo < hi - i {
            if i > lo {
                argsort_range(values, order, lo, i - 1, depth);
            }
            lo = i + 1;
        } else {
            if i < hi {
                argsort_range(values, order, i + 1, hi, depth);
            }
            if i == 0 {
                return;
            }
            hi = i - 1;
        }
        if lo >= hi {
            return;
        }
    }
    for i in lo + 1..=hi {
        let value = order[i];
        let mut j = i;
        while j > lo && numpy_cmp(&values[value], &values[order[j - 1]]).is_lt() {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = value;
    }
}

fn numpy_heapsort(values: &[f64], order: &mut [usize]) {
    let mut n = order.len();
    for start in (1..=n / 2).rev() {
        heap_sift(values, order, start, n, order[start - 1]);
    }
    while n > 1 {
        let last = order[n - 1];
        order[n - 1] = order[0];
        n -= 1;
        heap_sift(values, order, 1, n, last);
    }
}
fn heap_sift(values: &[f64], order: &mut [usize], mut i: usize, n: usize, value: usize) {
    let mut j = i * 2;
    while j <= n {
        if j < n && numpy_cmp(&values[order[j - 1]], &values[order[j]]).is_lt() {
            j += 1;
        }
        if numpy_cmp(&values[value], &values[order[j - 1]]).is_lt() {
            order[i - 1] = order[j - 1];
            i = j;
            j *= 2;
        } else {
            break;
        }
    }
    order[i - 1] = value;
}
fn coords(index: usize, shape: &[usize]) -> [usize; 3] {
    if shape.len() == 2 {
        [0, index / shape[1], index % shape[1]]
    } else {
        [
            index / (shape[1] * shape[2]),
            (index / shape[2]) % shape[1],
            index % shape[2],
        ]
    }
}
fn index(c: &[usize; 3], shape: &[usize]) -> usize {
    if shape.len() == 2 {
        c[1] * shape[1] + c[2]
    } else {
        (c[0] * shape[1] + c[1]) * shape[2] + c[2]
    }
}
fn in_bounds(c: &[isize; 3], shape: &[usize]) -> bool {
    if shape.len() == 2 {
        c[1] >= 0 && c[1] < shape[0] as isize && c[2] >= 0 && c[2] < shape[1] as isize
    } else {
        c.iter().zip(shape).all(|(&v, &s)| v >= 0 && v < s as isize)
    }
}
fn shifted(i: usize, shape: &[usize], delta: [isize; 3]) -> Option<usize> {
    let c = coords(i, shape);
    let next = [
        c[0] as isize + delta[0],
        c[1] as isize + delta[1],
        c[2] as isize + delta[2],
    ];
    if !in_bounds(&next, shape) {
        return None;
    }
    Some(index(
        &[next[0] as usize, next[1] as usize, next[2] as usize],
        shape,
    ))
}

pub fn measure(input: &Input<'_>) -> Result<ResultTable, String> {
    if !(2..=3).contains(&input.shape.len()) || input.shape.contains(&0) {
        return Err("shape must have two or three nonzero dimensions".into());
    }
    let size = input
        .shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or("shape product overflow")?;
    if input.images.is_empty() || input.objects.is_empty() {
        return Err("images and objects must be nonempty".into());
    }
    if input.mask.is_some_and(|m| m.len() != size) {
        return Err("mask length does not match shape".into());
    }
    let mut names = BTreeSet::new();
    for (name, pixels) in input.images {
        if name.is_empty() || !names.insert(*name) || pixels.len() != size {
            return Err("invalid image name or buffer length".into());
        }
    }
    names.clear();
    for (name, labels) in input.objects {
        if name.is_empty() || !names.insert(*name) || labels.len() != size {
            return Err("invalid object name or buffer length".into());
        }
        if labels.iter().any(|&id| id > MAX_LABEL) {
            return Err(format!("label exceeds {MAX_LABEL}"));
        }
    }
    let mut result = ResultTable {
        features: BTreeMap::new(),
        object_ids: BTreeMap::new(),
        relationships: Vec::new(),
    };
    for (scope, labels) in input.objects {
        let max = labels.iter().copied().max().unwrap_or(0) as usize;
        result
            .object_ids
            .insert((*scope).into(), (1..=max as i64).collect());
    }
    if input.families.image_intensity {
        for (name, pixels) in input.images {
            image_intensity(&mut result, name, pixels, input.mask);
        }
    }
    if input.families.object_intensity {
        for (scope, labels) in input.objects {
            for (name, pixels) in input.images {
                object_intensity(
                    &mut result,
                    scope,
                    name,
                    pixels,
                    labels,
                    input.mask,
                    input.shape,
                );
            }
        }
    }
    if input.families.neighbors {
        let labels = input
            .objects
            .iter()
            .find(|(name, _)| *name == "Cells")
            .ok_or("neighbors require Cells object set")?
            .1;
        neighbors(&mut result, labels, input.shape);
    }
    Ok(result)
}

fn image_intensity(result: &mut ResultTable, channel: &str, pixels: &[f64], mask: Option<&[bool]>) {
    let mut values: Vec<f64> = pixels
        .iter()
        .enumerate()
        .filter(|(i, v)| mask.is_none_or(|m| m[*i]) && v.is_finite())
        .map(|(_, v)| *v)
        .collect();
    let n = values.len();
    values.sort_by(numpy_cmp);
    let sum: f64 = values.iter().sum();
    let mean = if n == 0 { 0.0 } else { sum / n as f64 };
    let std = if n == 0 {
        0.0
    } else {
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    let median = percentile(&values, 0.5);
    let mut dev: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
    dev.sort_by(numpy_cmp);
    let mad = percentile(&dev, 0.5);
    let min = values.first().copied().unwrap_or(0.0);
    let max = values.last().copied().unwrap_or(0.0);
    let pct_max = if n == 0 {
        0.0
    } else {
        100.0 * values.iter().filter(|&&v| v == max).count() as f64 / n as f64
    };
    let features = [
        ("TotalIntensity", sum),
        ("MeanIntensity", mean),
        ("MedianIntensity", median),
        ("StdIntensity", std),
        ("MADIntensity", mad),
        ("MaxIntensity", max),
        ("MinIntensity", min),
        ("TotalArea", n as f64),
        ("PercentMaximal", pct_max),
        ("LowerQuartileIntensity", percentile(&values, 0.25)),
        ("UpperQuartileIntensity", percentile(&values, 0.75)),
        ("Percentile_10", percentile(&values, 0.1)),
        ("Percentile_50", percentile(&values, 0.5)),
        ("Percentile_90", percentile(&values, 0.9)),
    ];
    for (feature, value) in features {
        if feature == "TotalArea" {
            put_int(
                result,
                "Image",
                format!("Intensity_{feature}_{channel}"),
                vec![n as i64],
            );
        } else {
            put_float(
                result,
                "Image",
                format!("Intensity_{feature}_{channel}"),
                vec![value],
            );
        }
    }
}

fn object_intensity(
    result: &mut ResultTable,
    scope: &str,
    channel: &str,
    pixels: &[f64],
    labels: &[u32],
    mask: Option<&[bool]>,
    shape: &[usize],
) {
    let max = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut values: Vec<Vec<(f64, usize)>> = vec![Vec::new(); max];
    let mut edges: Vec<Vec<f64>> = vec![Vec::new(); max];
    for (i, &id) in labels.iter().enumerate() {
        if id == 0 || !mask.is_none_or(|m| m[i]) {
            continue;
        }
        let v = pixels[i];
        values[id as usize - 1].push((v, i));
        // find_boundaries(mode="inner") after a 2D-to-depth-one reshape.
        let mut boundary = false;
        for axis in 0..shape.len() {
            let k = if shape.len() == 2 { axis + 1 } else { axis };
            for d in [-1, 1] {
                let mut delta = [0; 3];
                delta[k] = d;
                if let Some(j) = shifted(i, shape, delta) {
                    if labels[j] != id {
                        boundary = true;
                    }
                }
            }
        }
        if boundary {
            edges[id as usize - 1].push(v);
        }
    }
    let selected: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(i, &id)| id > 0 && mask.is_none_or(|m| m[*i]))
        .map(|(i, _)| i)
        .collect();
    let selected_values: Vec<f64> = selected.iter().map(|&i| pixels[i]).collect();
    let mut max_position = vec![0usize; max];
    for offset in numpy_argsort(&selected_values) {
        let i = selected[offset];
        max_position[labels[i] as usize - 1] = i;
    }
    const NAMES: [&str; 21] = [
        "IntegratedIntensity",
        "MeanIntensity",
        "StdIntensity",
        "MinIntensity",
        "MaxIntensity",
        "IntegratedIntensityEdge",
        "MeanIntensityEdge",
        "StdIntensityEdge",
        "MinIntensityEdge",
        "MaxIntensityEdge",
        "MassDisplacement",
        "LowerQuartileIntensity",
        "MedianIntensity",
        "MADIntensity",
        "UpperQuartileIntensity",
        "CenterMassIntensity_X",
        "CenterMassIntensity_Y",
        "CenterMassIntensity_Z",
        "MaxIntensity_X",
        "MaxIntensity_Y",
        "MaxIntensity_Z",
    ];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; max]; 21];
    for id in 0..max {
        let v = &values[id];
        let e = &edges[id];
        if v.is_empty() {
            continue;
        }
        let n = v.len() as f64;
        let sum: f64 = v.iter().map(|x| x.0).sum();
        let mean = sum / n;
        let std = (v.iter().map(|x| (x.0 - mean).powi(2)).sum::<f64>() / n).sqrt();
        let mut sorted: Vec<f64> = v.iter().map(|x| x.0).collect();
        sorted.sort_by(numpy_cmp);
        out[0][id] = sum;
        out[1][id] = mean;
        out[2][id] = std;
        out[3][id] = sorted[0];
        out[4][id] = sorted[sorted.len() - 1];
        for (q, frac) in [(11, 0.25), (12, 0.5), (14, 0.75)] {
            out[q][id] = object_quantile(&sorted, frac);
        }
        let mut dev: Vec<f64> = v.iter().map(|x| (x.0 - out[12][id]).abs()).collect();
        dev.sort_by(numpy_cmp);
        out[13][id] = object_quantile(&dev, 1.0 / shape.len() as f64);
        let mut coord_sum = [0.0; 3];
        let mut weighted = [0.0; 3];
        for &(intensity, i) in v {
            let c = coords(i, shape);
            for k in 0..3 {
                coord_sum[k] += c[2 - k] as f64;
                weighted[k] += c[2 - k] as f64 * intensity;
            }
        }
        let mut disp = 0.0;
        for k in 0..3 {
            let cm = weighted[k] / sum;
            out[15 + k][id] = cm;
            disp += (coord_sum[k] / n - cm).powi(2);
        }
        out[10][id] = disp.sqrt();
        let best = max_position[id];
        let c = coords(best, shape);
        for k in 0..3 {
            out[18 + k][id] = c[2 - k] as f64;
        }
        if !e.is_empty() {
            let ne = e.len() as f64;
            let se: f64 = e.iter().sum();
            let me = se / ne;
            out[5][id] = se;
            out[6][id] = me;
            out[7][id] = (e.iter().map(|x| (x - me).powi(2)).sum::<f64>() / ne).sqrt();
            out[8][id] = e.iter().copied().min_by(numpy_cmp).unwrap();
            out[9][id] = e.iter().copied().max_by(numpy_cmp).unwrap();
        } else if !edges.iter().all(Vec::is_empty) {
            for row in &mut out[5..10] {
                row[id] = f64::NAN;
            }
        }
    }
    for (k, name) in NAMES.iter().enumerate() {
        let prefix = if k >= 15 { "Location" } else { "Intensity" };
        put_float(
            result,
            scope,
            format!("{prefix}_{name}_{channel}"),
            out[k].clone(),
        );
    }
}

fn neighbors(result: &mut ResultTable, labels: &[u32], shape: &[usize]) {
    let max = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut count = vec![0_i64; max];
    let mut first = vec![0_i64; max];
    let mut second = vec![0_i64; max];
    let mut d1 = vec![0.0; max];
    let mut d2 = vec![0.0; max];
    let mut angle = vec![0.0; max];
    let mut touching = vec![0.0; max];
    let mut centers = vec![[0.0; 3]; max];
    let mut areas = vec![0usize; max];
    for (i, &id) in labels.iter().enumerate() {
        if id > 0 {
            let k = id as usize - 1;
            areas[k] += 1;
            let c = coords(i, shape);
            for axis in 0..3 {
                centers[k][axis] += c[axis] as f64;
            }
        }
    }
    for (center, &area) in centers[..max].iter_mut().zip(&areas[..max]) {
        let divisor = area as f64;
        for axis in center.iter_mut() {
            *axis /= divisor;
        }
    }
    let ax = if shape.len() == 2 { 1 } else { 0 };
    let ay = ax + 1;
    if max > 1 {
        for k in 0..max {
            if areas[k] == 0 {
                continue;
            }
            let mut order: Vec<usize> = (0..max).collect();
            order.sort_by(|&a, &b| {
                let da = (centers[k][ax] - centers[a][ax]).powi(2)
                    + (centers[k][ay] - centers[a][ay]).powi(2);
                let db = (centers[k][ax] - centers[b][ax]).powi(2)
                    + (centers[k][ay] - centers[b][ay]).powi(2);
                numpy_cmp(&da.sqrt(), &db.sqrt()).then(a.cmp(&b))
            });
            first[k] = (order[1] + 1) as i64;
            if max > 2 {
                second[k] = (order[2] + 1) as i64;
            }
            let a = order[1];
            let dx = centers[k][ax] - centers[a][ax];
            let dy = centers[k][ay] - centers[a][ay];
            d1[k] = (dx * dx + dy * dy).sqrt();
            if max > 2 {
                let b = order[2];
                let ex = centers[k][ax] - centers[b][ax];
                let ey = centers[k][ay] - centers[b][ay];
                d2[k] = (ex * ex + ey * ey).sqrt();
                angle[k] = ((dx * ex + dy * ey)
                    / ((dx * dx + dy * dy) * (ex * ex + ey * ey)).sqrt())
                .acos()
                    * 180.0
                    / std::f64::consts::PI;
            }
        }
        let outline = neighbor_outline(labels, shape);
        let mut perimeters = vec![0usize; max];
        let mut overlap = vec![0usize; max];
        for (i, &id) in labels.iter().enumerate() {
            if id == 0 {
                continue;
            }
            let k = id as usize - 1;
            if outline[i] {
                perimeters[k] += 1;
                let mut touching_other = false;
                // Bounded local-neighborhood check, equivalent to the
                // exhaustive "scan every pixel and test closeness" version
                // this replaces, but O(1) per outline pixel instead of
                // O(total pixels): the `close` predicate below only ever
                // depends on a fixed, small set of relative offsets.
                let candidate_deltas: &[[isize; 3]] = if shape.len() == 2 {
                    &[
                        [0, -1, -1],
                        [0, -1, 0],
                        [0, -1, 1],
                        [0, 0, -1],
                        [0, 0, 1],
                        [0, 1, -1],
                        [0, 1, 0],
                        [0, 1, 1],
                    ]
                } else {
                    &[
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1],
                    ]
                };
                for &delta in candidate_deltas {
                    if let Some(j) = shifted(i, shape, delta) {
                        let other = labels[j];
                        if other != 0 && other != id {
                            touching_other = true;
                            break;
                        }
                    }
                }
                if touching_other {
                    overlap[k] += 1;
                }
            }
            for delta in if shape.len() == 2 {
                vec![[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            } else {
                vec![
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                ]
            } {
                if let Some(j) = shifted(i, shape, delta) {
                    let other = labels[j];
                    if other > 0 && other != id {
                        let edge = (id as i64, other as i64);
                        if !result.relationships.contains(&edge) {
                            result.relationships.push(edge);
                        }
                    }
                }
            }
        }
        for k in 0..max {
            touching[k] = 100.0 * overlap[k] as f64 / perimeters[k] as f64;
        }
        if let Some(first_present) = areas.iter().position(|&area| area > 0) {
            for k in 0..max {
                if areas[k] == 0 {
                    d1[k] = d1[first_present];
                    d2[k] = d2[first_present];
                    angle[k] = angle[first_present];
                    touching[k] = 0.0;
                }
            }
        }
    }
    result.relationships.sort_unstable();
    for &(a, _) in &result.relationships {
        count[a as usize - 1] += 1;
    }
    put_int(
        result,
        "Cells",
        "Neighbors_NumberOfNeighbors_Adjacent".into(),
        count,
    );
    put_int(
        result,
        "Cells",
        "Neighbors_FirstClosestObjectNumber_Adjacent".into(),
        first,
    );
    put_int(
        result,
        "Cells",
        "Neighbors_SecondClosestObjectNumber_Adjacent".into(),
        second,
    );
    for (name, values) in [
        ("FirstClosestDistance", d1),
        ("SecondClosestDistance", d2),
        ("AngleBetweenNeighbors", angle),
        ("PercentTouching", touching),
    ] {
        put_float(
            result,
            "Cells",
            format!("Neighbors_{name}_Adjacent"),
            values,
        );
    }
}

fn neighbor_outline(labels: &[u32], shape: &[usize]) -> Vec<bool> {
    let mut out = vec![false; labels.len()];
    for (i, &id) in labels.iter().enumerate() {
        if id == 0 {
            continue;
        }
        let c = coords(i, shape);
        let (a, b, na, nb) = if shape.len() == 2 {
            (c[1], c[2], shape[0], shape[1])
        } else {
            (c[0], c[1], shape[0], shape[1])
        };
        if a == 0 || b == 0 || a + 1 == na || b + 1 == nb {
            out[i] = true;
            continue;
        }
        for da in -1..=1 {
            for db in -1..=1 {
                if da == 0 && db == 0 {
                    continue;
                }
                let delta = if shape.len() == 2 {
                    [0, da, db]
                } else {
                    [da, db, 0]
                };
                if shifted(i, shape, delta).is_some_and(|j| labels[j] != id) {
                    out[i] = true;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_shape_and_large_id() {
        let images = [("DNA", &[1.0][..])];
        let objects = [("Cells", &[1][..])];
        assert!(measure(&Input {
            shape: &[1],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL
        })
        .is_err());
        let bad = [MAX_LABEL + 1];
        let objects = [("Cells", &bad[..])];
        assert!(measure(&Input {
            shape: &[1, 1],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL
        })
        .is_err());
    }
}

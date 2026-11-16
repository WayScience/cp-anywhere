//! CellProfiler 4.2.8 compatible measurements over borrowed, contiguous arrays.
//! The core has no Python, filesystem, or platform dependency.

use std::collections::{BTreeMap, BTreeSet};

pub const MAX_LABEL: u32 = 1_000_000;

pub mod colocalization;
pub mod granularity;
pub mod intensity_distribution;
pub mod size_shape;
pub mod size_shape_moments;
pub mod texture;

/// Optional MeasureObjectSizeShape settings. Selecting this module is explicit
/// so existing default measurement and benchmark scopes remain unchanged.
#[derive(Clone, Copy, Debug)]
pub struct SizeShapeOptions {
    pub calculate_zernikes: bool,
    pub calculate_advanced: bool,
}

impl Default for SizeShapeOptions {
    fn default() -> Self {
        Self {
            calculate_zernikes: true,
            calculate_advanced: false,
        }
    }
}

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
    /// None leaves shape measurements disabled; Some selects the 2D module.
    pub size_shape: Option<SizeShapeOptions>,
    /// Object scopes to compute Neighbors for. Empty (the default for
    /// every pre-existing caller) preserves `families.neighbors`'s own
    /// legacy "Cells"-only behavior exactly -- same column scope name,
    /// same 1-object-set requirement, same relationship shape -- so no
    /// existing fixture or golden comparison changes. A non-empty list
    /// *replaces* that legacy default: Neighbors is computed once per
    /// named scope here and nowhere else (the caller must include
    /// "Cells" explicitly to still get it), letting a caller (e.g. a
    /// translated CellProfiler pipeline) request Neighbors for
    /// `Nuclei`, `Cells`, or any other provided object set by name,
    /// matching real CellProfiler's own per-object-set
    /// `MeasureObjectNeighbors` invocations.
    pub neighbors_scopes: &'a [&'a str],
    /// Optional diagnostic geometry, corresponding to CellProfiler's
    /// `small_removed_segmented` layer. Its contents depend on the
    /// upstream module: primary segmentation already excluded border
    /// objects; secondary segmentation retains pre-filter grown objects.
    /// Diagnostic and retained IDs are matched by overlap. Excluded
    /// diagnostic objects affect geometry but get no rows or edges.
    /// A scope without an entry uses its own retained labels.
    pub neighbors_full_objects: &'a [(&'a str, &'a [u32])],
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
    /// (object scope, first object_id, second object_id) directed edges.
    pub relationships: Vec<(String, i64, i64)>,
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
    if !input.neighbors_scopes.is_empty() {
        let mut seen = BTreeSet::new();
        for &scope in input.neighbors_scopes {
            if !seen.insert(scope) {
                continue;
            }
            let labels = input
                .objects
                .iter()
                .find(|(name, _)| *name == scope)
                .ok_or_else(|| {
                    format!("neighbors requested for {scope:?}, which is not a provided object set")
                })?
                .1;
            let full_labels = input
                .neighbors_full_objects
                .iter()
                .find(|(name, _)| *name == scope)
                .map(|(_, arr)| *arr);
            neighbors(&mut result, scope, labels, full_labels, input.shape);
        }
    } else if input.families.neighbors {
        let labels = input
            .objects
            .iter()
            .find(|(name, _)| *name == "Cells")
            .ok_or("neighbors require Cells object set")?
            .1;
        let full_labels = input
            .neighbors_full_objects
            .iter()
            .find(|(name, _)| *name == "Cells")
            .map(|(_, arr)| *arr);
        neighbors(&mut result, "Cells", labels, full_labels, input.shape);
    }
    if let Some(options) = input.size_shape {
        if input.shape.len() != 2 {
            return Err("MeasureObjectSizeShape currently supports 2D labels only".into());
        }
        for (scope, labels) in input.objects {
            let ids: BTreeSet<_> = labels.iter().copied().filter(|&id| id != 0).collect();
            if ids.iter().copied().ne(1..=ids.len() as u32) {
                return Err(format!(
                    "MeasureObjectSizeShape on {scope:?} requires consecutive label IDs starting at 1; \
                     reference sparse-label columns have inconsistent alignment"
                ));
            }
            let mut columns = size_shape::measure(labels, input.shape[0], input.shape[1])?;
            columns.extend(size_shape_moments::measure(
                labels,
                input.shape[0],
                input.shape[1],
                options.calculate_zernikes,
                options.calculate_advanced,
            )?);
            for (name, values) in columns {
                put_float(&mut result, scope, name, values);
            }
        }
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

fn neighbors(
    result: &mut ResultTable,
    scope: &str,
    labels: &[u32],
    full_labels: Option<&[u32]>,
    shape: &[usize],
) {
    // Geometry uses CP's small-removed diagnostic layer when supplied.
    // Kept labels govern rows and relationships; their IDs may differ
    // from diagnostic IDs after filtering. The overlap mapping below
    // preserves that distinction.
    let geometry_labels = full_labels.unwrap_or(labels);
    let full_max = geometry_labels.iter().copied().max().unwrap_or(0) as usize;
    let kept_max = labels.iter().copied().max().unwrap_or(0) as usize;
    // Safety margin in case a caller's "full" array is inconsistent
    // with its kept array (a kept id beyond what the full array has) --
    // every working vector is sized to cover both, so indexing by any
    // valid kept or full id can never go out of bounds.
    let calc_max = full_max.max(kept_max);

    let mut count = vec![0_i64; calc_max];
    let mut first = vec![0_i64; calc_max];
    let mut second = vec![0_i64; calc_max];
    let mut d1 = vec![0.0; calc_max];
    let mut d2 = vec![0.0; calc_max];
    let mut angle = vec![0.0; calc_max];
    let mut touching = vec![0.0; calc_max];
    let mut centers = vec![[0.0; 3]; calc_max];
    let mut areas = vec![0usize; calc_max];
    let mut kept_areas = vec![0usize; calc_max];
    let mut full_edges: Vec<(i64, i64)> = Vec::new();
    for (i, &id) in geometry_labels.iter().enumerate() {
        if id > 0 {
            let k = id as usize - 1;
            areas[k] += 1;
            let c = coords(i, shape);
            for axis in 0..3 {
                centers[k][axis] += c[axis] as f64;
            }
        }
    }
    for &id in labels {
        if id > 0 {
            kept_areas[id as usize - 1] += 1;
        }
    }
    // Final IDs may have been renumbered after filtering. Match each
    // kept object to its dominant overlapping diagnostic label (CP's
    // Objects.relate_labels), rather than assuming identical integers.
    let mut overlap_counts = BTreeMap::new();
    for (&kept, &full) in labels.iter().zip(geometry_labels) {
        if kept != 0 && full != 0 {
            *overlap_counts
                .entry((kept as usize - 1, full as usize - 1))
                .or_insert(0usize) += 1;
        }
    }
    let mut kept_to_full = vec![None; kept_max];
    let mut largest_overlap = vec![0; kept_max];
    for ((kept, full), overlap) in overlap_counts {
        if overlap > largest_overlap[kept] {
            kept_to_full[kept] = Some(full);
            largest_overlap[kept] = overlap;
        }
    }
    let mut full_to_kept = vec![0_i64; calc_max];
    for (kept, full) in kept_to_full.iter().enumerate() {
        if let Some(full) = full {
            full_to_kept[*full] = kept as i64 + 1;
        }
    }
    for (center, &area) in centers[..full_max].iter_mut().zip(&areas[..full_max]) {
        let divisor = area as f64;
        for axis in center.iter_mut() {
            *axis /= divisor;
        }
    }
    let ax = if shape.len() == 2 { 1 } else { 0 };
    let ay = ax + 1;
    if full_max > 1 {
        for k in 0..full_max {
            if areas[k] == 0 {
                continue;
            }
            // Distances and angles use the full geometry. CP separately
            // recomputes only closest-object NUMBERS among retained objects.
            let mut order: Vec<usize> = (0..full_max).filter(|&c| areas[c] > 0).collect();
            if order.len() < 2 {
                continue;
            }
            order.sort_by(|&a, &b| {
                let da = (centers[k][ax] - centers[a][ax]).powi(2)
                    + (centers[k][ay] - centers[a][ay]).powi(2);
                let db = (centers[k][ax] - centers[b][ax]).powi(2)
                    + (centers[k][ay] - centers[b][ay]).powi(2);
                numpy_cmp(&da.sqrt(), &db.sqrt()).then(a.cmp(&b))
            });
            first[k] = (order[1] + 1) as i64;
            if order.len() > 2 {
                second[k] = (order[2] + 1) as i64;
            }
            let a = order[1];
            let dx = centers[k][ax] - centers[a][ax];
            let dy = centers[k][ay] - centers[a][ay];
            d1[k] = (dx * dx + dy * dy).sqrt();
            if order.len() > 2 {
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
        let outline = neighbor_outline(geometry_labels, shape);
        let mut perimeters = vec![0usize; full_max];
        let mut overlap = vec![0usize; full_max];
        for (i, &id) in geometry_labels.iter().enumerate() {
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
                        let other = geometry_labels[j];
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
                    let other = geometry_labels[j];
                    if other > 0 && other != id {
                        let edge = (id as i64, other as i64);
                        if !full_edges.contains(&edge) {
                            full_edges.push(edge);
                        }
                    }
                }
            }
        }
        for k in 0..full_max {
            touching[k] = 100.0 * overlap[k] as f64 / perimeters[k] as f64;
        }
    }
    full_edges.sort_unstable();
    for &(a, _) in &full_edges {
        count[a as usize - 1] += 1;
    }
    let kept_edges: Vec<(i64, i64)> = full_edges
        .into_iter()
        .map(|(a, b)| (full_to_kept[a as usize - 1], full_to_kept[b as usize - 1]))
        .filter(|&(a, b)| a != 0 && b != 0)
        .collect();
    result.relationships.extend(
        kept_edges
            .into_iter()
            .map(|(a, b)| (scope.to_owned(), a, b)),
    );

    // Project raw-geometry measurements into the final retained ID space.
    let project_int = |values: &[i64]| {
        kept_to_full
            .iter()
            .map(|id| id.map_or(0, |i| values[i]))
            .collect::<Vec<_>>()
    };
    let project_float = |values: &[f64]| {
        kept_to_full
            .iter()
            .map(|id| id.map_or(0.0, |i| values[i]))
            .collect::<Vec<_>>()
    };
    count = project_int(&count);
    d1 = project_float(&d1);
    d2 = project_float(&d2);
    angle = project_float(&angle);
    touching = project_float(&touching);
    first = vec![0; kept_max];
    second = vec![0; kept_max];
    for (kept, &full) in kept_to_full.iter().enumerate() {
        let Some(full) = full else { continue };
        let mut order: Vec<_> = kept_to_full
            .iter()
            .enumerate()
            .filter_map(|(id, source)| source.map(|source| (id, source)))
            .collect();
        order.sort_by(|&(a, ga), &(b, gb)| {
            let squared_distance = |g: usize| {
                (centers[full][ax] - centers[g][ax]).powi(2)
                    + (centers[full][ay] - centers[g][ay]).powi(2)
            };
            numpy_cmp(&squared_distance(ga).sqrt(), &squared_distance(gb).sqrt()).then(a.cmp(&b))
        });
        if order.len() > 1 {
            first[kept] = order[1].0 as i64 + 1;
        }
        if order.len() > 2 {
            second[kept] = order[2].0 as i64 + 1;
        }
    }
    if let Some(first_present) = kept_areas[..kept_max].iter().position(|&area| area > 0) {
        for k in 0..kept_max {
            if kept_areas[k] == 0 {
                d1[k] = d1[first_present];
                d2[k] = d2[first_present];
                angle[k] = angle[first_present];
                touching[k] = 0.0;
            }
        }
    }

    put_int(
        result,
        scope,
        "Neighbors_NumberOfNeighbors_Adjacent".into(),
        count,
    );
    put_int(
        result,
        scope,
        "Neighbors_FirstClosestObjectNumber_Adjacent".into(),
        first,
    );
    put_int(
        result,
        scope,
        "Neighbors_SecondClosestObjectNumber_Adjacent".into(),
        second,
    );
    for (name, values) in [
        ("FirstClosestDistance", d1),
        ("SecondClosestDistance", d2),
        ("AngleBetweenNeighbors", angle),
        ("PercentTouching", touching),
    ] {
        put_float(result, scope, format!("Neighbors_{name}_Adjacent"), values);
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
            families: Families::ALL,
            neighbors_scopes: &[],
            size_shape: None,
            neighbors_full_objects: &[]
        })
        .is_err());
        let bad = [MAX_LABEL + 1];
        let objects = [("Cells", &bad[..])];
        assert!(measure(&Input {
            shape: &[1, 1],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL,
            neighbors_scopes: &[],
            size_shape: None,
            neighbors_full_objects: &[]
        })
        .is_err());
    }

    fn two_touching_objects_labels() -> [u32; 6] {
        // 1x6 row: [1,1,1,2,2,2] -- two objects, touching at the middle.
        [1, 1, 1, 2, 2, 2]
    }

    #[test]
    fn neighbors_defaults_to_cells_only_when_no_scopes_requested() {
        let images = [("DNA", &[1.0; 6][..])];
        let labels = two_touching_objects_labels();
        let objects = [("Cells", &labels[..])];
        let result = measure(&Input {
            shape: &[1, 6],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL,
            neighbors_scopes: &[],
            size_shape: None,
            neighbors_full_objects: &[],
        })
        .unwrap();
        assert!(result.features["Cells"].contains_key("Neighbors_NumberOfNeighbors_Adjacent"));
        assert_eq!(result.relationships.len(), 2);
        assert!(result
            .relationships
            .iter()
            .all(|(scope, _, _)| scope == "Cells"));
    }

    #[test]
    fn neighbors_scopes_computes_for_an_arbitrary_object_scope() {
        let images = [("DNA", &[1.0; 6][..])];
        let labels = two_touching_objects_labels();
        let objects = [("Nuclei", &labels[..])];
        let result = measure(&Input {
            shape: &[1, 6],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL,
            neighbors_scopes: &["Nuclei"],
            size_shape: None,
            neighbors_full_objects: &[],
        })
        .unwrap();
        assert!(result.features["Nuclei"].contains_key("Neighbors_NumberOfNeighbors_Adjacent"));
        assert!(!result.features.contains_key("Cells"));
        assert!(result
            .relationships
            .iter()
            .all(|(scope, _, _)| scope == "Nuclei"));
    }

    #[test]
    fn neighbors_scopes_computes_each_requested_scope_independently() {
        let images = [("DNA", &[1.0; 6][..])];
        let cells_labels = two_touching_objects_labels();
        let nuclei_labels = two_touching_objects_labels();
        let objects = [("Cells", &cells_labels[..]), ("Nuclei", &nuclei_labels[..])];
        let result = measure(&Input {
            shape: &[1, 6],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL,
            neighbors_scopes: &["Cells", "Nuclei"],
            size_shape: None,
            neighbors_full_objects: &[],
        })
        .unwrap();
        assert!(result.features["Cells"].contains_key("Neighbors_NumberOfNeighbors_Adjacent"));
        assert!(result.features["Nuclei"].contains_key("Neighbors_NumberOfNeighbors_Adjacent"));
        let cells_count = result
            .relationships
            .iter()
            .filter(|(scope, _, _)| scope == "Cells")
            .count();
        let nuclei_count = result
            .relationships
            .iter()
            .filter(|(scope, _, _)| scope == "Nuclei")
            .count();
        assert_eq!(cells_count, 2);
        assert_eq!(nuclei_count, 2);
    }

    #[test]
    fn neighbors_scopes_errors_clearly_on_a_scope_with_no_object_set() {
        let images = [("DNA", &[1.0; 6][..])];
        let labels = two_touching_objects_labels();
        let objects = [("Cells", &labels[..])];
        let error = measure(&Input {
            shape: &[1, 6],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL,
            neighbors_scopes: &["Nuclei"],
            size_shape: None,
            neighbors_full_objects: &[],
        })
        .unwrap_err();
        assert!(error.contains("Nuclei"));
    }

    #[test]
    fn neighbors_full_objects_counts_a_border_excluded_object_as_a_real_neighbor() {
        // Real CellProfiler 4.2.8 oracle ground truth (MeasureObjectNeighbors,
        // wants_excluded_objects=True, i.e. its own default), for a 2x9
        // image with three touching objects [1,1,1|2,2,2|3,3,3] where
        // object 3 was discarded from "Cells" (kept) but is still present
        // in "small_removed" (full): count=[1,2], first=[2,1], second=[0,0],
        // touching=[33.333...,66.666...]%, relate=[(1,2),(2,1)] only (the
        // excluded object never gets its own relationship edge).
        let full: [u32; 18] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3];
        let kept: [u32; 18] = [1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0];
        let images = [("DNA", &[1.0; 18][..])];
        let objects = [("Cells", &kept[..])];
        let result = measure(&Input {
            shape: &[2, 9],
            images: &images,
            objects: &objects,
            mask: None,
            families: Families::ALL,
            neighbors_scopes: &["Cells"],
            size_shape: None,
            neighbors_full_objects: &[("Cells", &full[..])],
        })
        .unwrap();
        let features = &result.features["Cells"];
        let get_int = |name: &str| match &features[name] {
            Column::Int(v) => v.clone(),
            Column::Float(_) => panic!("expected int column {name}"),
        };
        let get_float = |name: &str| match &features[name] {
            Column::Float(v) => v.clone(),
            Column::Int(_) => panic!("expected float column {name}"),
        };
        assert_eq!(get_int("Neighbors_NumberOfNeighbors_Adjacent"), vec![1, 2]);
        assert_eq!(
            get_int("Neighbors_FirstClosestObjectNumber_Adjacent"),
            vec![2, 1]
        );
        assert_eq!(
            get_int("Neighbors_SecondClosestObjectNumber_Adjacent"),
            vec![0, 0]
        );
        let touching = get_float("Neighbors_PercentTouching_Adjacent");
        assert!((touching[0] - 33.333_333_333_333_336).abs() < 1e-9);
        assert!((touching[1] - 66.666_666_666_666_67).abs() < 1e-9);
        assert_eq!(
            result.relationships,
            vec![("Cells".to_owned(), 1, 2), ("Cells".to_owned(), 2, 1),]
        );
    }
}

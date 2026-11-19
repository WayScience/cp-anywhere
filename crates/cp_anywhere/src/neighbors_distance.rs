//! 2D self-paired CellProfiler 4.2.8 MeasureObjectNeighbors (D_WITHIN).

use std::collections::{BTreeMap, BTreeSet};

use crate::{neighbor_outline, numpy_cmp, put_float, put_int, ResultTable, MAX_LABEL};

pub(crate) fn within(
    result: &mut ResultTable,
    scope: &str,
    kept: &[u32],
    geometry: &[u32],
    shape: &[usize],
    distance: u32,
) -> Result<(), String> {
    if shape.len() != 2 {
        return Err(
            "MeasureObjectNeighbors 'Within a specified distance' supports 2D labels only".into(),
        );
    }
    if geometry.len() != kept.len() || geometry.iter().any(|&id| id > MAX_LABEL) {
        return Err(
            "invalid fixed-distance neighbor geometry: check label values and shape".into(),
        );
    }
    let kept_max = kept.iter().copied().max().unwrap_or(0) as usize;
    let full_max = geometry.iter().copied().max().unwrap_or(0) as usize;
    if kept_max > 0 && full_max == 0 {
        return Err("fixed-distance neighbor geometry is empty for a nonempty object set".into());
    }
    // CP does not recompute the closest-ID columns unless the diagnostic
    // maximum exceeds one. In particular, one fully excluded object yields
    // `[0]` in these two columns even though there are no retained rows.
    let closest_len = if full_max <= 1 { full_max } else { kept_max };
    let mut closest = vec![0; closest_len];
    let mut second = vec![0; closest_len];
    let mut counts = vec![0; kept_max];
    let mut d1 = vec![0.0; kept_max];
    let mut d2 = vec![0.0; kept_max];
    let mut angles = vec![0.0; kept_max];
    let mut touching = vec![0.0; kept_max];
    if full_max > 1 && kept_max > 0 {
        let mut centers = vec![[0.0_f64; 2]; full_max];
        let mut area = vec![0usize; full_max];
        let mut present = vec![false; kept_max];
        let mut overlap = BTreeMap::new();
        let width = shape[1];
        for (i, (&id, &kept_id)) in geometry.iter().zip(kept).enumerate() {
            if id != 0 {
                let k = id as usize - 1;
                area[k] += 1;
                centers[k][0] += (i / width) as f64;
                centers[k][1] += (i % width) as f64;
            }
            if kept_id != 0 {
                present[kept_id as usize - 1] = true;
                if id != 0 {
                    *overlap.entry((kept_id as usize - 1, id)).or_insert(0) += 1;
                }
            }
        }
        for (center, n) in centers.iter_mut().zip(area) {
            center[0] /= n as f64;
            center[1] /= n as f64;
        }
        // Objects.relate_histogram uses argmax, with the lowest diagnostic ID
        // winning an overlap tie. Zero is the mapping for absent kept IDs.
        let mut mapping = vec![0; kept_max];
        let mut best = vec![0; kept_max];
        for ((k, id), n) in overlap {
            if n > best[k] {
                best[k] = n;
                mapping[k] = id;
            }
        }
        let mut reverse = vec![0; full_max + 1];
        for (k, &id) in mapping.iter().enumerate() {
            reverse[id as usize] = k as i64 + 1;
        }
        let (edges, full_touching) = morphology(geometry, &reverse, shape, distance);
        let index = |id: u32| {
            if id == 0 {
                full_max - 1
            } else {
                id as usize - 1
            }
        };
        let mut cached = BTreeMap::new();
        let mut ranks = Vec::new();
        let mut partition = Vec::new();
        let mut distances = Vec::new();
        for (k, &id) in mapping.iter().enumerate() {
            let center = centers[index(id)];
            let [first_distance, second_distance, angle] = *cached.entry(id).or_insert_with(|| {
                // CP indexes diagnostic measurements with -1 for a missing
                // retained ID. It then zeroes only counts and touching.
                let idx = index(id);
                raw_nearest(centers[idx], &centers, &mut distances, &mut partition)
            });
            d1[k] = first_distance;
            d2[k] = second_distance;
            angles[k] = angle;
            if !present[k] {
                continue;
            }
            let adjacent = &edges[index(id)];
            counts[k] = adjacent.len() as i64;
            touching[k] = full_touching[index(id)];
            if id != 0 {
                // CP iterates retained IDs and visits each object's unique
                // diagnostic neighbor IDs in ascending order before remapping.
                let first = reverse[id as usize];
                for &other_id in adjacent {
                    let other = reverse[other_id as usize];
                    if other != 0 {
                        result.relationships.push((scope.into(), first, other));
                    }
                }
            }
            // Closest IDs use a stable ordering on *retained* IDs. Absent
            // IDs have infinite distance (and sort before NaNs).
            ranks.clear();
            ranks.extend(
                mapping
                    .iter()
                    .zip(&present)
                    .enumerate()
                    .map(|(j, (&b, &has_pixels))| {
                        let d = if has_pixels {
                            distance_to(center, centers[index(b)])
                        } else {
                            f64::INFINITY
                        };
                        (j, d)
                    }),
            );
            // Only ranks 1 and 2 (zero based) are used; including the ID in
            // the key gives the stable lexsort tie-break without sorting the
            // entire retained distance row.
            sort_first_three(&mut ranks, |a, b| {
                numpy_cmp(&a.1, &b.1).then(a.0.cmp(&b.0)).is_lt()
            });
            if kept_max > 1 {
                closest[k] = ranks[1].0 as i64 + 1;
            }
            if kept_max > 2 {
                second[k] = ranks[2].0 as i64 + 1;
            }
        }
    }
    for (name, values) in [
        ("NumberOfNeighbors", counts),
        ("FirstClosestObjectNumber", closest),
        ("SecondClosestObjectNumber", second),
    ] {
        put_int(
            result,
            scope,
            format!("Neighbors_{name}_{distance}"),
            values,
        );
    }
    for (name, values) in [
        ("FirstClosestDistance", d1),
        ("SecondClosestDistance", d2),
        ("AngleBetweenNeighbors", angles),
        ("PercentTouching", touching),
    ] {
        put_float(
            result,
            scope,
            format!("Neighbors_{name}_{distance}"),
            values,
        );
    }
    Ok(())
}

fn squared_norm([y, x]: [f64; 2]) -> f64 {
    y * y + x * x
}

fn difference(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [b[0] - a[0], b[1] - a[1]]
}

fn distance_to(a: [f64; 2], b: [f64; 2]) -> f64 {
    squared_norm(difference(a, b)).sqrt()
}

// The first three consecutive pivots of NumPy 1.23.5 argpartition
// (selection.cpp, `dumb_select_`, see licenses/numpy.txt) use minimum swaps.
// A stable sort is not interchangeable: a swap can reorder tied candidates
// and change the distance vectors used for the angle.
fn sort_first_three<T>(items: &mut [T], less: impl Fn(&T, &T) -> bool) {
    for pos in 0..3.min(items.len()) {
        let mut smallest = pos;
        for j in pos + 1..items.len() {
            if less(&items[j], &items[smallest]) {
                smallest = j;
            }
        }
        items.swap(pos, smallest);
    }
}

fn raw_nearest(
    center: [f64; 2],
    centers: &[[f64; 2]],
    distances: &mut Vec<f64>,
    order: &mut Vec<usize>,
) -> [f64; 3] {
    distances.clear();
    distances.extend(centers.iter().map(|&other| distance_to(center, other)));
    order.clear();
    order.extend(0..centers.len());
    sort_first_three(order, |&a, &b| {
        numpy_cmp(&distances[a], &distances[b]).is_lt()
    });
    let a = difference(center, centers[order[1]]);
    let a2 = squared_norm(a);
    if centers.len() == 2 {
        return [a2.sqrt(), 0.0, 0.0];
    }
    let b = difference(center, centers[order[2]]);
    let b2 = squared_norm(b);
    let angle =
        ((a[0] * b[0] + a[1] * b[1]) / (a2 * b2).sqrt()).acos() * 180.0 / std::f64::consts::PI;
    [a2.sqrt(), b2.sqrt(), angle]
}

fn disk_width(squared_radius: u64, maximum_width: usize) -> usize {
    let mut lo = 0;
    let mut hi = maximum_width as u64 + 1;
    while hi - lo > 1 {
        let midpoint = lo + (hi - lo) / 2;
        if midpoint * midpoint <= squared_radius {
            lo = midpoint;
        } else {
            hi = midpoint;
        }
    }
    lo as usize
}

fn morphology(
    labels: &[u32],
    reverse: &[i64],
    shape: &[usize],
    radius: u32,
) -> (Vec<BTreeSet<u32>>, Vec<f64>) {
    let (height, width) = (shape[0], shape[1]);
    let r = radius as usize;
    // The reference uses strel_disk(r) for neighbor counts and
    // strel_disk(r + 0.5) for percent touching. Both footprints have
    // offsets bounded by r in each dimension. Clamp to the image before
    // allocating so even a radius larger than the image fits comfortably.
    let disk2 = u64::from(radius).pow(2);
    let touching2 = disk2 + u64::from(radius); // floor((r + 0.5)^2)
    let vertical = r.min(height - 1);
    let horizontal = r.min(width - 1);
    let rows: Vec<_> = (0..=vertical)
        .map(|dy| {
            let dy2 = (dy as u64).pow(2);
            (
                disk_width(disk2 - dy2, horizontal),
                disk_width(touching2 - dy2, horizontal),
            )
        })
        .collect();
    let outline = neighbor_outline(labels, shape);
    let n = reverse.len() - 1;
    let mut edges = vec![BTreeSet::new(); n];
    let mut perimeters = vec![0usize; n];
    let mut touching = vec![0usize; n];
    // For two distinct labels the closest pair includes a boundary pixel
    // of either object: a step from an interior pixel toward the other
    // object would otherwise reduce the distance. The touching calculation
    // also evaluates only boundary pixels, but tests against all other pixels.
    for (i, (&id, &boundary)) in labels.iter().zip(&outline).enumerate() {
        if !boundary || reverse[id as usize] == 0 {
            continue;
        }
        let slot = id as usize - 1;
        perimeters[slot] += 1;
        let (y, x) = (i / width, i % width);
        let mut touches = false;
        for yy in y.saturating_sub(vertical)..=y.saturating_add(vertical).min(height - 1) {
            let (count_dx, touch_dx) = rows[y.abs_diff(yy)];
            let left = x.saturating_sub(touch_dx);
            let right = x.saturating_add(touch_dx).min(width - 1);
            for (offset, &other) in labels[yy * width + left..=yy * width + right]
                .iter()
                .enumerate()
            {
                if other != 0 && other != id {
                    touches = true;
                    if x.abs_diff(left + offset) <= count_dx {
                        edges[slot].insert(other);
                    }
                }
            }
        }
        touching[slot] += usize::from(touches);
    }
    let percentages = touching
        .into_iter()
        .zip(perimeters)
        .map(|(t, p)| {
            if p == 0 {
                0.0
            } else {
                t as f64 * 100.0 / p as f64
            }
        })
        .collect();
    (edges, percentages)
}

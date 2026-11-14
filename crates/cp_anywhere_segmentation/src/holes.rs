//! CellProfiler/centrosome's four-connected foreground/background label graph.
//!
//! The graph traversal is a Rust adaptation of `fill_labeled_holes` and
//! `fill_labeled_holes_loop` from CellProfiler/centrosome:
//! <https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py>
//! <https://github.com/CellProfiler/centrosome/blob/master/centrosome/_cpmorphology2.pyx>
//!
//! Upstream's BSD license is reproduced below and must accompany redistribution.
/*
The BSD 3-Clause License

Copyright © 2003 - 2019 Broad Institute, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1.  Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2.  Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

    3.  Neither the name of the Broad Institute, Inc. nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED “AS IS.”  BROAD MAKES NO EXPRESS OR IMPLIED
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
*/

/// Fill enclosed background and absorb enclosed foreground labels in place.
///
/// Unlike per-object binary hole filling, this preserves holes shared between
/// multiple exterior objects and can absorb nested foreground objects.
/// Foreground identity is the label value, even for disconnected pieces;
/// background components and graph edges use four-connectivity.
///
/// `None` implements centrosome's default `size_fn=None`. `Some(max_area)`
/// implements IdentifyPrimaryObjects' exact predicate `size < max_area` for
/// **both** foreground labels and background components. The module passes its
/// maximum diameter squared as `max_area`, not a circle area. Equality is
/// excluded; fractional limits are not rounded. Foreground area counts all
/// pixels with the same original label, even disconnected pieces. Regions that
/// fail the size predicate become non-hole seeds, which can prevent other
/// regions from being absorbed. This is not a background-only area cutoff and
/// is not a substitute for centrosome's arbitrary `size_fn` callbacks.
/// NaN and nonpositive limits reject every positive-area region; infinity
/// permits every region. No mask is applied by this API.
///
/// # Panics
///
/// Panics if `rows * cols` overflows or does not equal `labels.len()`.
pub fn fill_labeled_holes(labels: &mut [u32], rows: usize, cols: usize, max_area: Option<f64>) {
    assert_eq!(rows.checked_mul(cols), Some(labels.len()), "label shape mismatch");
    if labels.is_empty() {
        return;
    }

    // Compress only the foreground IDs, in original numeric order. This avoids
    // allocations proportional to the maximum label while preserving upstream's
    // sorted adjacency and LIFO traversal order, including sparse u32 labels.
    let mut foreground: Vec<u32> = labels.iter().copied().filter(|&x| x != 0).collect();
    foreground.sort_unstable();
    foreground.dedup();
    let foreground_count = foreground.len();
    let mut nodes: Vec<usize> = labels
        .iter()
        .map(|label| {
            if *label == 0 {
                0
            } else {
                foreground.binary_search(label).unwrap() + 1
            }
        })
        .collect();

    let mut node_count = foreground_count + 1; // Node zero stays unused.
    let mut pending = Vec::new();
    for seed in 0..nodes.len() {
        if nodes[seed] != 0 {
            continue;
        }
        let component = node_count;
        node_count += 1;
        nodes[seed] = component;
        pending.push(seed);
        while let Some(pixel) = pending.pop() {
            for neighbor in neighbors(pixel, rows, cols).into_iter().flatten() {
                if nodes[neighbor] == 0 {
                    nodes[neighbor] = component;
                    pending.push(neighbor);
                }
            }
        }
    }

    let mut graph = vec![Vec::new(); node_count];
    let mut non_hole = vec![false; node_count];
    for row in 0..rows {
        for col in 0..cols {
            let pixel = row * cols + col;
            let node = nodes[pixel];
            if row == 0 || col == 0 || row + 1 == rows || col + 1 == cols {
                non_hole[node] = true;
            }
            for neighbor in [
                (row + 1 < rows).then_some(pixel + cols),
                (col + 1 < cols).then_some(pixel + 1),
            ]
            .into_iter()
            .flatten()
            {
                let other = nodes[neighbor];
                if node != other {
                    graph[node].push(other);
                    graph[other].push(node);
                }
            }
        }
    }
    for adjacent in &mut graph {
        adjacent.sort_unstable();
        adjacent.dedup();
    }
    let mut todo: Vec<usize> = (1..node_count).filter(|&node| non_hole[node]).collect();
    if let Some(maximum) = max_area {
        let mut areas = vec![0_usize; node_count];
        for &node in &nodes {
            areas[node] += 1;
        }
        // Keep rejected regions behind the ascending edge seeds on the stack,
        // matching Python's size_fn enumeration and the compiled loop's order.
        for node in 1..node_count {
            if non_hole[node] || (areas[node] as f64) < maximum {
                continue;
            }
            non_hole[node] = true;
            todo.push(node);
        }
    }
    let mut owner = vec![0; node_count];

    // A background non-hole exposes neighboring foreground. A foreground
    // non-hole supplies a candidate owner; a second distinct owner exposes the
    // neighbor too. This is deliberately a LIFO walk, as in centrosome.
    while let Some(node) = todo.pop() {
        for &adjacent in &graph[node] {
            if non_hole[adjacent] {
                continue;
            }
            if node <= foreground_count {
                if owner[adjacent] == 0 {
                    owner[adjacent] = node;
                    continue;
                }
                if owner[adjacent] == node {
                    continue;
                }
            } else if adjacent > foreground_count {
                continue;
            }
            non_hole[adjacent] = true;
            todo.push(adjacent);
        }
    }

    // Propagate the surviving enclosing owner through all nested hole nodes,
    // including foreground nodes (the behavior a background-only fill misses).
    todo.extend((1..node_count).filter(|&node| !non_hole[node] && owner[node] != 0));
    while let Some(node) = todo.pop() {
        for &adjacent in &graph[node] {
            if !non_hole[adjacent] && owner[adjacent] == 0 {
                owner[adjacent] = owner[node];
                todo.push(adjacent);
            }
        }
    }
    for (label, node) in labels.iter_mut().zip(nodes) {
        if !non_hole[node] {
            *label = if owner[node] == 0 { 0 } else { foreground[owner[node] - 1] };
        }
    }
}

fn neighbors(pixel: usize, rows: usize, cols: usize) -> [Option<usize>; 4] {
    let row = pixel / cols;
    let col = pixel % cols;
    [
        (row > 0).then(|| pixel - cols),
        (col > 0).then(|| pixel - 1),
        (row + 1 < rows).then_some(pixel + cols),
        (col + 1 < cols).then_some(pixel + 1),
    ]
}

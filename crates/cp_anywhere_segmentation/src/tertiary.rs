//! `IdentifyTertiaryObjects` (CellProfiler 4.2.8), standalone: removes
//! a smaller "primary" object set (e.g. nuclei) from a larger
//! "secondary" object set (e.g. cells), leaving a ring shape (e.g.
//! cytoplasm). No thresholding, no declumping -- a pure per-pixel
//! subtraction, oracle-verified end to end against a real
//! `IdentifyTertiaryObjects.run()` call (both `shrink_primary=true`
//! and `shrink_primary=false`) before being ported here, not derived
//! from reading `centrosome.outline.outline`'s source alone.
//!
//! Unlike `IdentifyPrimaryObjects`/`IdentifySecondaryObjects`, the
//! real module never relabels: the output keeps `secondary_labels`'s
//! exact label numbering verbatim (some labels may end up with zero
//! surviving pixels if a secondary object is entirely consumed by its
//! own primary).

/// `centrosome.outline.outline(labels) != 0`: true for every nonzero
/// pixel that has at least one 8-connected neighbor with a different
/// label value, plus every pixel on the image's own border (confirmed
/// against `centrosome/outline.py`'s real source verbatim -- this is
/// a complete, unambiguous algorithm, not an approximation).
fn outline_mask(labels: &[u32], w: usize, h: usize) -> Vec<bool> {
    let mut different = vec![false; w * h];
    let at = |r: usize, c: usize| labels[r * w + c];
    for r in 0..h {
        for c in 0..w {
            let is_border = r == 0 || c == 0 || r + 1 == h || c + 1 == w;
            if is_border {
                different[r * w + c] = true;
                continue;
            }
            let v = at(r, c);
            let mut differs = false;
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nc < 0 || nr as usize >= h || nc as usize >= w {
                        continue;
                    }
                    if at(nr as usize, nc as usize) != v {
                        differs = true;
                        break;
                    }
                }
                if differs {
                    break;
                }
            }
            different[r * w + c] = differs;
        }
    }
    different
}

/// Removes `primary_labels` from `secondary_labels`, returning the
/// tertiary object labels (same shape, `secondary_labels`'s own label
/// numbering preserved verbatim). `shrink_primary` (CellProfiler's own
/// default is `true`) keeps the primary object's own 1-pixel-wide
/// outline as part of the tertiary output, guaranteeing a nonempty
/// tertiary object even when the primary fills its secondary almost
/// entirely; `false` subtracts the primary's full extent, leaving no
/// pixels shared between primary/secondary/tertiary.
pub fn identify_tertiary_objects(
    primary_labels: &[u32],
    secondary_labels: &[u32],
    w: usize,
    h: usize,
    shrink_primary: bool,
) -> Result<Vec<u32>, String> {
    let n = w.checked_mul(h).ok_or("image dimensions overflow")?;
    if n != primary_labels.len() || n != secondary_labels.len() {
        return Err(
            "primary_labels and secondary_labels must both match the given width/height"
                .to_string(),
        );
    }
    let mut tertiary = secondary_labels.to_vec();
    let outline = if shrink_primary {
        Some(outline_mask(primary_labels, w, h))
    } else {
        None
    };
    for i in 0..n {
        let on_outline = outline.as_ref().is_some_and(|o| o[i]);
        let primary_mask = primary_labels[i] == 0 || on_outline;
        if !primary_mask {
            tertiary[i] = 0;
        }
    }
    Ok(tertiary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_primary_has_no_interior_to_remove() {
        // A one-row strip is entirely on the image border.
        let secondary = [1u32; 8];
        let primary = [0, 0, 0, 1, 1, 0, 0, 0];
        let result = identify_tertiary_objects(&primary, &secondary, 8, 1, true).unwrap();
        // The primary's own outline (every primary pixel here, since
        // it's only 2px wide with no interior) stays part of tertiary.
        assert_eq!(result, [1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn no_shrink_subtracts_thin_primary() {
        let secondary = [1u32; 8];
        let primary = [0, 0, 0, 1, 1, 0, 0, 0];
        let result = identify_tertiary_objects(&primary, &secondary, 8, 1, false).unwrap();
        assert_eq!(result, [1, 1, 1, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn matches_a_real_2d_end_to_end_oracle_run_with_shrink_primary() {
        // 9x9, one nucleus/one cell (real CellProfiler 4.2.8
        // IdentifyPrimaryObjects -> IdentifySecondaryObjects ->
        // IdentifyTertiaryObjects, shrink_primary=True).
        #[rustfmt::skip]
        let nuclei: [u32; 81] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        #[rustfmt::skip]
        let cells: [u32; 81] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        #[rustfmt::skip]
        let expected_shrink: [u32; 81] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 1, 0, 1, 1, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let result = identify_tertiary_objects(&nuclei, &cells, 9, 9, true).unwrap();
        assert_eq!(result, expected_shrink);

        #[rustfmt::skip]
        let expected_no_shrink: [u32; 81] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let result_no_shrink = identify_tertiary_objects(&nuclei, &cells, 9, 9, false).unwrap();
        assert_eq!(result_no_shrink, expected_no_shrink);
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let err = identify_tertiary_objects(&[0u32; 4], &[0u32; 6], 2, 2, true).unwrap_err();
        assert!(err.contains("match"));
    }

    #[test]
    fn a_secondary_object_fully_consumed_by_its_primary_becomes_empty() {
        // shrink_primary=false, primary == secondary everywhere -> the
        // tertiary object has zero surviving pixels but the array
        // isn't relabeled (matches real CellProfiler's own behavior:
        // IdentifyTertiaryObjects never calls a relabeling step).
        let labels = [1u32; 4];
        let result = identify_tertiary_objects(&labels, &labels, 2, 2, false).unwrap();
        assert_eq!(result, [0, 0, 0, 0]);
    }
}

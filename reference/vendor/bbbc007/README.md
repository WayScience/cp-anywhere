# BBBC007 vendored real-data crop

Four small `.npy` arrays, each a 128x128 crop (rows 256:384, columns
192:320) of one real field of view from the Broad Bioimage Benchmark
Collection's "Drosophila Kc167 cells" image set (BBBC007v1, image `A9 p10`).

- `dna.npy`: real DNA-stain channel, uint8, from `A9 p10d.tif`.
- `actin.npy`: real actin-stain channel, uint8, from `A9 p10f.tif`.
- `cells.npy`: integer object labels, int32, 4 objects. Derived from the
  dataset's own manually-outlined cell-boundary ground truth
  (`A9 p10f.tif` in the outlines archive) by inverting the boolean
  outline mask, connected-component labeling (`skimage.measure.label`,
  4-connectivity), dropping the border-touching background component,
  and renumbering the remainder 1..N.
- `nuclei.npy`: integer object labels, int32, 4 objects. Same derivation,
  from the DNA-channel outline ground truth (`A9 p10d.tif` in the
  outlines archive). Each nucleus is fully contained within exactly one
  cell (verified: 100% pixel overlap between each nucleus label and its
  corresponding cell label).

## Source and license

- Dataset: BBBC007v1 ("Drosophila Kc167 cells"), Jones et al., Proc. ICCV
  Workshop on Computer Vision for Biomedical Image Applications, 2005;
  distributed by the Broad Bioimage Benchmark Collection
  (Ljosa et al., Nature Methods, 2012): <https://bbbc.broadinstitute.org/BBBC007>
- License: the official page states "To the extent possible under law,
  Anne Carpenter has waived all copyright and related or neighboring
  rights to Drosophila Kc167 images and ground truth" -- a CC0 (public
  domain) waiver, matching the project's existing real-data fixture
  (`real_cell_cc0`, also CC0).
- Downloaded from `https://data.broadinstitute.org/bbbc/BBBC007/BBBC007_v1_images.zip`
  and `.../BBBC007_v1_outlines.zip` on 2026-11-14.
- Original full-field source file SHA-256 (before cropping, for
  provenance -- these source TIFFs are not themselves vendored, only the
  128x128 crop derived from them):
  - `A9 p10d.tif` (DNA, 450x450 uint8):
    `1abc5c6e7c53bdb7e97fd9409ee963e9cb903496e1776f585fa338cd9f5563e2`
  - `A9 p10f.tif` (Actin, 450x450 uint8):
    `fabeb893e3060813d650cf4c233c06732fd2f0cdbf468da55381bbf334bc2313`
  - outline `A9 p10d.tif` (boolean, 450x450):
    `c96f5f2e01746d4ce4fe6abff50d370c70abc65e94a5dbf3605c72774945d0d1`
  - outline `A9 p10f.tif` (boolean, 450x450):
    `f49ebad42056494a7358641413497af2ff304d63bf4c53bfa3fcdee96999b0ea`

No calibrated pixel spacing is published for this image set (the BBBC007
page states only "cells roughly 25 pixels in diameter"); the fixture uses
uncalibrated `(1.0, 1.0)` spacing, matching the project's convention for
other uncalibrated fixtures.

## Reproducing the vendored arrays

```sh
curl -A 'Mozilla/5.0' -o images.zip 'https://data.broadinstitute.org/bbbc/BBBC007/BBBC007_v1_images.zip'
curl -A 'Mozilla/5.0' -o outlines.zip 'https://data.broadinstitute.org/bbbc/BBBC007/BBBC007_v1_outlines.zip'
unzip images.zip && unzip outlines.zip
```

Then, in an environment with `numpy`, `tifffile`, and `scikit-image`
(the pinned reference venv already has all three via
`reference/requirements.lock`):

```python
import numpy as np, tifffile
from skimage.measure import label

dna = tifffile.imread("BBBC007_v1_images/A9/A9 p10d.tif")
actin = tifffile.imread("BBBC007_v1_images/A9/A9 p10f.tif")
dna_outline = tifffile.imread("BBBC007_v1_outlines/A9/A9 p10d.tif")
actin_outline = tifffile.imread("BBBC007_v1_outlines/A9/A9 p10f.tif")


def labels_from_outline(outline):
    labeled = label(~outline, connectivity=1)
    border = (
        set(labeled[0, :])
        | set(labeled[-1, :])
        | set(labeled[:, 0])
        | set(labeled[:, -1])
    )
    border.discard(0)
    out = labeled.copy()
    for b in border:
        out[out == b] = 0
    remap = {
        old: new for new, old in enumerate(sorted(set(np.unique(out)) - {0}), start=1)
    }
    renumbered = np.zeros_like(out, dtype=np.int32)
    for old, new in remap.items():
        renumbered[out == old] = new
    return renumbered


y0, x0, size = 256, 192, 128
dna_crop = dna[y0 : y0 + size, x0 : x0 + size]
actin_crop = actin[y0 : y0 + size, x0 : x0 + size]
cells = labels_from_outline(actin_outline[y0 : y0 + size, x0 : x0 + size])
nuclei = labels_from_outline(dna_outline[y0 : y0 + size, x0 : x0 + size])
```

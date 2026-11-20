# Third-party notices

Original project code uses BSD-3-Clause; see `LICENSE`.
Third-party code retains its own license.

## Repository template

CU-DBMI/template-uv-python-research-software, revision `23ac69fab29d229762ffd6c3741770292efd5d62`.
Copyright 2025, the Regents of the University of Colorado.
License: BSD-3-Clause; full notice in `licenses/template-uv-python-research-software.txt`.
The generated repository infrastructure and agent guidance derive from this template.
Individual imported guidance retains source and license metadata in its skill file.

## Scientific references

CellProfiler v4.2.8 source, revision `2df090ff281cff80943fe679a1c3192fbda9cd6d`.
Copyright 2003–2021 Broad Institute, Inc.
License: BSD-3-Clause; full notice in `licenses/cellprofiler.txt`.
The original modules supply scientific reference measurements.

Centrosome v1.2.3 source, revision `533caaa5dd56d20fa914da4db6e31be1c6ecea88`.
Copyright 2003–2019 Broad Institute, Inc.
License: BSD-3-Clause; full notice in `crates/cp_anywhere/LICENSE-centrosome`.
The native `MeasureImageQuality` core follows the radial power spectrum and
Haralick correlation behavior. The Rust FFT is standalone; centrosome and
SciPy are required only to regenerate the reference measurements.

cp_measure source, revision `cdf693187e3dcaf3d067da7b58bdd4ab8896f91e`, package version 0.3.0.
Copyright 2025 Alán F. Muñoz.
License: MIT; full notice in `licenses/cp_measure.txt`.
This is a comparison implementation, not the scientific oracle.

NumPy v1.23.5 source, revision `de82cd9468704a033702974010ee7e7efc85b393`.
Copyright 2005–2022 NumPy Developers.
License: BSD-3-Clause; full notice in `licenses/numpy.txt`.
The native core translates NumPy's index quicksort ordering to match the pinned
SciPy maximum-position behavior for tied image intensities.

## SLEEF

SLEEF 3.9.0 source, revision `906ca7512ee483296780a81a21b9ca715d40dfe1`.
Copyright Naoki Shibata and contributors 2010–2025.
The Zernike circle solver uses a Rust port of the scalar `acos_u35` path.
The port retains the full Boost Software License 1.0 notice and source file
references in `crates/cp_anywhere/src/sleef_acos.rs`.

Dependency notices remain in their package distributions and lockfiles.
Fixture-specific source and redistribution terms are recorded alongside fixture provenance.

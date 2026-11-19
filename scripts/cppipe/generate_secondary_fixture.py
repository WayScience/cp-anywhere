#  ty: ignore[import-not-found]
"""Generate a real .cppipe fixture for IdentifyPrimaryObjects ->
IdentifySecondaryObjects, round-trip-verifying
cp_anywhere.cppipe's IdentifySecondaryObjects translation (and the
full Pipeline.run() wiring: primary.labels/small_removed_labels ->
identify_secondary_objects) against genuine CellProfiler-written
text and a freshly-run real oracle -- never guessed.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.identifysecondaryobjects import (
    M_PROPAGATION,
    M_WATERSHED_I,
    IdentifySecondaryObjects,
)
from cellprofiler_core.image import Image, ImageSetList, VanillaImage
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "fixtures/cppipe/secondary"

sys.path.insert(0, str(ROOT))

# Two overlapping circular blobs (DNA/nuclei seeds) inside a larger,
# smooth secondary stain -- real "Cells" propagation around real
# "Nuclei" primaries, matching the shared pattern this project's own
# segmentation fixtures already use.
ROWS, COLS = 20, 24
YY, XX = np.mgrid[0:ROWS, 0:COLS]
D1 = np.sqrt((YY - 9) ** 2 + (XX - 8) ** 2)
D2 = np.sqrt((YY - 9) ** 2 + (XX - 15) ** 2)
DNA_IMAGE = (
    np.maximum(np.clip(1 - D1 / 3, 0, 1), np.clip(1 - D2 / 3, 0, 1)) * 65535
).astype(np.uint16)
STAIN_IMAGE = np.maximum(np.clip(1 - D1 / 7, 0, 1), np.clip(1 - D2 / 7, 0, 1))


def build_and_run(method: str = M_PROPAGATION) -> None:
    stem = "identify_secondary_objects_" + (
        "watershed_image" if method == M_WATERSHED_I else "propagation"
    )
    pipeline = Pipeline()
    m1 = IdentifyPrimaryObjects()
    m1.set_module_num(1)
    m1.x_name.value = "DNA"
    m1.y_name.value = "Nuclei"
    m1.exclude_size.value = False
    m1.use_advanced.value = True
    pipeline.add_module(m1)

    m2 = IdentifySecondaryObjects()
    m2.set_module_num(2)
    m2.x_name.value = "Nuclei"
    m2.y_name.value = "Cells"
    m2.image_name.value = "Stain"
    m2.method.value = method
    m2.regularization_factor.value = 0.05
    m2.fill_holes.value = True
    m2.wants_discard_edge.value = False
    pipeline.add_module(m2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{stem}.cppipe").open("w") as fp:
        pipeline.dump(fp, save_image_plane_details=False)

    # Run the real pipeline end to end for ground truth -- one
    # Workspace per module run (matching
    # scripts/segmentation/generate_reference.py's own established
    # pattern), sharing the same image_set/object_set/measurements so
    # IdentifySecondaryObjects can find IdentifyPrimaryObjects's
    # "Nuclei" output.
    image_set_list = ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.providers.append(
        VanillaImage("DNA", Image(DNA_IMAGE, convert=True, dimensions=2))
    )
    image_set.providers.append(
        VanillaImage(
            "Stain", Image(STAIN_IMAGE, convert=method == M_WATERSHED_I, dimensions=2)
        )
    )
    object_set = ObjectSet()
    measurements = Measurements()
    for module in pipeline.modules():
        workspace = Workspace(
            pipeline, module, image_set, object_set, measurements, image_set_list
        )
        module.run(workspace)
    nuclei = object_set.get_objects("Nuclei")
    cells = object_set.get_objects("Cells")

    if method == M_PROPAGATION:
        np.save(OUT_DIR / f"{stem}.dna.npy", DNA_IMAGE)
        np.save(OUT_DIR / f"{stem}.stain.npy", STAIN_IMAGE)
        np.save(
            OUT_DIR / f"{stem}.nuclei_labels.npy", nuclei.segmented.astype(np.uint32)
        )
    (OUT_DIR / f"{stem}.expected.json").write_text(
        json.dumps(
            {
                "nuclei_object_count": int(nuclei.segmented.max()),
                "cells_object_count": int(cells.segmented.max()),
                "cells_labels": cells.segmented.astype(int).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "nuclei_object_count": int(nuclei.segmented.max()),
                "cells_object_count": int(cells.segmented.max()),
            }
        )
    )


if __name__ == "__main__":
    build_and_run()
    build_and_run(M_WATERSHED_I)

# ty: ignore[import-not-found]
"""Generate a real .cppipe SaveImages module so cp_anywhere.cppipe's
translation can be checked against real CellProfiler's own literal
on-disk settings layout (see src/cp_anywhere/cppipe.py's
_SAVE_IMAGES_FIELDS and the project's standing file-format rule: never
guess a .cppipe field layout from source reading alone)."""

# ruff: noqa: E402

from pathlib import Path

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.saveimages import FN_SINGLE_NAME, SaveImages
from cellprofiler_core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/cppipe/illumination"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module = SaveImages()
    module.save_image_or_figure.value = "Image"
    module.image_name.value = "DNACorrected"
    module.file_name_method.value = FN_SINGLE_NAME
    module.single_file_name.value = "dna_corrected"
    module.file_format.value = "tiff"
    module.bit_depth.value = "16-bit integer"
    module.overwrite.value = True
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)
    with (OUT / "save_images.cppipe").open("w") as stream:
        pipeline.dump(stream, save_image_plane_details=False)
    print("save_images.cppipe written")


if __name__ == "__main__":
    generate()

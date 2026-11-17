# ty: ignore[import-not-found]
"""Generate oracle fixtures for real CellProfiler SaveImages' bit-depth
conversion (skimage.util.img_as_ubyte/img_as_uint/img_as_float32), the
exact functions cellprofiler.modules.saveimages.save_image calls before
writing anything to disk (see its BIT_DEPTH_8/16/FLOAT/RAW branches)."""

from pathlib import Path

import numpy as np
import skimage.util

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/save_images"


def _cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    grid = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    return {
        "linspace_0_to_1": grid,
        "random_uniform": rng.uniform(0.0, 1.0, size=4096),
        "edge_values": np.array(
            [0.0, 1.0, 1 / 255, 2 / 255, 254.5 / 255, 0.5, 1 / 65535, 32767.5 / 65535],
            dtype=np.float64,
        ),
        "float32_input": rng.uniform(0.0, 1.0, size=512).astype(np.float32),
    }


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pixels in _cases().items():
        ubyte = skimage.util.img_as_ubyte(pixels)
        uint16 = skimage.util.img_as_uint(pixels)
        float32 = skimage.util.img_as_float32(pixels)
        np.savez(
            OUT / f"{name}.npz",
            input=pixels,
            ubyte=ubyte,
            uint16=uint16,
            float32=float32,
        )
    print(f"written {len(_cases())} cases to {OUT}")


if __name__ == "__main__":
    generate()

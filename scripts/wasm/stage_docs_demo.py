"""Copy the WASM package and microscopy inputs into a built Sphinx site."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Stage only the runtime files and the two original uint8 planes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="Sphinx HTML output directory")
    args = parser.parse_args()
    if not (args.site / "browser-demo.html").is_file():
        parser.error("Build the Sphinx HTML before staging the demo assets.")
    package = ROOT / "crates/cp_anywhere_wasm/pkg"
    files = ("cp_anywhere.js", "cp_anywhere_bg.wasm", "LICENSE")
    if not all((package / name).is_file() for name in files):
        parser.error("Build the WASM package with scripts/wasm/build_package.py first.")

    sample = ROOT / "fixtures/real/real_bbbc007"
    dest = args.site / "_static/browser-demo"
    dest.mkdir(parents=True, exist_ok=True)
    for channel in ("DNA", "Actin"):
        pixels = np.load(sample / f"{channel}.npy", allow_pickle=False)
        if pixels.dtype != np.uint8 or pixels.shape != (128, 128):
            parser.error(f"Expected a 128 x 128 uint8 plane for {channel}.")
        (dest / f"{channel.lower()}.u8").write_bytes(pixels.tobytes(order="C"))
    for name in files:
        copyfile(package / name, dest / name)
    copyfile(ROOT / "fixtures/cppipe/tertiary/no_shrink.cppipe", dest / "demo.cppipe")


if __name__ == "__main__":
    main()

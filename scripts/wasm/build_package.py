"""Build the web bindings and set the public npm package name."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates/cp_anywhere_wasm"


def main() -> None:
    """Build the web package without publishing it."""
    subprocess.run(
        [
            "wasm-pack",
            "build",
            str(CRATE),
            "--target",
            "web",
            "--release",
            "--out-dir",
            "pkg",
            "--out-name",
            "cp_anywhere",
            "--locked",
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = CRATE / "pkg/package.json"
    package = json.loads(manifest.read_text(encoding="utf-8"))
    # wasm-pack uses the binding crate's Cargo name, not the public npm name.
    package["name"] = "cp-anywhere"
    manifest.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

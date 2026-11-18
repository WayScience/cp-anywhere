"""Generate binding-only edge cases by executing real CellProfiler, not cp-anywhere.

Run with the pinned reference Python and CellProfiler source on PYTHONPATH.
Existing reference fixtures are never changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/cp_anywhere_wasm/tests/fixtures/illumination_apply_edges.json"
REFERENCE = ROOT / "scripts/reference/generate_illumination_apply.py"


def encode(value: float) -> float | dict[str, str]:
    """Represent IEEE nonfinite values without invalid JSON numbers."""
    if math.isnan(value):
        return {"nonfinite": "NaN"}
    if math.isinf(value):
        return {"nonfinite": "+Infinity" if value > 0 else "-Infinity"}
    return float(value)


def main() -> None:
    """Run real CorrectIlluminationApply for all methods and clipping flags."""
    spec = importlib.util.spec_from_file_location("reference_apply", REFERENCE)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    values = [-math.inf, -1.0, -0.0, 0.0, 0.1, 1.0, 1e40, math.inf, math.nan]
    functions = [-math.inf, -1.0, -0.0, 0.0, 1e-300, 0.3, math.inf, math.nan]
    image = np.repeat(np.array(values)[:, None], len(functions), axis=1)
    illum = np.repeat(np.array(functions)[None, :], len(values), axis=0)
    cases = []
    for method in ("Divide", "Subtract"):
        for low in (False, True):
            for high in (False, True):
                settings = {
                    "divide_or_subtract": method,
                    "truncate_low": low,
                    "truncate_high": high,
                }
                with np.errstate(all="ignore"):
                    expected = reference._run_case(
                        {"image": image, "illum": illum, **settings}
                    )
                cases.append(
                    {
                        "settings": settings,
                        "rows": image.shape[0],
                        "cols": image.shape[1],
                        "image": [encode(v) for v in image.flat],
                        "illum": [encode(v) for v in illum.flat],
                        "expected": [encode(v) for v in expected.flat],
                    }
                )
    result = {
        "reference": (
            "real CellProfiler CorrectIlluminationApply.run "
            "via pinned reference interpreter"
        ),
        "source": str(REFERENCE.relative_to(ROOT)),
        "module_sha256": hashlib.sha256(
            Path(inspect.getfile(reference.CorrectIlluminationApply)).read_bytes()
        ).hexdigest(),
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {len(cases)} real-CP edge cases to {OUT}")


if __name__ == "__main__":
    main()

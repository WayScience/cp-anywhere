"""Stage and compare bounded QC decisions with the pinned PedCP90 pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, fields, replace
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Union
from urllib.parse import unquote, urlsplit

import numpy as np

if TYPE_CHECKING:
    from cp_anywhere.cppipe import Pipeline

PIPELINE_SHA256 = "3690993a3c75197dfd79c300fcd66cb74b47182f73412ba27b184b50e6d58e4f"
FIELD_COLUMNS = ("Metadata_Plate", "Metadata_Well", "Metadata_Site")
CHANNELS = ("DNA", "AGP", "ER", "Mito", "RNA")
ORIGINALS = tuple(f"Orig{channel}" for channel in CHANNELS)
COMPLETE_INPUTS = (
    *ORIGINALS,
    "OrigBrightfield",
    *(f"Illum{ch}" for ch in (*CHANNELS, "Brightfield")),
)
CORRECTED = tuple(f"Corr{channel}" for channel in (*CHANNELS, "Brightfield"))
FLAG = "Blur_and_Saturation_QCFlag"
CRITERIA = {
    f"ImageQuality_{measure}_{name}"
    for name in ORIGINALS
    for measure in ("PowerLogLogSlope", "PercentMaximal")
}
MAX_FIELDS = 8
ROOT = Path(__file__).resolve().parents[2]
SENTINEL_FIXTURE = ROOT / "fixtures/cppipe/image_quality/quality_flag.cppipe"
Field = tuple[str, str, str]
Scalar = Union[int, float, str]


@dataclass(frozen=True)
class Decision:
    """Observable field decision and downstream execution, not a metric tolerance."""

    flag: int
    skipped: bool
    reached_after_qc: bool

    def record(self) -> DecisionRecord:
        """Serialize the exact three decision components."""
        return {
            "flag": self.flag,
            "skipped": self.skipped,
            "reached_after_qc": self.reached_after_qc,
        }


class DecisionRecord(TypedDict):
    """Serializable decision with three required components."""

    flag: int
    skipped: bool
    reached_after_qc: bool


class Observation(TypedDict):
    """Evidence collected independently from one runtime."""

    decision: DecisionRecord
    criteria_failed: dict[str, bool]
    metrics: dict[str, Scalar]


def decision_differences(reference: Decision, actual: Decision) -> list[str]:
    """Name exact disagreements, including downstream behavior."""
    return [
        f.name
        for f in fields(Decision)
        if getattr(reference, f.name) != getattr(actual, f.name)
    ]


def sha256(payload: bytes) -> str:
    """Hash the original bytes of a file or a decoded array."""
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write an evidence file without overwriting an earlier result."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metric(value: float) -> Scalar:
    """Keep distinct nonfinite values in strict JSON."""
    return value if isfinite(value) else str(value)


def field_id(row: dict[str, str]) -> Field:
    """Read the exact plate, well, and site tokens from a published row."""
    return row[FIELD_COLUMNS[0]], row[FIELD_COLUMNS[1]], row[FIELD_COLUMNS[2]]


def select_rows(manifest: Path, requested: list[Field]) -> list[dict[str, str]]:
    """Select one published row per explicit field, in request order."""
    if not 1 <= len(requested) <= MAX_FIELDS or len(requested) != len(set(requested)):
        raise ValueError(f"Select 1-{MAX_FIELDS} unique fields")
    matches: dict[Field, list[dict[str, str]]] = {key: [] for key in requested}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if (key := field_id(row)) in matches:
                matches[key].append(row)
    if invalid := [key for key, rows in matches.items() if len(rows) != 1]:
        raise ValueError(f"Expected exactly one manifest row per field: {invalid}")
    return [matches[key][0] for key in requested]


def qc_pipeline_text(payload: bytes, *, full_input_set: bool) -> str:
    """Keep exact module bodies; add a probe only when no correction maps are used."""
    if sha256(payload) != PIPELINE_SHA256:
        raise ValueError(
            "Pipeline SHA-256 does not match pinned PedCP90 analysis.cppipe"
        )
    header, load, quality, flag, correction, _rest = payload.decode().split("\n\n", 5)
    if not full_input_set:
        fixture = SENTINEL_FIXTURE.read_text(encoding="utf-8")
        correction = "ImageMath:[" + fixture.split("\nImageMath:[", 1)[1].rstrip()
        correction = correction.replace("module_num:3|", "module_num:4|", 1)
        correction = correction.replace(
            "Select the image:DNA\n", "Select the image:OrigDNA\n"
        )
    return (
        "\n\n".join(
            (
                header.replace("ModuleCount:21", "ModuleCount:4"),
                load,
                quality,
                flag,
                correction,
            )
        )
        + "\n"
    )


def published_digests(row: dict[str, str]) -> dict[str, str]:
    """Bind published original-file MD5 values by filename, not channel alias."""
    digests = {
        name: row[f"MD5Digest_{key.removeprefix('FileName_')}"]
        for key, name in row.items()
        if key.startswith("FileName_Orig")
    }
    if not digests:
        raise ValueError(f"No published original-file digests for {field_id(row)}")
    return digests


def local_asset(
    row: dict[str, str], name: str, root: Path, published: dict[str, str]
) -> dict:
    """Verify and record a staged file before handing its path to CellProfiler."""
    key = f"FileName_{name}"
    url = row.get(f"URL_{name}", "")
    filename = row.get(key) or Path(unquote(urlsplit(url).path)).name
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"Expected a manifest basename for {name} in {field_id(row)}")
    path = root / filename
    payload = path.read_bytes()
    expected = published.get(filename)
    if (
        published
        and name.startswith("Orig")
        and (
            not expected
            or hashlib.md5(payload, usedforsecurity=False).hexdigest() != expected
        )
    ):
        raise ValueError(f"Published original-file MD5 mismatch for {path}")
    return {
        "path": str(path),
        "source": url or row.get(f"PathName_{name}"),
        "file_sha256": sha256(payload),
        "verified_published_md5": expected,
    }


def stage(args: argparse.Namespace) -> None:
    """Write a local LoadData CSV for the selected originals and hash all inputs."""
    requested = [(plate, well, site) for plate, well, site in args.field]
    if len({plate for plate, _, _ in requested}) != 1:
        raise ValueError("A flat image root must refer to exactly one plate")
    full_input_set = args.scope == "pedcp90-original"
    rows = select_rows(args.manifest, requested)
    qc_digests = (
        [published_digests(row) for row in select_rows(args.published_qc, requested)]
        if args.published_qc
        else [{} for _ in rows]
    )
    names = COMPLETE_INPUTS if full_input_set else ORIGINALS
    source = args.pipeline.read_bytes()
    projection = qc_pipeline_text(source, full_input_set=full_input_set)
    root = args.image_root.resolve()
    assets = [
        {name: local_asset(row, name, root, digests) for name in names}
        for row, digests in zip(rows, qc_digests)
    ]
    original_paths = [field[name]["path"] for field in assets for name in ORIGINALS]
    if len(original_paths) != len(set(original_paths)):
        raise ValueError("Manifest reuses an original path in the selected fields")
    args.run.mkdir(parents=True)
    (args.run / "qc.cppipe").write_text(projection, encoding="utf-8")
    with (args.run / "selected.csv").open("x", newline="", encoding="utf-8") as stream:
        columns = [*FIELD_COLUMNS, *(f"URL_{name}" for name in names)]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, field in zip(requested, assets):  # CP Python 3.9
            writer.writerow(
                {
                    **dict(zip(FIELD_COLUMNS, key)),
                    **{
                        f"URL_{name}": Path(asset["path"]).as_uri()
                        for name, asset in field.items()
                    },
                }
            )
    source_files = {"pipeline": args.pipeline, "manifest": args.manifest}
    if args.published_qc:
        source_files["published_qc"] = args.published_qc
    write_json(
        args.run / "inputs.json",
        {
            "scope": args.scope,
            "provenance": args.provenance,
            "source_files": {
                name: {"path": str(path.resolve()), "sha256": sha256(path.read_bytes())}
                for name, path in source_files.items()
            },
            "projection_sha256": sha256(projection.encode()),
            "selected_csv_sha256": sha256((args.run / "selected.csv").read_bytes()),
            "after_qc_outputs": CORRECTED if full_input_set else ("AfterQC",),
            "fields": [
                {"field": key, "assets": field} for key, field in zip(requested, assets)
            ],
        },
    )


def read_inputs(run: Path) -> dict:
    """Read the generated inputs and reject a changed pipeline or local CSV."""
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    for filename, key in (
        ("qc.cppipe", "projection_sha256"),
        ("selected.csv", "selected_csv_sha256"),
    ):
        if sha256((run / filename).read_bytes()) != inputs[key]:
            raise ValueError(f"{filename} changed after staging")
    return inputs


def native_observation(
    pipeline: Pipeline, run_dir: Path, field: dict, outputs: list[str]
) -> Observation:
    """Run the native pipeline with exactly the archived CP-decoded arrays."""
    images = {}
    scales = {}
    for name, asset in field["decoded"].items():
        pixels = np.load(run_dir / asset["file"], allow_pickle=False)
        if [list(pixels.shape), pixels.dtype.str, sha256(pixels.tobytes())] != [
            asset["shape"],
            asset["dtype"],
            asset["sha256"],
        ]:
            raise ValueError(f"Decoded input changed: {asset['file']}")
        images[name] = pixels
        scales[name] = asset["scale"]
    result = pipeline.run_detailed(images, scaling=scales)
    if result.measurements is None:
        raise ValueError("Native pipeline did not produce QC measurements")
    columns = result.measurements.features["Image"]
    raw_flag = np.asarray(columns[FLAG]).item()
    if raw_flag not in (0, 1):
        raise ValueError(f"Invalid native flag: {raw_flag}")
    produced = set(outputs) & result.derived_images.keys()
    if produced and produced != set(outputs):
        raise ValueError(
            f"Native pipeline produced only some downstream images: {produced}"
        )
    step = pipeline.flag_image_steps[0]
    flag = step.flags[0]
    failed = {
        criterion.measurement: replace(
            step, flags=(replace(flag, criteria=(criterion,)),)
        ).evaluate(
            {criterion.measurement: np.asarray(columns[criterion.measurement]).item()}
        )[1]
        for criterion in flag.criteria
    }
    return {
        "decision": Decision(int(raw_flag), result.skipped, bool(produced)).record(),
        "criteria_failed": failed,
        "metrics": {
            key: metric(np.asarray(value).item())
            for key, value in sorted(columns.items())
            if key.startswith("ImageQuality_")
        },
    }


def compare_observations(reference: Observation, native: Observation) -> dict:
    """Compare decisions and schemas exactly; report metric values separately."""
    ref_decision = Decision(**reference["decision"])
    got_decision = Decision(**native["decision"])
    ref_metrics, got_metrics = reference["metrics"], native["metrics"]
    ref_criteria, got_criteria = reference["criteria_failed"], native["criteria_failed"]
    return {
        "decision_mismatches": decision_differences(ref_decision, got_decision),
        "criterion_mismatches": sorted(
            k
            for k in ref_criteria.keys() | got_criteria.keys() | CRITERIA
            if ref_criteria.get(k) != got_criteria.get(k)
            or (k in CRITERIA) != (k in ref_criteria)
        ),
        "metric_schema_mismatches": sorted(ref_metrics.keys() ^ got_metrics.keys()),
        "criterion_values": {
            key: {"cellprofiler": ref_metrics[key], "native": got_metrics[key]}
            for key in sorted(ref_criteria.keys() & got_criteria.keys())
        },
    }


def compare(run: Path) -> int:
    """Replay the native pipeline and return nonzero on an exact mismatch."""
    # The reference interpreter only imports the shared staging/evidence helpers.
    from cp_anywhere import _native, load_pipeline  # noqa: PLC0415

    inputs = read_inputs(run)
    oracle = json.loads((run / "cellprofiler.json").read_text(encoding="utf-8"))
    if oracle["inputs_sha256"] != sha256((run / "inputs.json").read_bytes()):
        raise ValueError("Oracle inputs do not match staged inputs")
    if [f["field"] for f in inputs["fields"]] != [f["field"] for f in oracle["fields"]]:
        raise ValueError("Oracle field coverage does not match staged input fields")
    pipeline = load_pipeline(run / "qc.cppipe")
    reports = []
    for staged, field in zip(inputs["fields"], oracle["fields"]):
        if field["decoded"].keys() != staged["assets"].keys():
            raise ValueError(f"Decoded channel coverage differs for {field['field']}")
        reference = field["observation"]
        native = native_observation(pipeline, run, field, inputs["after_qc_outputs"])
        reports.append(
            {
                "field": field["field"],
                "cellprofiler": reference,
                "native": native,
                **compare_observations(reference, native),
            }
        )
    ok = all(
        not (
            row["decision_mismatches"]
            or row["criterion_mismatches"]
            or row["metric_schema_mismatches"]
        )
        for row in reports
    )
    write_json(
        run / "comparison.json",
        {
            "scope": inputs["scope"],
            "native_extension_sha256": sha256(Path(_native.__file__).read_bytes()),
            "exact_match": ok,
            "fields": reports,
        },
    )
    print(
        f"{len(reports)} fields; exact decision match={ok}; {run / 'comparison.json'}"
    )
    return 0 if ok else 1


def main() -> int:
    """Expose staging and comparison without installing CellProfiler in cp-anywhere."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    for option in ("run", "pipeline", "manifest", "image-root"):
        staging.add_argument(f"--{option}", required=True, type=Path)
    staging.add_argument(
        "--field",
        nargs=3,
        metavar=("PLATE", "WELL", "SITE"),
        action="append",
        required=True,
    )
    staging.add_argument(
        "--scope", choices=("pedcp90-original", "cpjump1-control"), required=True
    )
    staging.add_argument(
        "--published-qc", type=Path, help="Verify original-file MD5 by filename"
    )

    staging.add_argument("--provenance", action="append", default=[])
    commands.add_parser("compare").add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args)
        return 0
    return compare(args.run)


if __name__ == "__main__":
    raise SystemExit(main())

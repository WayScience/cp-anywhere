"""Set all package versions from the GitHub release tag before building.

Usage (run from the project root):

    python tools/sync_release_version.py v1.2.3          # rewrite Cargo.toml
    python tools/sync_release_version.py v1.2.3 --check  # verify only

Cargo, maturin, and wasm-pack all read the workspace version. For a
prerelease, maturin converts the SemVer spelling to Python's version format.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path

PACKAGES = ("cp-anywhere", "cp_anywhere_python", "cp-anywhere-wasm")
NUMBER = r"(?:0|[1-9][0-9]*)"
TAG_PATTERN = re.compile(
    rf"v?(?P<version>{NUMBER}\.{NUMBER}\.{NUMBER}(?:-(?:alpha|beta|rc)\.{NUMBER})?)"
)
WORKSPACE_PACKAGE = re.compile(r"(?m)^\[workspace\.package\][^\S\n]*$")
VERSION_LINE = re.compile(r'(?m)^(version\s*=\s*")[^"\n]+(")')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="release tag, for example v1.2.3")
    parser.add_argument("--check", action="store_true", help="verify without editing")
    args = parser.parse_args()

    version = version_from_tag(args.tag)
    if not args.check:
        sync_workspace_version(version)
        # Update local package versions without resolving every dependency anew.
        subprocess.run(["cargo", "update", "--workspace"], check=True)
    check_workspace_version(version)
    print(f"Release version {version}: {', '.join(PACKAGES)}")
    return 0


def version_from_tag(tag: str) -> str:
    match = TAG_PATTERN.fullmatch(tag)
    if not match:
        raise SystemExit(
            f"release tag must look like v1.2.3 or v1.2.3-rc.1 "
            f"(alpha, beta, or rc prerelease): {tag!r}"
        )
    return match.group("version")


def sync_workspace_version(version: str) -> None:
    manifest = Path("Cargo.toml")
    text = manifest.read_text(encoding="utf-8")
    header = WORKSPACE_PACKAGE.search(text)
    if header is None:
        raise SystemExit("missing [workspace.package] in Cargo.toml")
    start = header.end()
    rest = text[start:]
    next_header = re.search(r"(?m)^\[", rest)
    end = next_header.start() if next_header else len(rest)
    stanza, count = VERSION_LINE.subn(rf"\g<1>{version}\g<2>", rest[:end])
    if count != 1:
        raise SystemExit("expected one version in Cargo.toml [workspace.package]")
    manifest.write_text(text[:start] + stanza + rest[end:], encoding="utf-8")


def check_workspace_version(version: str) -> None:
    output = subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        text=True,
    )
    packages = {row["name"]: row["version"] for row in json.loads(output)["packages"]}
    lock = tomllib.loads(Path("Cargo.lock").read_text(encoding="utf-8"))
    locked = {
        row["name"]: row["version"] for row in lock["package"] if "source" not in row
    }
    for filename, versions in (("Cargo.toml", packages), ("Cargo.lock", locked)):
        for name in PACKAGES:
            found = versions.get(name, "missing")
            if found != version:
                raise SystemExit(f"{filename}: {name} is {found}, expected {version}")


if __name__ == "__main__":
    raise SystemExit(main())

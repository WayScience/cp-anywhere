"""Exercise release version sync with Cargo in a disposable workspace."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/sync_release_version.py"
PACKAGES = ("cp-anywhere", "cp_anywhere_python", "cp-anywhere-wasm")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a minimal copy of the release version inheritance layout."""
    (tmp_path / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = ["rust", "python", "wasm"]\n'
        '\n[workspace.package]\nversion = "0.1.0"\n'
    )
    for folder, name in zip(("rust", "python", "wasm"), PACKAGES, strict=True):
        crate = tmp_path / folder
        (crate / "src").mkdir(parents=True)
        (crate / "src/lib.rs").touch()
        (crate / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion.workspace = true\n'
        )
    subprocess.run(
        ["cargo", "generate-lockfile", "--offline"], cwd=tmp_path, check=True
    )
    return tmp_path


def _sync(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=workspace,
        env={**os.environ, "CARGO_NET_OFFLINE": "true"},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("version", ["1.7.12", "1.7.12-rc.2"])
def test_release_tag_updates_all_locked_packages(workspace: Path, version: str) -> None:
    result = _sync(workspace, f"v{version}")
    assert result.returncode == 0, result.stderr
    locked = tomllib.loads((workspace / "Cargo.lock").read_text())["package"]
    assert {row["name"]: row["version"] for row in locked} == dict.fromkeys(
        PACKAGES, version
    )
    result = _sync(workspace, f"v{version}", "--check")
    assert result.returncode == 0, result.stderr


def test_check_rejects_a_binding_with_a_different_version(workspace: Path) -> None:
    manifest = workspace / "wasm/Cargo.toml"
    manifest.write_text(
        manifest.read_text().replace("version.workspace = true", 'version = "0.2.0"')
    )
    result = _sync(workspace, "v0.1.0", "--check")
    assert result.returncode != 0
    assert "cp-anywhere-wasm" in result.stderr
    assert "0.2.0" in result.stderr


@pytest.mark.parametrize("tag", ["main", "v01.2.3", "v1.2.3+build", "v1.2.3-preview.1"])
def test_rejects_tags_without_a_shared_release_version(
    workspace: Path, tag: str
) -> None:
    manifest = workspace / "Cargo.toml"
    before = manifest.read_bytes()
    result = _sync(workspace, tag)
    assert result.returncode != 0
    assert "release tag must look like" in result.stderr
    assert manifest.read_bytes() == before

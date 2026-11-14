"""Check template identity, license retention, and agent metadata."""

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_project_identity_is_lowercase() -> None:
    """Keep the public distribution name distinct from the Python identifier."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "cp-anywhere"
    assert metadata["tool"]["maturin"]["module-name"] == "cp_anywhere._native"


def test_original_and_template_licenses_are_preserved() -> None:
    """Retain the original project notice and the generated code's notice."""
    assert "Copyright (c) 2026, Dave Bunten" in (ROOT / "LICENSE").read_text()
    notice = ROOT / "licenses/template-uv-python-research-software.txt"
    assert "Regents of the University of Colorado" in notice.read_text()


def test_repository_skills_have_parseable_frontmatter() -> None:
    """Codex requires real YAML metadata, not a Markdown approximation."""
    paths = list((ROOT / ".agents/skills").glob("*/SKILL.md"))
    assert paths
    for path in paths:
        match = re.match(r"\A---\n(.*?)\n---\n", path.read_text(), re.DOTALL)
        assert match is not None, path
        metadata = yaml.safe_load(match.group(1))
        assert metadata["name"] == path.parent.name
        assert isinstance(metadata["description"], str)
        assert metadata["description"]

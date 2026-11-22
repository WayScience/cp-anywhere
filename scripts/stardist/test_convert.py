"""Offline checks for the StarDist converter's input and output handling."""

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from scripts.stardist import convert


def test_rejects_corrupt_archive_even_with_extracted_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model.zip").write_bytes(b"corrupt archive")
    legacy_cache = tmp_path / convert.MODEL_NAME / convert.MODEL_NAME
    legacy_cache.mkdir(parents=True)
    (legacy_cache / "weights_best.h5").write_bytes(b"unverified weights")
    output = tmp_path / "export.onnx"
    monkeypatch.setattr(
        "sys.argv", ["convert.py", "--workdir", str(tmp_path), "--out", str(output)]
    )
    monkeypatch.setattr(convert, "convert", lambda *_: output)
    monkeypatch.setattr(convert, "verify", lambda *_: None)

    with pytest.raises(SystemExit, match=r"model\.zip.*SHA-256 mismatch"):
        convert.main()
    assert not output.exists()


@pytest.mark.parametrize("cached", [False, True])
def test_extracts_only_pinned_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    members = {
        "config.json": b"test config",
        "thresholds.json": b"test thresholds",
        "weights_best.h5": b"test weights",
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    zip_bytes = stream.getvalue()
    monkeypatch.setattr(
        convert, "ARCHIVE_SHA256", hashlib.sha256(zip_bytes).hexdigest()
    )
    archive_path = tmp_path / "model.zip"
    if cached:
        archive_path.write_bytes(zip_bytes)

    def fetch(url: str, *, timeout: int) -> io.BytesIO:
        assert not cached, "A verified cache hit must not use the network"
        assert url == convert.MODEL_URL
        assert timeout > 0
        return io.BytesIO(zip_bytes)

    monkeypatch.setattr(convert.urllib.request, "urlopen", fetch)
    old_model_dir = tmp_path / convert.MODEL_NAME / convert.MODEL_NAME
    old_model_dir.mkdir(parents=True)
    old_config = old_model_dir / "config.json"
    old_config.write_bytes(b"unverified old config")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    model_dir = convert.download_and_extract(tmp_path, run_dir)

    assert {p.name: p.read_bytes() for p in model_dir.iterdir()} == members
    assert old_config.read_bytes() == b"unverified old config"
    assert archive_path.read_bytes() == zip_bytes


def test_rejects_unverified_download_without_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        convert.urllib.request, "urlopen", lambda *_, **__: io.BytesIO(b"wrong bytes")
    )
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        convert.download_and_extract(tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())


def test_refuses_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "export.onnx"
    output.write_bytes(b"keep this model")
    monkeypatch.setattr(
        "sys.argv", ["convert.py", "--workdir", str(tmp_path), "--out", str(output)]
    )
    with pytest.raises(SystemExit):
        convert.main()
    assert "will not replace existing path" in capsys.readouterr().err
    assert output.read_bytes() == b"keep this model"

# scripts/instanseg

`convert.py` converts the pinned upstream InstanSeg
`single_channel_nuclei` TorchScript model (instanseg_models_v0.1.2
release, see `src/cp_anywhere/data/instanseg/NOTICE.md` for provenance,
hashes, licensing, and the exact upstream pins) into the two ONNX files
bundled at `src/cp_anywhere/data/instanseg/`. It is finished and
verified; do not re-run it to "refresh" the bundled exports without
updating the pinned SHA-256 hashes in that NOTICE and in
`src/cp_anywhere/autosegment.py`.

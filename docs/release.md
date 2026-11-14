# Release process

## What publishes where, and how

| Package                                          | Registry              | Workflow                                                          | Trigger              | Auth                                                                                                 |
| ------------------------------------------------ | --------------------- | ----------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `cp-anywhere` (sdist + abi3 wheels, 5 platforms) | PyPI                  | `.github/workflows/publish-pypi.yml`                              | `release: published` | OIDC trusted publishing (`pypa/gh-action-pypi-publish`)                                              |
| `cp_anywhere_core` (Rust crate)                  | crates.io             | `.github/workflows/publish-crate.yml`                             | `release: published` | `CARGO_REGISTRY_TOKEN` secret (crates.io trusted publishing is not GA yet; see `.github/zizmor.yml`) |
| `cp-anywhere-wasm` (npm package)                 | npm                   | `.github/workflows/publish-npm.yml`                               | `release: published` | OIDC trusted publishing (`id-token: write`, no stored token)                                         |
| Standalone `.wasm` + checksum                    | GitHub Release assets | `.github/workflows/publish-npm.yml` (`build_and_publish_npm` job) | `release: published` | `GITHUB_TOKEN`                                                                                       |

`tools/sync_release_version.py` rewrites the Cargo workspace `version` from
the release tag before any build; because all three crates
(`cp_anywhere_core`, `cp_anywhere_python`, `cp_anywhere_wasm`) use
`version.workspace = true`, this single rewrite keeps the PyPI wheel,
npm package, and crates.io crate at the same version with no separate
sync step.

Only `cp_anywhere_core` is published to crates.io: it is the portable,
dependency-free measurement kernel usable from any Rust project.
`cp_anywhere_python` and `cp_anywhere_wasm` are thin FFI bindings consumed
through PyPI and npm respectively, not as Cargo dependencies of other
projects.

## Cutting a release (manual steps)

1. Open a PR that updates `CHANGELOG`-equivalent notes if desired (this
   project uses `release-drafter` -- `.github/workflows/draft-release.yml`
   drafts release notes automatically from merged PR labels on every push
   to `main`; review and edit the draft before publishing).
1. Publish the drafted GitHub Release with a `vX.Y.Z` tag. This single
   `release: published` event fans out to all three publish workflows
   above, plus the `.wasm`/checksum release-asset upload.
1. Watch the four jobs in the Actions tab. PyPI and npm use
   environment-scoped (`environment: release`) trusted publishing, so no
   secret needs rotating for those two. crates.io needs a
   `CARGO_REGISTRY_TOKEN` configured once in the `release` environment's
   secrets.

## Verified locally, every M7 session (not a real publish)

Every artifact below was built and checked locally in this session with
the real toolchain -- no registry credentials involved, nothing uploaded:

```sh
# crates.io: packaging + a full publish dry run (compiles the packaged
# tarball from scratch, as crates.io would)
cargo publish --manifest-path crates/cp_anywhere_core/Cargo.toml --dry-run

# PyPI: sdist + wheel build, then install from the wheel into a throwaway
# venv with no project source on sys.path
uv build
python3 -m venv /tmp/smoke && /tmp/smoke/bin/pip install dist/*.whl numpy
/tmp/smoke/bin/python -c "import cp_anywhere; cp_anywhere.measure(...)"

# npm: wasm-pack build, then a full publish dry run
wasm-pack build crates/cp_anywhere_wasm --target web --release --out-dir pkg --out-name cp_anywhere
cd crates/cp_anywhere_wasm/pkg && npm publish --dry-run --access public
```

Results: `cargo publish --dry-run` packaged and compiled 7 files (46.5 KiB,
12.9 KiB compressed) with zero errors. `uv build` produced a working
abi3 wheel that installs and runs `measure()` correctly from a venv with no
project source path. `npm publish --dry-run` packaged 5 files (97.0 KiB
tarball, 224.2 KiB unpacked).

## Known gaps

- The publish workflows themselves have not been exercised by a real
  GitHub Actions run (that would require actually cutting a release,
  which this session does not do). The dry runs above verify the
  artifacts the workflows would build and upload are correct; they do not
  verify the workflow YAML executes successfully end to end on GitHub's
  runners.
- `CARGO_REGISTRY_TOKEN` and the `release` GitHub Environment (with its
  PyPI/npm trusted-publishing configuration) must be set up once, outside
  this repository, before the first real release.

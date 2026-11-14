# Release process

## What publishes where, and how

| Package                                          | Registry              | Workflow                                                          | Trigger              | Auth                                                                                                                                                                          |
| ------------------------------------------------ | --------------------- | ----------------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cp-anywhere` (sdist + abi3 wheels, 5 platforms) | PyPI                  | `.github/workflows/publish-pypi.yml`                              | `release: published` | OIDC trusted publishing (`pypa/gh-action-pypi-publish`); works from the first release via a PyPI "pending publisher"                                                          |
| `cp_anywhere_core` (Rust crate)                  | crates.io             | `.github/workflows/publish-crate.yml`                             | `release: published` | `CARGO_REGISTRY_TOKEN` secret for the first release (crates.io OIDC trusted publishing can only be configured for a crate that already exists); can migrate to OIDC afterward |
| `cp-anywhere-wasm` (npm package)                 | npm                   | `.github/workflows/publish-npm.yml`                               | `release: published` | OIDC trusted publishing (`id-token: write`, no stored token) -- **but only once the package already exists on npm**; see the bootstrap step below                             |
| Standalone `.wasm` + checksum                    | GitHub Release assets | `.github/workflows/publish-npm.yml` (`build_and_publish_npm` job) | `release: published` | `GITHUB_TOKEN`                                                                                                                                                                |

`tools/sync_release_version.py` rewrites the Cargo workspace `version` from
the release tag before any build; because all three crates
(`cp_anywhere_core`, `cp_anywhere_python`, `cp_anywhere_wasm`) use
`version.workspace = true`, this single rewrite keeps the PyPI wheel,
npm package, and crates.io crate at the same version with no separate
sync step. The git tag is the single source of truth for every published
version across all three registries.

Only `cp_anywhere_core` is published to crates.io: it is the portable,
dependency-free measurement kernel usable from any Rust project.
`cp_anywhere_python` and `cp_anywhere_wasm` are thin FFI bindings consumed
through PyPI and npm respectively, not as Cargo dependencies of other
projects.

## One-time setup (must be done once, outside this repo, before the first release)

None of the three packages (`cp-anywhere`, `cp_anywhere_core`,
`cp-anywhere-wasm`) have been published anywhere yet. Each registry has a
different bootstrap story for a brand-new, never-published package:

1. **GitHub "release" environment.** Already created
   (`repos/d33bs/cp-anywhere/environments/release`), referenced by all
   three publish workflows' `environment: release`. Optional hardening:
   add required reviewers under Settings -> Environments -> release.

1. **PyPI -- zero-touch, even for the first release.** PyPI supports a
   "pending publisher" that creates the project on first use:

   1. Sign in at <https://pypi.org/manage/account/publishing/>.
   1. Add a pending GitHub publisher: PyPI project name `cp-anywhere`,
      owner `d33bs`, repository `cp-anywhere`, workflow filename
      `publish-pypi.yml`, environment name `release`.
   1. Nothing else to do -- the first tagged release creates the project
      and publishes to it automatically.

1. **npm -- needs one manual publish first.** npm's trusted publishing
   has no "pending publisher" equivalent: a trusted publisher can only be
   attached to a package that already exists, so the very first publish
   cannot go through OIDC.

   1. Build the package once locally with the version you intend to tag,
      e.g. for `v0.1.0`:
      ```sh
      python tools/sync_release_version.py v0.1.0
      wasm-pack build crates/cp_anywhere_wasm --target web --release \
        --out-dir pkg --out-name cp_anywhere
      cd crates/cp_anywhere_wasm/pkg && npm publish --access public
      git checkout Cargo.toml Cargo.lock  # undo the local version bump
      ```
   1. Once `cp-anywhere-wasm` exists on npm, go to its package settings
      -> Publishing access -> add a trusted publisher: GitHub Actions,
      owner `d33bs`, repository `cp-anywhere`, workflow filename
      `publish-npm.yml`, environment `release`.
   1. Every release from that point on (including re-tagging `v0.1.0`'s
      GitHub Release, or any later tag) publishes automatically via OIDC,
      no token stored anywhere.

1. **crates.io -- needs a token for the first release.** Trusted
   publishing on crates.io reached general availability in 2025, but like
   npm, the trusted publisher can only be configured for a crate that
   already exists -- so the first publish needs a classic API token.

   1. Sign in at <https://crates.io/settings/tokens>, create a token
      scoped to `publish-new` (and `publish-update` for later releases).
   1. Add it as a `CARGO_REGISTRY_TOKEN` secret in the GitHub `release`
      environment (Settings -> Environments -> release -> Environment
      secrets). `publish-crate.yml` already reads this secret, so no
      workflow change is needed.
   1. Optional, after the crate exists: migrate `publish-crate.yml` from
      the stored token to OIDC (`rust-lang/crates-io-auth-action@v1`,
      the modern equivalent of this project's PyPI/npm setup) and delete
      the `CARGO_REGISTRY_TOKEN` secret. Not required -- the token-based
      workflow keeps working for every future release too.

## Cutting a release

1. Open a PR that updates `CHANGELOG`-equivalent notes if desired (this
   project uses `release-drafter` -- `.github/workflows/draft-release.yml`
   drafts release notes automatically from merged PR labels on every push
   to `main`; review and edit the draft before publishing).
1. Publish the drafted GitHub Release with a `vX.Y.Z` tag. This single
   `release: published` event fans out to all three publish workflows
   above, plus the `.wasm`/checksum release-asset upload. The tag is the
   only place a version is chosen; nothing else needs editing first.
1. Watch the four jobs in the Actions tab. After the one-time setup above,
   PyPI, npm, and crates.io all publish with no further manual steps,
   using the Cargo workspace version that `tools/sync_release_version.py`
   derived from the tag.

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
- The one-time setup steps above (PyPI pending publisher, npm bootstrap
  publish + trusted publisher, crates.io token) require the registry
  account owner's own login and have not been done yet -- they are not
  something a CI workflow or an automated tool can perform.

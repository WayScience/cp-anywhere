# Release process

## What publishes where, and how

| Package                                          | Registry              | Workflow                                                          | Trigger              | Auth                                                                                                                                                                          |
| ------------------------------------------------ | --------------------- | ----------------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cp-anywhere` (sdist + abi3 wheels, 5 platforms) | PyPI                  | `.github/workflows/publish-pypi.yml`                              | `release: published` | OIDC trusted publishing (`pypa/gh-action-pypi-publish`); works from the first release via a PyPI "pending publisher"                                                          |
| `cp-anywhere` (Rust crate)                       | crates.io             | `.github/workflows/publish-crate.yml`                             | `release: published` | `CARGO_REGISTRY_TOKEN` secret for the first release (crates.io OIDC trusted publishing can only be configured for a crate that already exists); can migrate to OIDC afterward |
| `cp-anywhere` (npm package)                      | npm                   | `.github/workflows/publish-npm.yml`                               | `release: published` | OIDC trusted publishing (`id-token: write`, no stored token) -- **but only once the package already exists on npm**; see the bootstrap step below                             |
| Standalone `.wasm` + checksum                    | GitHub Release assets | `.github/workflows/publish-npm.yml` (`build_and_publish_npm` job) | `release: published` | `GITHUB_TOKEN`                                                                                                                                                                |

The GitHub release tag is the only version you set. All three workflows
build the commit attached to that tag. `tools/sync_release_version.py`
updates the Cargo workspace and lockfile in each build checkout and
checks all three workspace packages before building. Do not bump versions
in source files or commit the generated changes.

For `v1.2.3`, all three registries get `1.2.3`. For prerelease tags such
as `v1.2.3-rc.1`, crates.io and npm get `1.2.3-rc.1` and PyPI gets
`1.2.3rc1`. Use `-alpha.N`, `-beta.N`, or `-rc.N` for prereleases.
npm publishes prereleases under `next` and stable releases under `latest`.

`cp-anywhere` is the only crate configured for crates.io publishing. It includes
segmentation, illumination, measurements, and `.cppipe` execution. To use
the individual operations without pipeline parsing or its serialization
dependencies, set `default-features = false` in your Cargo dependency.
`cp_anywhere_python` and `cp_anywhere_wasm` remain internal binding crates.
The Python and browser packages ship through PyPI and npm.

## One-time setup (must be done once, outside this repo, before the first release)

Each registry has different setup steps for a first release of its package:

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

   1. Use a disposable checkout. Publish a bootstrap prerelease so the
      GitHub workflow can publish the first stable version. For a first
      stable release of `v0.1.0`, use:
      ```sh
      python tools/sync_release_version.py v0.1.0-rc.1
      python scripts/wasm/build_package.py
      npm publish ./crates/cp_anywhere_wasm/pkg --access public --tag next
      ```
   1. Once `cp-anywhere` exists on npm, go to its package settings
      -> Publishing access -> add a trusted publisher: GitHub Actions,
      owner `d33bs`, repository `cp-anywhere`, workflow filename
      `publish-npm.yml`, environment `release`.
   1. Enable direct `npm publish` for the trusted publisher. Publish the
      first stable version through GitHub Releases, not by hand.

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
1. Check the three publishing workflows in the Actions tab. After registry
   setup, no per-release version edits or manual registry uploads are needed.

## Local packaging checks

Build and check the artifacts locally before cutting a release. These
commands do not upload anything:

```sh
# crates.io: build and verify the archive
cargo package --locked -p cp-anywhere

# PyPI: sdist + wheel build, then install from the wheel into a throwaway
# venv with no project source on sys.path
uv build
python3 -m venv /tmp/smoke && /tmp/smoke/bin/pip install dist/*.whl numpy
/tmp/smoke/bin/python -c "import cp_anywhere; cp_anywhere.measure(...)"

# npm: wasm-pack build, then a full publish dry run
python scripts/wasm/build_package.py
cd crates/cp_anywhere_wasm/pkg && npm publish --dry-run --access public
```

Use `--allow-dirty` with `cargo package` to check uncommitted changes locally.
The release workflow uses it because version sync modifies the checkout
before building.

## Known gaps

- The publish workflows themselves have not been exercised by a real
  GitHub Actions run (that would require actually cutting a release,
  which this session does not do). Local packaging checks do not verify
  registry authentication or publishing on GitHub's runners.
- The one-time setup steps above (PyPI pending publisher, npm bootstrap
  publish + trusted publisher, crates.io token) require the registry
  account owner's own login and have not been done yet -- they are not
  something a CI workflow or an automated tool can perform.

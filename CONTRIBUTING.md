# Contributing

Read the [code of conduct](https://github.com/wayscience/cp-anywhere/blob/main/CODE_OF_CONDUCT.md) before participating. Use [GitHub issues](https://github.com/wayscience/cp-anywhere/issues) for bugs and feature requests, and [private reporting](https://github.com/wayscience/cp-anywhere/blob/main/SECURITY.md) for security vulnerabilities.

## Set up the project

Install [Python](https://www.python.org/downloads/), [uv](https://docs.astral.sh/uv/getting-started/installation/), and [Rust](https://www.rust-lang.org/tools/install). Run these commands from the repository root:

```sh
uv sync --frozen --all-groups
uv tool install pre-commit
```

The Python package uses a compiled Rust extension. The sync step builds it with maturin. The `--all-groups` flag also installs the dependencies for the docs and browser fixture tools.

## Run local checks

```sh
uv run --no-sync poe pipeline
```

The pipeline runs the Rust tests, rebuilds the Python extension, runs the pre-commit checks, and runs pytest. For a narrower check, use `uv run --no-sync poe test` for Python tests or `uv run --no-sync poe lint` for the pre-commit checks.

## Build the docs

```sh
uv run --no-sync sphinx-build -W --keep-going -b html docs/src docs/_build/html
```

Open `docs/_build/html/index.html`. The build runs the example notebook and fails if any cell raises an error.

## Open a pull request

Keep the change focused. In the pull request, describe what changed and how you tested it. For changes to CellProfiler compatibility, include the reference pipeline or fixture and the comparison results.

## Versioning

Release builds take their version from the GitHub release tag. `tools/sync_release_version.py` copies that version into the Cargo workspace before the publish workflows build packages. See the [release guide](https://github.com/wayscience/cp-anywhere/blob/main/docs/release.md) for setup and release steps.

# Template provenance

cp-anywhere starts from CU-DBMI/template-uv-python-research-software.
The source revision is `23ac69fab29d229762ffd6c3741770292efd5d62`.
Its BSD-3-Clause notice is preserved in `licenses/template-uv-python-research-software.txt`.
The project keeps its original Dave Bunten BSD-3-Clause `LICENSE`.

Copier answers:

- project_name: `cp-anywhere`
- project_description: `CellProfiler-compatible measurements for native Python and web browsers`
- author_name: `Dave Bunten`
- author_orcid: `https://orcid.org/0000-0001-6041-3665`
- github_url: `https://github.com/d33bs/cp-anywhere`
- github_primary_reviewer: `@d33bs`
- use_rust: `true`
- formal_verification: `none`

The ORCID comes from the author's existing emojibench `CITATION.cff`.
No placeholder author identities were used.

The template baseline passed `uv sync` and `uv run --frozen pytest` (three tests).
`env -u CONDA_PREFIX uv run --frozen poe pipeline` passed two Rust tests and three Python tests.
The environment sets both `CONDA_PREFIX` and `VIRTUAL_ENV`; maturin requires that only one remains set.

The template's Markdown skills lacked YAML frontmatter, which Codex rejected.
The original text remains intact beneath added `name` and `description` frontmatter.
The mdformat hook includes `mdformat-frontmatter` to preserve that metadata.

The project, Rust public crate, Python distribution, and npm scope use lowercase names.
The Python import is `cp_anywhere`, following Python identifier rules.

`registry-checks.json` records actual namespace lookup responses.
A missing package is not a reservation or proof of npm scope ownership.
No packages or releases have been published.

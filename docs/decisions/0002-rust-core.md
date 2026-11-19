# 0002: Proceed with Rust core; keep the browser target open between WASM and a Pyodide hybrid

Status: accepted for M3. The browser target choice itself remains open
for M4; this decision is about the native core only.

The crate referred to here as `cp_anywhere_core` is now `cp-anywhere`,
with `cp_anywhere` as the Rust import name.

## Decision

Proceed with the Rust core built in M3. It implements the full declared
scope (`spec/compatibility.md`) and passes the original CellProfiler 4.2.8
golden fixtures exactly: 1,440 of 1,440 feature-and-relationship rows
across all 12 fixtures, with the only non-exact rows (25) explained by
float64 reduction-order tolerance already defined in
`spec/features.yaml` (see `docs/native.md`). The hard difficult-family
spike — same-set adjacent `MeasureObjectNeighbors`, including its 3D
two-axis quirk and its non-standard outline/dilation semantics — passed
without relaxing scope or tolerance. This clears the correctness
precondition this decision was gated on.

## Evidence for the browser-target question (informs M4, not decided here)

A bounded, real (non-mocked) Pyodide feasibility probe
(`/Users/buntend/.hermes/cache/scratch/cp-anywhere-pyodide-feasibility.md`)
ran the pinned `cp_measure==0.3.0` inside an actual Chromium module worker
via Pyodide 314.0.8:

- **Object intensity executes genuinely in Pyodide.** NumPy, SciPy, and
  scikit-image installed as real WASM wheels; `get_intensity(...)` ran on
  saved fixture arrays with real timings (1.6–4.9 ms per call) and
  independently verified output checksums.
- **Adjacent neighbors are blocked in Pyodide, not merely slow.**
  `centrosome` and `mahotas` have no WASM wheels. A real attempt to mount
  an unmodified, licensed `centrosome` source tree still failed at
  `centrosome._cpmorphology2`, a compiled C extension with no WASM build
  path attempted. No neighbor family compatibility claim is possible via
  cp_measure-on-Pyodide without a genuine from-scratch port of that
  family, which is exactly the hard spike this M3 Rust core already
  passed.
- Core Pyodide + scientific package payload is roughly 53 MB encoded
  before application code or fixtures, and cold start to "packages ready"
  was about 2.06 s locally. These are real measured numbers, not
  production guarantees (single run, uncompressed localhost, no
  cross-origin isolation).

This does not decide the browser architecture. It does establish that an
cp_measure-on-Pyodide path cannot reach full declared-scope browser parity
without a real neighbor-family implementation of its own — the same work
a `wasm32-unknown-unknown` build of `cp_anywhere_core` already requires
and has already passed on native. `wasm-pack 0.15.1` and
`wasm-bindgen 0.2.130` are installed and the `wasm32-unknown-unknown`
target is present on the build host; a WASM build of the existing,
already-passing Rust core is the straightforward next step for M4, with a
Pyodide-based intensity-only path remaining a possible fallback or hybrid
if the WASM build hits an unexpected blocker.

## Scope of this decision

This approves continuing with the Rust core as the measurement engine. It
does not approve claiming WASM feasibility, browser performance, or any
browser-target architecture — those remain open until M4 produces a real
browser build and M6 produces real cross-target benchmarks. It also does
not relax `spec/compatibility.md`'s declared scope or its parity gate: any
future feature family needs its own fixtures and its own full-scope
parity pass before a compatibility claim, exactly as `MeasureObjectIntensity`,
`MeasureImageIntensity`, and same-set `MeasureObjectNeighbors` did here.
The project remains BSD-3-Clause, with Dave Bunten as the license holder.

## Update: M4 resolved the open browser-target question

M4 built the straightforward next step named above: `crates/cp_anywhere_wasm`
compiles the same core to `wasm32-unknown-unknown` via `wasm-bindgen`, and a
real Playwright-driven Chromium run of a real Vite production build passed
1,320 of 1,320 golden-fixture rows across all 11 declared 2D fixtures (see
`docs/wasm.md`). The browser target is therefore the WASM build of this
Rust core, not a Pyodide hybrid — the Pyodide path's blocker identified
above (no WASM `centrosome`/`mahotas`) never needed to be worked around,
because the Rust core's own neighbor implementation already covers that
family. This still does not claim WASM performance parity or a
cross-browser guarantee; those remain M6 and future work respectively (see
`docs/wasm.md`'s Known limitations).

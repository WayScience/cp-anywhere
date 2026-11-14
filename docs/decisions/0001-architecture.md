# 0001: Scientific reference before measurement implementation

Status: accepted for M0/M1.

Use original CellProfiler 4.2.8 measurement modules as the numerical reference. Use `cp_measure` 0.3.0 as a comparison implementation and native performance baseline in M2. Keep reference packages outside the project runtime. The source and package pins are in `reference/source-lock.json` and `reference/requirements.lock`.

The first release scope has three modules: image intensity, object intensity, and same-set adjacent object neighbors. The object-neighbor family is the difficult spatial relationship case. The saved `.cppipe` holds the selected settings. The reference runner loads that pipeline and invokes each original module with real CellProfiler core images, objects, measurements, and workspaces. The runner saves the full emitted feature inventory and relationship rows. It does not copy measurement code into this project.

Keep the intended implementation core in Rust, independent of Python bindings and the browser. Make parity against every declared feature a release gate. Judge native speed against an equivalent `cp_measure` feature subset, using the benchmark weights fixed in `spec/benchmark-contract.yaml`. Measure browser budgets after a browser prototype exists.

This decision does not approve a full Rust rewrite. The difficult-family parity and WASM feasibility spikes must inform that decision in later milestones. The project remains BSD-3-Clause, with Dave Bunten as the license holder.

// Drives the worker from the page, runs a real in-browser parity check
// against all 2D golden fixtures, then a real timed benchmark on the
// four representative workloads from spec/benchmark-contract.yaml.
// Results are attached to `window.__CP_ANYWHERE_RESULTS__` for the
// Hermes-driven browser automation that verifies this demo end to end
// (see docs/wasm.md) -- this is not test-only scaffolding removed from
// production; a real UI reads the same worker messages (see #status).

declare global {
  interface Window {
    __CP_ANYWHERE_RESULTS__?: {
      parity: Record<string, string>;
      segmentation: Record<string, { object_count: number; threshold: number; labels: number[] }>;
      benchmark: Record<string, { raw_ms: number[] }>;
      memoryBytes: number | null;
      wasmTransferBytes: number | null;
    };
    __CP_ANYWHERE_DONE__?: boolean;
  }
}

const statusEl = document.querySelector<HTMLPreElement>("#status");

const BENCHMARK_FIXTURES = ["touching_gradient", "real_cell_cc0", "grid_medium", "grid_large"];

async function allFixtureNames(): Promise<string[]> {
  const response = await fetch("/fixtures/index.json");
  const payload = (await response.json()) as { fixtures: string[] };
  return payload.fixtures;
}

async function allSegmentationFixtureNames(): Promise<string[]> {
  const response = await fetch("/segmentation-fixtures/index.json");
  const payload = (await response.json()) as { fixtures: string[] };
  return payload.fixtures;
}

function wasmTransferBytesFallback(): number | null {
  const entry = performance
    .getEntriesByType("resource")
    .find((item) => item.name.endsWith(".wasm")) as PerformanceResourceTiming | undefined;
  return entry ? entry.encodedBodySize : null;
}

async function main(): Promise<void> {
  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onerror = (event: ErrorEvent) => {
    if (statusEl) statusEl.textContent = `worker error: ${event.message}`;
  };
  worker.onmessageerror = (event: MessageEvent) => {
    if (statusEl) statusEl.textContent = `worker message error: ${String(event.data)}`;
  };
  const fixtures = await allFixtureNames();
  const segmentationFixtures = await allSegmentationFixtureNames();
  const results: NonNullable<Window["__CP_ANYWHERE_RESULTS__"]> = {
    parity: {},
    segmentation: {},
    benchmark: {},
    memoryBytes: null,
    wasmTransferBytes: null,
  };
  window.__CP_ANYWHERE_RESULTS__ = results;

  worker.onmessage = (event: MessageEvent) => {
    const message = event.data as
      | { type: "parity_done"; results: Record<string, string>; memoryBytes: number | null; wasmTransferBytes: number | null }
      | { type: "segmentation_done"; results: Record<string, { object_count: number; threshold: number; labels: number[] }>; memoryBytes: number | null; wasmTransferBytes: number | null }
      | { type: "benchmark_done"; report: Record<string, { raw_ms: number[] }>; memoryBytes: number | null; wasmTransferBytes: number | null }
      | { type: "worker_error"; error: string };
    if (message.type === "worker_error") {
      if (statusEl) statusEl.textContent = `worker_error: ${message.error}`;
      window.__CP_ANYWHERE_DONE__ = true;
      return;
    }
    if (message.type === "parity_done") {
      results.parity = message.results;
      results.memoryBytes = message.memoryBytes;
      results.wasmTransferBytes = message.wasmTransferBytes ?? wasmTransferBytesFallback();
      if (statusEl) statusEl.textContent = `parity done: ${Object.keys(message.results).length} fixtures`;
      worker.postMessage({ type: "segmentation", fixtures: segmentationFixtures });
      return;
    }
    if (message.type === "segmentation_done") {
      results.segmentation = message.results;
      results.memoryBytes = message.memoryBytes ?? results.memoryBytes;
      results.wasmTransferBytes = message.wasmTransferBytes ?? results.wasmTransferBytes;
      if (statusEl) statusEl.textContent = `segmentation done: ${Object.keys(message.results).length} fixtures`;
      worker.postMessage({ type: "benchmark", fixtures: BENCHMARK_FIXTURES, warmups: 3, runs: 10 });
      return;
    }
    results.benchmark = message.report;
    results.memoryBytes = message.memoryBytes ?? results.memoryBytes;
    results.wasmTransferBytes = message.wasmTransferBytes ?? results.wasmTransferBytes;
    if (statusEl) statusEl.textContent = "done";
    window.__CP_ANYWHERE_DONE__ = true;
  };

  worker.postMessage({ type: "parity", fixtures });
}

const start = new URLSearchParams(location.search).has("batch-check")
  ? import("./batch-check").then(({ check }) => check())
  : main();

start.catch((error: unknown) => {
  if (statusEl) statusEl.textContent = `error: ${String(error)}`;
  throw error;
});

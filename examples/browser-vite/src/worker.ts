// Runs the real cp-anywhere WASM module inside a dedicated worker. Nothing
// here is mocked: the module is the same build produced by `wasm-pack
// build crates/cp_anywhere_wasm --target web --release`, and fixture
// inputs are the project's real saved golden-fixture arrays, flattened by
// `scripts/wasm/export_fixtures.py` (measurement) and
// `scripts/wasm/export_segmentation_fixtures.py` (segmentation).
import init, { identifyPrimaryObjects, measure } from "cp-anywhere-wasm";

interface FixtureMetadata {
  shape: number[];
  image_names: string[];
  object_names: string[];
  has_mask: boolean;
  image_id: string;
}

interface FlatFixture {
  shape: number[];
  image_names: string[];
  images: Float64Array;
  object_names: string[];
  objects: Uint32Array;
  mask: Uint8Array | null;
  image_id: string;
}

let wasmMemory: WebAssembly.Memory | null = null;

async function ensureInit(): Promise<void> {
  if (wasmMemory) return;
  const output = await init();
  wasmMemory = output.memory;
}

// Numeric arrays are fetched as raw binary (not JSON): a few fixtures
// contain NaN/Infinity pixel values, which strict JSON cannot represent.
// Typed-array views over the fetched bytes preserve the exact IEEE754 bit
// patterns and double as the "typed-array input" test the plan calls for.
async function loadFixture(name: string): Promise<FlatFixture> {
  const metaResponse = await fetch(`/fixtures/${name}.json`);
  if (!metaResponse.ok) throw new Error(`fixture metadata fetch failed: ${name}`);
  const meta: FixtureMetadata = await metaResponse.json();
  const images = new Float64Array(
    await (await fetch(`/fixtures/${name}.images.f64`)).arrayBuffer(),
  );
  const objects = new Uint32Array(
    await (await fetch(`/fixtures/${name}.objects.u32`)).arrayBuffer(),
  );
  const mask = meta.has_mask
    ? new Uint8Array(await (await fetch(`/fixtures/${name}.mask.u8`)).arrayBuffer())
    : null;
  return {
    shape: meta.shape,
    image_names: meta.image_names,
    images,
    object_names: meta.object_names,
    objects,
    mask,
    image_id: meta.image_id,
  };
}

function runOne(fixture: FlatFixture, families: string[]): string {
  return measure(
    new Uint32Array(fixture.shape),
    fixture.image_names,
    fixture.images,
    fixture.object_names,
    fixture.objects,
    fixture.mask ?? undefined,
    fixture.image_id,
    families,
  );
}

const FAMILIES = ["image_intensity", "object_intensity", "neighbors"];

interface SegmentationFixtureMetadata {
  shape: number[];
  settings: Record<string, unknown>;
  has_mask: boolean;
}

async function loadSegmentationFixture(
  name: string,
): Promise<{ shape: number[]; settings: Record<string, unknown>; pixels: Float64Array; mask: Uint8Array | null }> {
  const metaResponse = await fetch(`/segmentation-fixtures/${name}.json`);
  if (!metaResponse.ok) throw new Error(`segmentation fixture metadata fetch failed: ${name}`);
  const meta: SegmentationFixtureMetadata = await metaResponse.json();
  const pixels = new Float64Array(
    await (await fetch(`/segmentation-fixtures/${name}.pixels.f64`)).arrayBuffer(),
  );
  const mask = meta.has_mask
    ? new Uint8Array(await (await fetch(`/segmentation-fixtures/${name}.mask.u8`)).arrayBuffer())
    : null;
  return { shape: meta.shape, settings: meta.settings, pixels, mask };
}

function runSegmentationOne(fixture: {
  shape: number[];
  settings: Record<string, unknown>;
  pixels: Float64Array;
  mask: Uint8Array | null;
}): { object_count: number; threshold: number; labels: number[] } {
  const result = identifyPrimaryObjects(
    fixture.pixels,
    fixture.shape[0],
    fixture.shape[1],
    fixture.mask ?? undefined,
    JSON.stringify(fixture.settings),
  );
  return {
    object_count: result.objectCount,
    threshold: result.threshold,
    labels: Array.from(result.labels),
  };
}

function wasmTransferBytes(): number | null {
  const entry = self.performance
    .getEntriesByType("resource")
    .find((item) => item.name.endsWith(".wasm")) as PerformanceResourceTiming | undefined;
  return entry ? entry.encodedBodySize : null;
}

self.onmessage = async (event: MessageEvent) => {
  const message = event.data as
    | { type: "parity"; fixtures: string[] }
    | { type: "segmentation"; fixtures: string[] }
    | { type: "benchmark"; fixtures: string[]; warmups: number; runs: number };
  try {
    await ensureInit();

    if (message.type === "parity") {
      const results: Record<string, string> = {};
      for (const name of message.fixtures) {
        const fixture = await loadFixture(name);
        results[name] = runOne(fixture, FAMILIES);
      }
      self.postMessage({
        type: "parity_done",
        results,
        memoryBytes: wasmMemory?.buffer.byteLength ?? null,
        wasmTransferBytes: wasmTransferBytes(),
      });
      return;
    }

    if (message.type === "segmentation") {
      const results: Record<string, { object_count: number; threshold: number; labels: number[] }> = {};
      for (const name of message.fixtures) {
        const fixture = await loadSegmentationFixture(name);
        results[name] = runSegmentationOne(fixture);
      }
      self.postMessage({
        type: "segmentation_done",
        results,
        memoryBytes: wasmMemory?.buffer.byteLength ?? null,
        wasmTransferBytes: wasmTransferBytes(),
      });
      return;
    }

    const report: Record<string, { raw_ms: number[] }> = {};
    for (const name of message.fixtures) {
      const fixture = await loadFixture(name);
      for (let i = 0; i < message.warmups; i += 1) runOne(fixture, FAMILIES);
      const samples: number[] = [];
      for (let i = 0; i < message.runs; i += 1) {
        const start = performance.now();
        runOne(fixture, FAMILIES);
        samples.push(performance.now() - start);
      }
      report[name] = { raw_ms: samples };
    }
    self.postMessage({
      type: "benchmark_done",
      report,
      memoryBytes: wasmMemory?.buffer.byteLength ?? null,
      wasmTransferBytes: wasmTransferBytes(),
    });
  } catch (error) {
    self.postMessage({ type: "worker_error", error: String(error) });
  }
};

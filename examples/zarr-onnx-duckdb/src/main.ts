// Real browser integration: Zarr read (zarrita.js) -> ONNX Runtime Web
// inference (a reference/comparison mask) -> cp-anywhere-wasm's own
// IdentifyPrimaryObjects segmentation (CellProfiler-compatible
// declumping, not the ONNX demo mask) -> cp-anywhere-wasm measurement
// -> DuckDB-WASM Parquet export, validated afterward by native DuckDB
// in Python (scripts/zarr/validate_export.py). No stage here is mocked;
// see docs/browser-integration.md for scope and known limitations.

import * as duckdb from "@duckdb/duckdb-wasm";
import duckdbWorkerUrl from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import duckdbWasmUrl from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import * as ort from "onnxruntime-web/wasm";
import * as zarr from "zarrita";
import init, { identifyPrimaryObjects, measure } from "cp-anywhere-wasm";

declare global {
  interface Window {
    __CP_ANYWHERE_ZOD_DONE__?: boolean;
    __CP_ANYWHERE_ZOD_RESULT__?: Record<string, unknown>;
    __CP_ANYWHERE_ZOD_ERROR__?: string;
  }
}

const status = document.querySelector<HTMLPreElement>("#status")!;
const log = (message: string) => {
  status.textContent += `\n${message}`;
  // eslint-disable-next-line no-console
  console.log(message);
};

async function run(): Promise<void> {
  log("1. Reading OME-Zarr array with zarrita.js...");
  const store = new zarr.FetchStore(new URL("/dna.zarr", window.location.href).href);
  const array = await zarr.open.v2(store, { kind: "array" });
  const block = (await zarr.get(array)) as unknown as {
    data: Float64Array;
    shape: number[];
  };
  const shape = block.shape;
  const size = shape[0] * shape[1];
  log(`   shape=${JSON.stringify(shape)} pixels=${size}`);

  log("2. Running the real ONNX threshold model with onnxruntime-web (reference mask)...");
  const session = await ort.InferenceSession.create("/threshold.onnx", {
    executionProviders: ["wasm"],
  });
  const imageFloat32 = Float32Array.from(block.data);
  const inputTensor = new ort.Tensor("float32", imageFloat32, shape);
  const outputs = await session.run({ image: inputTensor });
  const maskInt32 = outputs.mask.data as Int32Array;
  const onnxForegroundPixels = maskInt32.reduce((acc, v) => acc + v, 0);
  log(`   onnx_foreground_pixels=${onnxForegroundPixels}`);
  await session.release();

  log("3. Segmenting with cp-anywhere-wasm's own IdentifyPrimaryObjects...");
  await init();
  const seg = identifyPrimaryObjects(
    block.data,
    shape[0],
    shape[1],
    undefined,
    JSON.stringify({
      threshold_method: "otsu",
      unclump_method: "intensity",
      watershed_method: "intensity",
      min_diameter: 4.0,
    }),
  );
  log(`   cp_anywhere_object_count=${seg.objectCount} threshold=${seg.threshold.toFixed(6)}`);

  log("4. Measuring with cp-anywhere-wasm (real Rust kernel)...");
  const resultJson = measure(
    Uint32Array.from(shape),
    ["DNA"],
    block.data,
    ["Cells"],
    seg.labels,
    undefined,
    "zarr-onnx-demo",
    ["object_intensity"],
  );
  const result = JSON.parse(resultJson) as {
    image_id: string;
    features: Record<string, Record<string, unknown>>;
    object_ids: Record<string, number[]>;
  };
  log(`   features for object 1: ${JSON.stringify(result.features.Cells)}`);

  log("5. Exporting cp-anywhere's measurement to Parquet via DuckDB-WASM...");
  const worker = new Worker(duckdbWorkerUrl, { type: "module" });
  const db = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
  await db.instantiate(duckdbWasmUrl);
  const conn = await db.connect();
  await conn.query("LOAD parquet");
  await conn.query(
    "CREATE TABLE cp_anywhere_features (image_id VARCHAR, scope VARCHAR, object_id INTEGER, feature VARCHAR, value DOUBLE)",
  );
  const objectIds = result.object_ids.Cells;
  const rows: string[] = [];
  for (const [feature, values] of Object.entries(result.features.Cells)) {
    const valueArray = values as number[];
    objectIds.forEach((objectId, index) => {
      rows.push(
        `('${result.image_id}', 'Cells', ${objectId}, '${feature}', ${valueArray[index]})`,
      );
    });
  }
  await conn.query(`INSERT INTO cp_anywhere_features VALUES ${rows.join(",")}`);
  await conn.query(
    "COPY cp_anywhere_features TO 'cp_anywhere_features.parquet' (FORMAT PARQUET, COMPRESSION UNCOMPRESSED)",
  );
  const parquetBytes = await db.copyFileToBuffer("cp_anywhere_features.parquet");
  await conn.close();
  await db.terminate();

  window.__CP_ANYWHERE_ZOD_RESULT__ = {
    shape,
    onnxForegroundPixels,
    cpAnywhereObjectCount: seg.objectCount,
    cpAnywhereThreshold: seg.threshold,
    measurement: result,
    parquetBytesLength: parquetBytes.length,
    parquetBase64: btoa(String.fromCharCode(...parquetBytes)),
  };
  window.__CP_ANYWHERE_ZOD_DONE__ = true;
  log("done.");
}

run().catch((error: unknown) => {
  const message = error instanceof Error ? `${error.message}\n${error.stack}` : String(error);
  window.__CP_ANYWHERE_ZOD_ERROR__ = message;
  window.__CP_ANYWHERE_ZOD_DONE__ = true;
  log(`ERROR: ${message}`);
});

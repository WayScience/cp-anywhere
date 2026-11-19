// Browser orchestration around the atomic WASM operations.
import type { Field, FieldResult, Reply } from "./batch-field";

/** Process complete fields with one WASM instance per worker, in input order. */
export async function runBatch(fields: readonly Field[], workers = 1): Promise<FieldResult[]> {
  if (!Number.isSafeInteger(workers) || workers < 1) throw new RangeError("workers must be a positive integer");
  const ids = new Set<string>();
  for (const { imageId } of fields) {
    if (typeof imageId !== "string" || !imageId || ids.has(imageId)) {
      throw new Error("image IDs must be nonempty and unique");
    }
    ids.add(imageId);
  }
  const pool: Worker[] = [];
  // Terminating a worker does not settle its outstanding message promise.
  const pending = new Map<Worker, (reason: Error) => void>();
  const results: FieldResult[] = [];
  let next = 0;
  async function consume(worker: Worker) {
    while (next < fields.length) {
      const index = next++;
      const field = fields[index];
      results[index] = await new Promise<FieldResult>((resolve, reject) => {
        const fail = (reason: unknown) => reject(new Error(`Image set ${JSON.stringify(field.imageId)}`, { cause: reason }));
        pending.set(worker, fail);
        worker.onmessage = ({ data }: MessageEvent<Reply>) => {
          pending.delete(worker);
          if (data.ok) resolve(data.result);
          else fail(data.error);
        };
        worker.onerror = (event) => fail(event.message || "worker failed");
        worker.onmessageerror = () => fail("could not deserialize worker reply");
        try {
          // Structured cloning preserves the caller's input buffers.
          worker.postMessage(field);
        } catch (error) {
          fail(error);
        }
      });
    }
  }
  try {
    for (let i = 0; i < Math.min(workers, fields.length); i++) {
      pool.push(new Worker(new URL("./batch.worker.ts", import.meta.url), { type: "module" }));
    }
    await Promise.all(pool.map(consume));
    return results;
  } finally {
    for (const worker of pool) worker.terminate();
    for (const reject of pending.values()) reject(new Error("batch stopped"));
    pending.clear();
  }
}

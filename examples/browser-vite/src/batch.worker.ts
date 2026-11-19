import init from "cp-anywhere";
import { runField, type Field, type Reply } from "./batch-field";

const worker = self as unknown as DedicatedWorkerGlobalScope;
let ready: ReturnType<typeof init> | undefined;
worker.onmessage = async ({ data }: MessageEvent<Field>) => {
  try {
    await (ready ??= init());
    const result = runField(data);
    worker.postMessage({ ok: true, result } satisfies Reply, [
      result.labels.buffer, result.smallRemovedLabels.buffer, result.uneditedLabels.buffer,
    ]);
  } catch (error) {
    worker.postMessage({ ok: false, error: String(error) } satisfies Reply);
  }
};

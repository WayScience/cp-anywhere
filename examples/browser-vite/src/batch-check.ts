// Real browser entry point for the cross-runtime check.
import { runBatch } from "./batch";
import type { Field, FieldResult } from "./batch-field";

interface Fixture {
  image_name: string;
  object_name: string;
  families: string[];
  settings: Field["settings"];
  fields: { image_id: string; shape: Field["shape"]; pixels: string }[];
}

declare global {
  interface Window {
    __batchCheck?: Promise<unknown>;
  }
}

function serialize({ labels, smallRemovedLabels, uneditedLabels, measurements, ...rest }: FieldResult) {
  return {
    ...rest,
    labels: Array.from(labels),
    smallRemovedLabels: Array.from(smallRemovedLabels),
    uneditedLabels: Array.from(uneditedLabels),
    measurements: JSON.parse(measurements),
  };
}

async function bytes(path: string) {
  const response = await fetch(`/batch-fixtures/${path}`);
  if (!response.ok) throw new Error(`Fixture fetch failed: ${path}`);
  return response;
}

async function run() {
  const fixture: Fixture = await (await bytes("manifest.json")).json();
  const fields: Field[] = await Promise.all(fixture.fields.map(async (field) => ({
    imageId: field.image_id,
    shape: field.shape,
    pixels: new Float64Array(await (await bytes(field.pixels)).arrayBuffer()),
    imageName: fixture.image_name,
    objectName: fixture.object_name,
    settings: fixture.settings,
    families: fixture.families,
  })));
  const serial = (await runBatch(fields, 1)).map(serialize);
  const parallel = (await runBatch(fields, 2)).map(serialize);
  if (JSON.stringify(serial) !== JSON.stringify(parallel)) {
    throw new Error("Browser serial and parallel outputs differ");
  }
  try {
    await runBatch([{ ...fields[0], imageId: "invalid-shape", shape: [0, 1] }, ...fields], 2);
    throw new Error("Invalid field unexpectedly succeeded");
  } catch (error) {
    if (!(error instanceof Error) || !error.message.includes('Image set "invalid-shape"')) throw error;
  }
  return parallel;
}

export function check(): void {
  window.__batchCheck = run();
  // Playwright awaits the same promise and reports the error.
  window.__batchCheck.catch(() => {});
}

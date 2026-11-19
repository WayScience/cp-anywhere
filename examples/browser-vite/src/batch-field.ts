// Atomic operation and transport types for the browser batch example.
import { identifyPrimaryObjects, measure } from "cp-anywhere";

export interface Field {
  imageId: string;
  shape: [number, number];
  pixels: Float64Array;
  imageName: string;
  objectName: string;
  settings: Record<string, unknown>;
  families: string[];
}

export interface FieldResult {
  imageId: string;
  labels: Uint32Array;
  smallRemovedLabels: Uint32Array;
  uneditedLabels: Uint32Array;
  objectCount: number;
  threshold: number;
  measurements: string;
}

export type Reply = { ok: true; result: FieldResult } | { ok: false; error: string };

/** Segment and measure one normalized grayscale field after WASM initialization. */
export function runField(field: Field): FieldResult {
  const [rows, cols] = field.shape;
  if (![rows, cols].every((n) => Number.isSafeInteger(n) && n > 0 && n <= 0xffffffff)
    || rows * cols !== field.pixels.length || field.pixels.length > 0xffffffff) {
    throw new RangeError("pixels must match a positive 2D shape");
  }
  const segmented = identifyPrimaryObjects(
    field.pixels, rows, cols, undefined, JSON.stringify(field.settings),
  );
  try {
    const labels = segmented.labels;
    return {
      imageId: field.imageId,
      labels,
      smallRemovedLabels: segmented.smallRemovedLabels,
      uneditedLabels: segmented.uneditedLabels,
      objectCount: segmented.objectCount,
      threshold: segmented.threshold,
      measurements: measure(
        new Uint32Array(field.shape), [field.imageName], field.pixels,
        [field.objectName], labels, undefined, field.imageId, field.families,
      ),
    };
  } finally {
    segmented.free();
  }
}

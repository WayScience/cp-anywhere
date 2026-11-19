// Compare both WASM entry points with pinned CP 4.2.8 float32 reductions.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const artifacts = resolve(process.argv[2] ?? 'crates/cp_anywhere_wasm/pkg');
const wasm = await import(pathToFileURL(`${artifacts}/cp_anywhere.js`));
await wasm.default({ module_or_path: await readFile(`${artifacts}/cp_anywhere_bg.wasm`) });
const pixels = Array.from({ length: 17 * 23 }, (_, i) => Math.fround(Math.fround((i * 29) % 251) / Math.fround(255)));
const input = {
  shape: [17, 23], images: { DNA: pixels }, objects: { Cells: pixels.map(() => 0) },
  image_id: 'CP-4.2.8', families: ['image_intensity'], float32_images: ['DNA'],
};
const json = JSON.parse(wasm.measureJson(JSON.stringify(input)));
const typed = JSON.parse(wasm.measureWithOptions(
  Uint32Array.from(input.shape), ['DNA'], Float64Array.from(pixels),
  ['Cells'], Uint32Array.from(input.objects.Cells), undefined,
  input.image_id, input.families, [], [], [], new Uint32Array(),
  undefined, undefined, undefined, ['DNA'],
));
assert.deepEqual(typed, json);
const columns = json.features.Image;
for (const [name, expected] of Object.entries({
  TotalIntensity: 190.15293884277344,
  MeanIntensity: 0.48632465177179907,
  StdIntensity: 0.2849389612674713,
  MedianIntensity: 0.48627451062202454,
  MADIntensity: 0.24705882370471954,
})) {
  const actual = columns[`Intensity_${name}_DNA`];
  assert.ok(Math.abs(actual - expected) <= 1e-10 + 1e-9 * Math.abs(expected), `${name}: ${actual} != ${expected}`);
}
console.log('Pinned CP float32 reductions match WASM JSON and typed-array APIs.');

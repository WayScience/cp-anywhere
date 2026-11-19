import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const artifacts = resolve(process.argv[2] ?? 'crates/cp_anywhere_wasm/pkg');
const wasm = await import(pathToFileURL(`${artifacts}/cp_anywhere.js`));
await wasm.default({ module_or_path: await readFile(`${artifacts}/cp_anywhere_bg.wasm`) });
const fixture = (name) => readFile(new URL(`../../fixtures/neighbors_distance/${name}`, import.meta.url), 'utf8');

function compare(actual, expected, imageId = actual.image_id) {
  assert.deepEqual(actual.object_ids, expected.object_ids);
  assert.deepEqual(actual.relationships.map((row) => [row.object_name, row.first_object_id, row.second_object_id]), expected.relationships);
  for (const row of actual.relationships) {
    assert.equal(row.first_image_id, imageId);
    assert.equal(row.second_image_id, imageId);
  }
  assert.deepEqual(Object.keys(actual.features).sort(), Object.keys(expected.features).sort());
  for (const [scope, columns] of Object.entries(expected.features)) {
    assert.deepEqual(Object.keys(actual.features[scope]).sort(), Object.keys(columns).sort());
    for (const [name, values] of Object.entries(columns)) {
      const output = actual.features[scope][name];
      assert.equal(output.length, values.length, name);
      for (const [i, value] of values.entries()) {
        if (typeof value !== 'number' || name.includes('Number')) {
          assert.deepEqual(output[i], value, name);
        } else {
          assert.ok(typeof output[i] === 'number' && Math.abs(output[i] - value) <= 1e-12 * (1 + Math.abs(value)), `${name}[${i}]: ${output[i]} != ${value}`);
        }
      }
    }
  }
}

const cases = JSON.parse(await fixture('cases.json'));
for (const expected of cases) {
  const labels = expected.labels.flat();
  const shape = [expected.labels.length, expected.labels[0].length];
  const dna = labels.map(() => 0);
  const full = expected.full_labels?.flat().map((id, i) => expected.excluded || labels[i] ? id : 0);
  const distances = { Cells: expected.distance };
  const input = {
    shape,
    images: { DNA: dna },
    objects: { Cells: labels },
    image_id: 'Example',
    families: ['neighbors'],
    neighbors_distances: distances,
    ...(full ? { neighbors_full_objects: { Cells: full } } : {}),
  };
  const fromJson = JSON.parse(wasm.measureJson(JSON.stringify(input)));
  const fromArrays = JSON.parse(wasm.measureWithOptions(
    Uint32Array.from(shape), ['DNA'], Float64Array.from(dna), ['Cells'], Uint32Array.from(labels),
    undefined, 'Example', ['neighbors'], [], [], full ? ['Cells'] : [], Uint32Array.from(full ?? []),
    undefined, undefined, JSON.stringify(distances),
  ));
  compare(fromJson, expected);
  assert.deepEqual(fromArrays, fromJson);
}

const text = await fixture('primary_mixed.cppipe');
const expected = JSON.parse(await fixture('primary_mixed.json'));
const pipeline = new wasm.CppipePipeline(text);
try {
  assert.deepEqual(JSON.parse(pipeline.plan()).measurement_steps.map((step) => step.neighbors_distance), [5, 1, 1, null, 6]);
  const shape = [expected.pixels.length, expected.pixels[0].length];
  const images = JSON.stringify([{ name: 'DNA', shape, dtype: 'uint16' }]);
  const result = JSON.parse(pipeline.run(images, Float64Array.from(expected.pixels.flat())));
  for (const [name, layers] of Object.entries(expected.outputs)) {
    assert.deepEqual(result.primary[name].labels, layers.labels.flat());
    assert.deepEqual(result.primary[name].small_removed_labels, layers.full_labels.flat());
  }
  compare(result.measurements, expected, 'Image');
} finally {
  pipeline.free();
}

console.log(`Matched ${cases.length} fixed-distance cases via both WASM array APIs and the mixed .cppipe oracle.`);

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const artifacts = resolve(process.argv[2] ?? 'crates/cp_anywhere_wasm/pkg');
const wasm = await import(pathToFileURL(`${artifacts}/cp_anywhere.js`));
await wasm.default({ module_or_path: await readFile(`${artifacts}/cp_anywhere_bg.wasm`) });

const fixture = (path) => readFile(new URL(`../../fixtures/${path}`, import.meta.url), 'utf8');
const split = (text) => text.trim().split('\n\n');
const pipeline = (...blocks) => `CellProfiler Pipeline:\n\n${blocks.join('\n\n')}\n`;
const [, primary, intensity, , neighbors] = split(await fixture('cppipe/manual.cppipe'));
const [, quality] = split(await fixture('cppipe/image_quality/quality_flag.cppipe'));
const [, sizeShape] = split(await fixture('size_shape/rectangle.cppipe'));
const [, secondaryPrimary, secondary, secondaryNeighbors] = split(
  await fixture('cppipe/secondary_regressions/pipeline_border_distance_n.cppipe'),
);
const excluded = neighbors.replace('touching image border?:No', 'touching image border?:Yes');
const allLoaded = quality.replace('images?:Select...', 'images?:All loaded images');
const text = pipeline('RunCellpose:[variable_revision_number:3]\n    Name the output object:NotCells',
  allLoaded, intensity, excluded);
const shape = [8, 8];
const image = { name: 'DNA', shape, dtype: 'float64', scaling: 2 };
const kept = { name: 'Cells', shape, dtype: 'uint32', role: 'objects' };
const diagnostics = { ...kept, role: 'object_diagnostics' };
const dna = Array(64).fill(0.25);
const labels = dna.map((_, i) => [18, 19, 26, 27].includes(i) ? 1 : 0);
const full = labels.map((id, i) => [20, 21, 28, 29].includes(i) ? 2 : id);
const reject = (fn, pattern) => assert.throws(fn, (error) => pattern.test(String(error)), String(pattern));
const pack = (...planes) => Float64Array.from(planes.flat());
const once = (text, descriptors, ...planes) => JSON.parse(
  wasm.run_cppipe(text, JSON.stringify(descriptors), pack(...planes)),
);

const runner = new wasm.CppipePipeline(text);
try {
  const run = (descriptors, ...planes) => JSON.parse(runner.run(JSON.stringify(descriptors), pack(...planes)));
  assert.ok(JSON.parse(runner.plan()).skipped_modules.includes('RunCellpose'));
  reject(() => run([image], dna), /MeasureObjectIntensity needs object set "Cells"/);
  reject(() => run([image, kept], dna, labels), /MeasureObjectNeighbors.*"Cells".*object_diagnostics/);
  const result = run([kept, image, diagnostics], labels, dna, full);
  assert.deepEqual(result, once(text, [kept, image, diagnostics], labels, dna, full));
  assert.deepEqual(result.measurements.object_ids.Cells, [1]);
  assert.deepEqual(result.measurements.features.Cells.Neighbors_NumberOfNeighbors_Adjacent, [1]);
  assert.equal(result.measurements.features.Image.ImageQuality_Scaling_DNA, 2);
  assert.ok(Object.keys(result.measurements.features.Image).every((name) => !name.endsWith('_Cells')));
  assert.deepEqual(result.primary, {});

  for (const [descriptors, planes, pattern] of [
    [[image, diagnostics], [dna, full], /object_diagnostics.*Cells.*objects/],
    [[image, kept, kept, diagnostics], [dna, labels, labels, full], /duplicate.*Cells/],
    [[image, kept, diagnostics, diagnostics], [dna, labels, full, full], /duplicate.*Cells/],
    [[image, kept, { ...diagnostics, shape: [4, 16] }], [dna, labels, full], /Cells.*same shape/],
    [[image, { ...kept, scaling: 2 }, diagnostics], [dna, labels, full], /scaling.*image/],
    [[image, { ...kept, dtype: 'float64' }, diagnostics], [dna, labels, full], /Cells.*integers/],
    [[image, { ...kept, role: 'labels' }, diagnostics], [dna, labels, full], /invalid image descriptors/],
    [[image, kept, diagnostics], [dna, labels, full.slice(1)], /insufficient pixels.*Cells/],
    [[image, kept, diagnostics], [dna, labels, [...full, 0]], /trailing values/],
  ]) {
    reject(() => run(descriptors, ...planes), pattern);
  }
} finally {
  runner.free();
}

reject(() => once(pipeline(primary, excluded), [image, kept, diagnostics], dna, labels, full),
  /Cells.*produced by IdentifyPrimaryObjects/);
reject(() => once(pipeline(primary, excluded.replaceAll(':Cells', ':Missing')),
  [{ ...image, shape: [1, ...shape] }], dna), /MeasureObjectNeighbors needs object set "Missing"/);

// Legacy object-only fixture: no role on the descriptor.
assert.deepEqual(once(pipeline(sizeShape), [{ name: 'Cells', shape, dtype: 'uint32' }], labels)
  .measurements.features.Cells.AreaShape_Area, [4]);
assert.deepEqual(once(pipeline(sizeShape), [kept], labels).measurements.features.Cells.AreaShape_Area, [4]);

// Seed with kept external labels; the diagnostic is not an unedited seed plane.
const stain = { ...image, name: 'Stain' };
const nuclei = { ...kept, name: 'Nuclei' };
const cells = once(pipeline(secondary, secondaryNeighbors), [stain, nuclei], dna, labels).secondary.Cells;
assert.ok(cells.labels.every((id) => id === 1));
assert.deepEqual(once(pipeline(secondary, secondaryNeighbors),
  [stain, nuclei, { ...nuclei, role: 'object_diagnostics' }], dna, labels, full).secondary.Cells, cells);
reject(() => once(pipeline(secondaryPrimary, secondary), [image, stain, nuclei], dna, dna, labels),
  /Nuclei.*produced by IdentifyPrimaryObjects/);

console.log('WASM explicit object labels, diagnostics, preflight, QC, external seeds, and legacy input passed.');

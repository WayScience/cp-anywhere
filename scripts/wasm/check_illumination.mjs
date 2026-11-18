// Execute the release web-target WASM against independent real-CP fixtures.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import init, * as wasm from '../../crates/cp_anywhere_wasm/pkg/cp_anywhere.js';

const root = new URL('../../', import.meta.url);
const load = async (path) => JSON.parse(await readFile(new URL(path, root), 'utf8'));
await init({ module_or_path: await readFile(new URL('crates/cp_anywhere_wasm/pkg/cp_anywhere_bg.wasm', root)) });

function compare(actual, expected, name) {
  assert.ok(actual instanceof Float64Array, `${name}: output must be Float64Array`);
  const flat = expected.flat();
  assert.equal(actual.length, flat.length, `${name}: output length`);
  for (let i = 0; i < flat.length; i++) {
    const want = flat[i];
    const got = actual[i];
    // Identical to tests/test_illumination.py: |a-b| <= atol + rtol*|b|.
    assert.ok(Number.isNaN(want) ? Number.isNaN(got) :
      got === want || Math.abs(got - want) <= 1e-6 + 1e-6 * Math.abs(want),
    `${name}[${i}]: ${got} != ${want}`);
  }
  return flat.length;
}

async function checkCalculate() {
  assert.equal(typeof wasm.calculateIllumination, 'function', 'calculateIllumination export missing');
  const manifest = await load('fixtures/illumination_calculate/manifest.json');
  let scalars = 0;
  let masked = 0;
  let defaultsChecked = false;
  for (const { name } of manifest.cases) {
    const f = await load(`fixtures/illumination_calculate/${name}.json`);
    const pixels = Float64Array.from(f.pixels.flat());
    const before = pixels.slice();
    const mask = f.mask === null ? undefined : Uint8Array.from(f.mask.flat(), v => v ? 255 : 0);
    const beforeMask = mask?.slice();
    const actual = wasm.calculateIllumination(pixels, f.pixels.length, f.pixels[0].length, mask, JSON.stringify(f.settings));
    scalars += compare(actual, f.expected, name);
    assert.deepEqual(pixels, before, `${name}: input was mutated`);
    assert.deepEqual(mask, beforeMask, `${name}: mask was mutated`);
    const s = f.settings;
    if (mask === undefined && s.intensity_choice === 'Regular' && !s.dilate_objects &&
        s.smoothing_method === 'Gaussian Filter' && s.automatic_object_width === 'Automatic' && s.rescale_option === 'Yes') {
      compare(wasm.calculateIllumination(pixels, f.pixels.length, f.pixels[0].length, undefined, '{}'), f.expected, `${name}: defaults`);
      defaultsChecked = true;
    }
    masked += Number(mask !== undefined);
  }
  assert.ok(defaultsChecked, 'default calculation settings checked against independent CP oracle');
  assert.ok(masked > 0, 'oracle cases must exercise masks');
  return { cases: manifest.cases.length, masked_cases: masked, scalars };
}

const decode = (v) => typeof v === 'object' ?
  ({ NaN, '+Infinity': Infinity, '-Infinity': -Infinity })[v.nonfinite] : v;

async function checkApply() {
  assert.equal(typeof wasm.applyIllumination, 'function', 'applyIllumination export missing');
  const manifest = await load('fixtures/illumination_apply/manifest.json');
  let scalars = 0;
  for (const name of manifest.cases) {
    const f = await load(`fixtures/illumination_apply/${name}.json`);
    const pixels = Float64Array.from(f.image.flat());
    const fn = Float64Array.from(f.illum.flat());
    const before = [pixels.slice(), fn.slice()];
    const actual = wasm.applyIllumination(pixels, fn, f.image.length, f.image[0].length, JSON.stringify(f.settings));
    scalars += compare(actual, f.expected, name);
    for (const value of actual) assert.equal(value, Math.fround(value), `${name}: float32 round trip`);
    assert.deepEqual([pixels, fn], before, `${name}: inputs mutated`);
    if (name === 'divide_tl1_th1') {
      assert.deepEqual(wasm.applyIllumination(pixels, fn, f.image.length, f.image[0].length, '{}'), actual);
    }
  }
  const edges = await load('crates/cp_anywhere_wasm/tests/fixtures/illumination_apply_edges.json');
  let edgeScalars = 0;
  for (const f of edges.cases) {
    const actual = wasm.applyIllumination(Float64Array.from(f.image, decode), Float64Array.from(f.illum, decode), f.rows, f.cols, JSON.stringify(f.settings));
    // Exact float32-round-tripped real CP oracle, including NaN and infinity.
    assert.deepEqual(Array.from(actual), f.expected.map(decode), JSON.stringify(f.settings));
    edgeScalars += actual.length;
  }
  return { cases: manifest.cases.length, scalars, edge_cases: edges.cases.length, exact_edge_scalars: edgeScalars };
}

function checkValidation() {
  const pixels = Float64Array.of(0.25, 0.5, 0.75, 1);
  const fn = Float64Array.of(1, 1, 1, 1);
  let cases = 0;
  const rejects = (call, message) => {
    assert.throws(call, error => {
      assert.ok(!(error instanceof WebAssembly.RuntimeError), `must reject, not trap: ${error}`);
      return message.test(String(error));
    });
    cases++;
  };
  for (const [rows, cols] of [[0, 4], [-1, 4], [1.5, 2], [NaN, 4], [Infinity, 4], [1, -1], [2 ** 32 + 2, 2], [65536, 65536], [1, 3]]) {
    rejects(() => wasm.calculateIllumination(pixels, rows, cols, undefined, '{}'), /dimensions|length/);
    rejects(() => wasm.applyIllumination(pixels, fn, rows, cols, '{}'), /dimensions|length/);
  }
  rejects(() => wasm.calculateIllumination(new Float64Array(), 0, 0, undefined, '{}'), /dimensions/);
  rejects(() => wasm.applyIllumination(pixels, fn.subarray(0, 3), 2, 2, '{}'), /equal dimensions/);
  for (const mask of [new Uint8Array(), new Uint8Array(3), new Uint8Array(5)]) {
    rejects(() => wasm.calculateIllumination(pixels, 2, 2, mask, '{}'), /mask/);
  }
  for (const config of ['{', '[]', 'null', '{"typo":true}', '{"dilate_objects":1}', '{"object_width":"10"}', '{"block_size":-1}', '{"block_size":2.5}', '{"object_dilation_radius":null}']) {
    rejects(() => wasm.calculateIllumination(pixels, 2, 2, undefined, config), /config/);
  }
  for (const key of ['intensity_choice', 'smoothing_method', 'automatic_object_width', 'rescale_option']) {
    rejects(() => wasm.calculateIllumination(pixels, 2, 2, undefined, JSON.stringify({ [key]: 'invalid' })), new RegExp(key));
  }
  for (const method of ['Convex Hull', 'Splines']) {
    rejects(() => wasm.calculateIllumination(pixels, 2, 2, undefined, JSON.stringify({ smoothing_method: method })), /not yet implemented/);
  }
  for (const config of ['{', '[]', 'null', '{"typo":true}', '{"truncate_low":1}', '{"truncate_high":null}', '{"divide_or_subtract":4}']) {
    rejects(() => wasm.applyIllumination(pixels, fn, 2, 2, config), /config/);
  }
  rejects(() => wasm.applyIllumination(pixels, fn, 2, 2, '{"divide_or_subtract":"divide"}'), /divide_or_subtract/);
  const retainedExports = ['measure', 'measureWithOptions', 'measureJson', 'identifyPrimaryObjects', 'identifyPrimaryObjectsJson', 'measureGranularityJson', 'measureIntensityDistributionJson', 'measureTextureJson', 'measureColocalizationJson'];
  for (const name of retainedExports) {
    assert.equal(typeof wasm[name], 'function', `${name}: existing export retained`);
  }
  return { rejected_cases: cases, retained_exports: retainedExports.length };
}

const report = { runtime: 'Node WebAssembly, release web-target module', calculation: await checkCalculate(), application: await checkApply(), validation: checkValidation() };
console.log(JSON.stringify(report, null, 2));

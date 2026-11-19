// Run the same Rust engine as the Python notebook, with no backend service.
const demo = document.querySelector('#wasm-demo');
const status = demo.querySelector('[role=status]');
const results = demo.querySelector('[data-results]');
const correction = demo.querySelector('[name=correction]');
const channelSelect = demo.querySelector('[name=channel]');
const asset = (name) => new URL(`browser-demo/${name}`, import.meta.url);
const size = 128;
const count = size * size;

async function fetchAsset(name) {
  const response = await fetch(asset(name));
  if (!response.ok) throw new Error(`Could not load ${name} (HTTP ${response.status}).`);
  return response;
}

async function loadChannel(name) {
  const bytes = new Uint8Array(await (await fetchAsset(`${name.toLowerCase()}.u8`)).arrayBuffer());
  if (bytes.length !== count) throw new Error(`Unexpected dimensions for ${name}.`);
  return Float64Array.from(bytes, (value) => value / 255);
}

function boundary(labels, i) {
  const id = labels[i];
  if (!id) return false;
  const x = i % size;
  const y = Math.floor(i / size);
  return x === 0 || x === size - 1 || y === 0 || y === size - 1 ||
    labels[i - 1] !== id || labels[i + 1] !== id || labels[i - size] !== id || labels[i + size] !== id;
}

function draw(canvas, pixels, maximum, labels) {
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(size, size);
  for (let i = 0; i < count; i++) {
    const offset = i * 4;
    if (labels && boundary(labels, i)) {
      image.data.set([0, 255, 128, 255], offset);
    } else {
      const gray = Math.round(255 * pixels[i] / maximum);
      image.data.set([gray, gray, gray, 255], offset);
    }
  }
  ctx.putImageData(image, 0, 0);
}

function fail(error) {
  results.hidden = true;
  status.textContent = `Demo unavailable: ${error.message ?? error}`;
}

async function main() {
  const [wasm, dna, actin, pipeline] = await Promise.all([
    import(asset('cp_anywhere.js')),
    loadChannel('DNA'),
    loadChannel('Actin'),
    fetchAsset('demo.cppipe').then((response) => response.text()),
  ]);
  await wasm.default();
  const flat = Float64Array.from({ length: count }, (_, i) => 0.25 + 0.75 * (i % size) / (size - 1));
  const map = wasm.calculateIllumination(flat, size, size, undefined, JSON.stringify({
    smoothing_method: 'Fit Polynomial', rescale_option: 'No',
  }));
  const apply = JSON.stringify({ divide_or_subtract: 'Divide' });
  const channels = Object.fromEntries([['DNA', dna], ['Actin', actin]].map(([name, original]) => {
    const shaded = original.map((value, i) => value * flat[i]);
    const corrected = wasm.applyIllumination(shaded, map, size, size, apply);
    return [name, { original, shaded, corrected, maximum: Math.max(...original) || 1 }];
  }));
  const descriptors = JSON.stringify(['DNA', 'Stain'].map((name) => ({ name, shape: [size, size], dtype: 'float64' })));
  let csvUrl;

  function compare() {
    const name = channelSelect.value;
    const channel = channels[name];
    demo.querySelector('[data-channel]').textContent = name;
    for (const key of ['shaded', 'corrected', 'original']) {
      draw(demo.querySelector(`[data-image="${key}"]`), channel[key], channel.maximum);
      if (key === 'original') continue;
      const error = channel[key].reduce((total, v, i) => total + Math.abs(v - channel.original[i]), 0) / count;
      demo.querySelector(`[data-error="${key}"]`).textContent = error.toPrecision(3);
    }
  }

  function run() {
    const mode = correction.checked ? 'corrected' : 'shaded';
    const pixels = new Float64Array(count * 2);
    pixels.set(channels.DNA[mode]);
    pixels.set(channels.Actin[mode], count);
    const output = JSON.parse(wasm.run_cppipe(pipeline, descriptors, pixels));
    const overlays = [
      ['Nuclei', 'DNA', output.primary.Nuclei.labels],
      ['Cells', 'Actin', output.secondary.Cells.labels],
      ['Cytoplasm', 'Actin', output.tertiary.Cytoplasm.labels],
    ];
    for (const [scope, name, labels] of overlays) {
      draw(demo.querySelector(`[data-overlay="${scope}"]`), channels[name][mode], channels[name].maximum, labels);
    }

    const ids = output.measurements.object_ids.Cytoplasm;
    const means = output.measurements.features.Cytoplasm.Intensity_MeanIntensity_Stain;
    const rows = ids.map((id, i) => {
      const row = document.createElement('tr');
      row.insertCell().textContent = id;
      row.insertCell().textContent = Number.isFinite(means[i]) ? means[i].toFixed(4) : means[i];
      return row;
    });
    demo.querySelector('tbody').replaceChildren(...rows);
    const csv = ['object_id,mean_actin_intensity', ...ids.map((id, i) => `${id},${means[i]}`)].join('\n') + '\n';
    if (csvUrl) URL.revokeObjectURL(csvUrl);
    csvUrl = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const link = demo.querySelector('[data-csv]');
    link.href = csvUrl;
    link.download = `cytoplasm-intensity-${mode}.csv`;
    results.hidden = false;
    status.textContent = `WASM run complete: ${ids.length} cytoplasm objects on ${mode} images.`;
  }

  compare();
  run();
  channelSelect.addEventListener('change', compare);
  correction.addEventListener('change', () => {
    try { run(); } catch (error) { fail(error); }
  });
}

main().catch(fail);

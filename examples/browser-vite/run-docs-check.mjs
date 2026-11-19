// Check the actual Sphinx build under a GitHub Pages-style project prefix.
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';
import { chromium } from 'playwright';

const [site, expectedPath] = process.argv.slice(2);
assert(site, 'Pass the built Sphinx HTML directory');
const root = resolve(site);
const prefix = '/cp-anywhere/';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm' };
const server = createServer(async (request, response) => {
  const path = new URL(request.url, 'http://localhost').pathname;
  const file = resolve(root, path.slice(prefix.length));
  try {
    assert(path.startsWith(prefix) && file.startsWith(root + sep));
    const body = await readFile(file);
    response.writeHead(200, { 'Content-Type': types[extname(file)] ?? 'application/octet-stream' }).end(body);
  } catch {
    response.writeHead(404).end();
  }
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
try {
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    page.setDefaultTimeout(15000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.route('**/*', (route) => {
      assert(route.request().url().startsWith(origin + prefix), 'Demo must not request external assets');
      return route.continue();
    });
    let loadedWasm = false;
    page.on('response', (response) => {
      if (response.url().endsWith('/cp_anywhere_bg.wasm') && response.ok()) loadedWasm = true;
    });
    assert((await page.goto(`${origin}${prefix}browser-demo.html`))?.ok(), 'Missing live demo docs page');
    const demo = page.locator('#wasm-demo');
    const status = demo.getByRole('status');
    await page.waitForFunction(() => document.querySelector('#wasm-demo [role=status]')?.textContent.includes('4 cytoplasm objects'));
    assert(loadedWasm, 'The page must load the compiled WASM binary');
    const tableRows = () => demo.locator('tbody tr').allTextContents();
    const corrected = await tableRows();
    assert.equal(corrected.length, 4);
    const meanError = (name) => demo.locator(`[data-error="${name}"]`).textContent().then(Number);
    assert(await meanError('shaded') > 0.005);
    assert(await meanError('corrected') < 1e-6);
    for (const canvas of await demo.locator('[data-overlay]').all()) {
      assert(await canvas.evaluate((c) => {
        const data = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i] === 0 && data[i + 1] === 255 && data[i + 2] === 128) return true;
        }
        return false;
      }), 'Rendered overlay must contain green boundaries');
    }
    assert(await demo.evaluate((node) => node.scrollWidth <= node.clientWidth + 1), 'Demo overflows mobile width');
    const [download] = await Promise.all([
      page.waitForEvent('download'), demo.getByRole('link', { name: 'Download measurements (CSV)' }).click(),
    ]);
    const csv = await readFile(await download.path(), 'utf8');
    assert.equal(download.suggestedFilename(), 'cytoplasm-intensity-corrected.csv');
    assert.match(csv, /^object_id,mean_actin_intensity\r?\n1,/);
    assert.equal(csv.trim().split(/\r?\n/).length, 5);
    if (expectedPath) {
      const expected = JSON.parse(await readFile(expectedPath, 'utf8'));
      csv.trim().split(/\r?\n/).slice(1).forEach((line, index) => {
        const [id, mean] = line.split(',').map(Number);
        assert.equal(id, expected[index][0]);
        assert(Math.abs(mean - expected[index][1]) < 1e-10, `Cytoplasm ${id} differs from Python`);
      });
    }
    await demo.getByLabel('Channel for comparison').selectOption('Actin');
    assert.equal(await demo.locator('[data-channel]').textContent(), 'Actin');
    await demo.getByLabel('Use illumination correction').uncheck();
    await page.waitForFunction(() => document.querySelector('#wasm-demo [role=status]')?.textContent.includes('shaded images'));
    assert.notDeepEqual(await tableRows(), corrected);
    await demo.getByLabel('Use illumination correction').check();
    await page.waitForFunction(() => document.querySelector('#wasm-demo [role=status]')?.textContent.includes('corrected images'));
    assert.deepEqual(await tableRows(), corrected);
    assert.deepEqual(errors, []);
    console.log(await status.textContent());

    // Missing assets must surface an error, not leave the page loading forever.
    await page.route('**/dna.u8', (route) => route.abort());
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#wasm-demo [role=status]')?.textContent.startsWith('Demo unavailable:'));
    console.log('Chromium docs check passed: real WASM, overlays, controls, CSV, and load failure');
  } finally {
    await browser.close();
  }
} finally {
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

// Run the built app and real WASM workers in Chromium without shared memory.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { preview } from "vite";

const [fixturePath, resultPath] = process.argv.slice(2);
assert(fixturePath && resultPath, "Pass the fixture directory and result file");
const directory = resolve(fixturePath);
const root = fileURLToPath(new URL(".", import.meta.url));
const manifest = await readFile(join(directory, "manifest.json"));
const files = new Map([["manifest.json", manifest]]);
for (const { pixels } of JSON.parse(manifest).fields) {
  assert.match(pixels, /^[\w-]+\.f64$/);
  files.set(pixels, await readFile(join(directory, pixels)));
}
const server = await preview({ root, preview: { host: "127.0.0.1", port: 0 } });
try {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const failures = [];
    page.on("pageerror", (error) => failures.push(String(error)));
    await page.route("**/batch-fixtures/*", (route) => {
      const name = new URL(route.request().url()).pathname.split("/").at(-1);
      const body = files.get(name);
      return body ? route.fulfill({ body, contentType: name.endsWith(".json") ? "application/json" : "application/octet-stream" })
        : route.fulfill({ status: 404 });
    });
    const address = server.httpServer.address();
    await page.goto(`http://127.0.0.1:${address.port}/?batch-check`);
    await page.waitForFunction(() => Boolean(window.__batchCheck), undefined, { timeout: 15000 });
    assert.equal(await page.evaluate(() => crossOriginIsolated), false);
    const results = await page.evaluate(() => window.__batchCheck);
    assert.deepEqual(failures, []);
    await writeFile(resultPath, JSON.stringify(results));
  } finally {
    await browser.close();
  }
} finally {
  await new Promise((resolve, reject) => server.httpServer.close((error) => error ? reject(error) : resolve()));
}

// Exercise the generated wasm-pack web package in real Playwright Chromium.
// Run: node run-cppipe-check.mjs CASES.json RESULTS.json
import assert from "node:assert/strict";
import { once } from "node:events";
import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { chromium } from "playwright";

const [input, output, ...extra] = process.argv.slice(2);
assert(input && output && !extra.length, "Pass CASES.json and RESULTS.json");
const body = await readFile(input);
const cases = JSON.parse(body);
assert(Array.isArray(cases) && cases.length > 0, "No cases to run");

// Serve the checkout's package directly, not a stale npm copy or a JS fallback.
// This test does not depend on the demo page or change its Vite build.
const pkg = new URL("../../crates/cp_anywhere_wasm/pkg/", import.meta.url);
const files = new Map([
  ["/", ["text/html", "<!doctype html><title>cppipe WASM parity</title>"]],
  ["/cases.json", ["application/json", body]],
  ["/cp_anywhere.js", ["text/javascript", await readFile(new URL("cp_anywhere.js", pkg))]],
  ["/cp_anywhere_bg.wasm", ["application/wasm", await readFile(new URL("cp_anywhere_bg.wasm", pkg))]],
]);
const server = createServer((request, response) => {
  const file = files.get(request.url);
  if (!file) {
    response.writeHead(404).end();
    return;
  }
  const [contentType, body] = file;
  response.writeHead(200, { "Content-Type": contentType }).end(body);
});
server.listen(0, "127.0.0.1");
await once(server, "listening");
try {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(String(error)));
    const response = await page.goto(`http://127.0.0.1:${server.address().port}/`);
    assert(response?.ok(), "Could not load the browser check page");
    const results = await page.evaluate(async () => {
      const wasm = await import("/cp_anywhere.js");
      if (typeof wasm.compile_cppipe !== "function" || typeof wasm.run_cppipe !== "function") {
        throw new Error("Rebuild the web package with compile_cppipe and run_cppipe");
      }
      await wasm.default();
      const response = await fetch("/cases.json");
      if (!response.ok) throw new Error(`Fixture fetch failed: HTTP ${response.status}`);
      const cases = await response.json();

      function invoke(callback) {
        let json;
        try {
          json = callback();
        } catch (error) {
          // An ordinary Rust rejection is a string or Error. Do not confuse
          // a WebAssembly trap, JS argument error, or missing API with one.
          if (typeof error === "string") return { error };
          if (error?.constructor === Error) return { error: error.message };
          throw error;
        }
        if (typeof json !== "string") throw new Error("Expected a JSON string from WASM");
        // Keep JSON text intact: round-tripping via JS would discard duplicate
        // keys and blur integer JSON tokens with floating-point ones.
        return { json };
      }

      return cases.map(({ text, images, pixels }) => {
        const imagesJson = JSON.stringify(images);
        const plane = Float64Array.from(pixels);
        return {
          compile: invoke(() => wasm.compile_cppipe(text)),
          run: invoke(() => wasm.run_cppipe(text, imagesJson, plane)),
        };
      });
    });
    assert.deepEqual(pageErrors, []);
    await writeFile(output, JSON.stringify(results));
  } finally {
    await browser.close();
  }
} finally {
  await new Promise((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve());
  });
}

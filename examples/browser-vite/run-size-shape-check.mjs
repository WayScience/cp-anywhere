// Execute both WASM measurement entry points in real Chromium.
// Read valid JSON inputs from stdin; write only actual browser outputs to stdout.
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
let text = "";
for await (const chunk of process.stdin) text += chunk;
const inputs = JSON.parse(text);
const server = createServer(async (request, response) => {
  if (request.url === "/") {
    response.setHeader("Content-Type", "text/html");
    response.end("<!doctype html><title>WASM size/shape verification</title>");
    return;
  }
  const path = resolve(root, `.${new URL(request.url, "http://localhost").pathname}`);
  if (!path.startsWith(root + sep)) {
    response.writeHead(403).end();
    return;
  }
  try {
    const content = await readFile(path);
    response.setHeader("Content-Type", extname(path) === ".wasm" ? "application/wasm" : "text/javascript");
    response.end(content);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const outputs = await page.evaluate(async (cases) => {
    const wasm = await import("/crates/cp_anywhere_wasm/pkg/cp_anywhere.js");
    await wasm.default();
    return cases.map((input) => {
      const [scope] = Object.keys(input.objects);
      const json = JSON.parse(wasm.measureJson(JSON.stringify(input)));
      const typed = JSON.parse(wasm.measureWithOptions(
        new Uint32Array(input.shape), ["DNA"], new Float64Array(input.images.DNA),
        [scope], new Uint32Array(input.objects[scope]), undefined,
        input.image_id, input.families, [], [], [], new Uint32Array(),
        input.size_shape_calculate_zernikes, input.size_shape_calculate_advanced,
      ));
      return { json, typed };
    });
  }, inputs);
  process.stdout.write(JSON.stringify(outputs));
} finally {
  if (browser) await browser.close();
  await new Promise((done) => server.close(done));
}

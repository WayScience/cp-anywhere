// Reproducible headless-browser acceptance check for the cp-anywhere WASM
// demo: serves the real Vite production build, runs the real worker-driven
// measurement + benchmark pass in a real Chromium instance (Playwright,
// not a mock DOM), and writes the captured results to disk so
// scripts/parity/compare_wasm.py can verify them against the golden
// CellProfiler fixtures using the same comparator used for the native
// build.
//
// Usage: node run-browser-check.mjs [output.json]
// Requires: npm run build (dist/ must exist) and `npx playwright install
// chromium` once.

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { writeFile } from "node:fs/promises";

const here = fileURLToPath(new URL(".", import.meta.url));
const distDir = join(here, "dist");
const outputPath = process.argv[2] ?? join(here, "browser-results.json");

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".f64": "application/octet-stream",
  ".u32": "application/octet-stream",
  ".u8": "application/octet-stream",
};

function serve(port) {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      const urlPath = (req.url ?? "/").split("?")[0];
      const filePath = join(distDir, urlPath === "/" ? "index.html" : urlPath);
      try {
        const body = await readFile(filePath);
        res.setHeader("Content-Type", MIME[extname(filePath)] ?? "application/octet-stream");
        res.end(body);
      } catch {
        res.statusCode = 404;
        res.end("not found");
      }
    });
    server.listen(port, "127.0.0.1", () => resolve(server));
  });
}

async function main() {
  const server = await serve(4174);
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    await page.goto("http://127.0.0.1:4174/");
    await page.waitForFunction("window.__CP_ANYWHERE_DONE__ === true", { timeout: 30000 });
    const results = await page.evaluate(() => window.__CP_ANYWHERE_RESULTS__);
    const status = await page.textContent("#status");
    const environment = await page.evaluate(() => ({
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
    }));
    await writeFile(outputPath, JSON.stringify({ status, errors, results, environment }, null, 2));
    console.log(JSON.stringify({ status, errors, output: outputPath }));
    if (status !== "done" || errors.length > 0) {
      process.exitCode = 1;
    }
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

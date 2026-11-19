// Reproducible headless-browser acceptance check for the Zarr/ONNX/DuckDB
// integration demo: serves the real Vite production build, runs the real
// zarrita.js -> onnxruntime-web -> cp-anywhere -> DuckDB-WASM pipeline
// in a real Chromium instance (Playwright, not a mock DOM), and writes the
// captured results (including the real Parquet bytes) to disk so
// scripts/zarr/validate_export.py can verify them with native DuckDB.
//
// Usage: node run-browser-check.mjs [output.json]
// Requires: npm run build (dist/ must exist) and `npx playwright install
// chromium` once. Requires network access at run time: DuckDB-WASM fetches
// its `parquet` extension from extensions.duckdb.org on first `LOAD parquet`
// (see docs/browser-integration.md "Known limitations" for the offline
// vendoring alternative).

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const distDir = join(here, "dist");
const outputPath = process.argv[2] ?? join(here, "browser-results.json");
const parquetPath = join(here, "cp_anywhere_features.parquet");

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".zarray": "application/json",
  ".zattrs": "application/json",
  ".onnx": "application/octet-stream",
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
  const server = await serve(4175);
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(String(error)));
    await page.goto("http://127.0.0.1:4175/");
    await page.waitForFunction("window.__CP_ANYWHERE_ZOD_DONE__ === true", { timeout: 60000 });
    const result = await page.evaluate(() => window.__CP_ANYWHERE_ZOD_RESULT__);
    const error = await page.evaluate(() => window.__CP_ANYWHERE_ZOD_ERROR__ ?? null);
    const status = await page.textContent("#status");
    if (result?.parquetBase64) {
      await writeFile(parquetPath, Buffer.from(result.parquetBase64, "base64"));
    }
    const { parquetBase64, ...resultWithoutBlob } = result ?? {};
    await writeFile(
      outputPath,
      JSON.stringify({ status, error, pageErrors, result: resultWithoutBlob, parquetPath }, null, 2),
    );
    console.log(JSON.stringify({ status, error, pageErrors, output: outputPath, parquetPath }));
    if (error || pageErrors.length > 0) {
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

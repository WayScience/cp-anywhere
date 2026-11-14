import { defineConfig } from "vite";

// Real local packages (cp-anywhere-wasm's pkg/ output, plus this example's
// own public/ demo assets built by scripts/zarr/export_demo_fixture.py).
// optimizeDeps.exclude keeps Vite from pre-bundling onnxruntime-web, whose
// WASM asset resolution needs to see its own package layout untouched.
export default defineConfig({
  server: { fs: { allow: [".."] } },
  preview: { fs: { allow: [".."] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
});

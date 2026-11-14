import { defineConfig } from "vite";

// The demo loads a real WASM binary and runs it inside a dedicated web
// worker; nothing here mocks either. `fs.allow` lets Vite's dev/preview
// server serve the local `cp-anywhere-wasm` package from outside this
// example's own directory.
export default defineConfig({
  server: { fs: { allow: [".."] } },
  preview: { fs: { allow: [".."] } },
});

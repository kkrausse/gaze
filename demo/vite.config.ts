// Static build: relative asset paths (works under any sub-path, e.g. GitHub Pages), no server routes.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// onnxruntime-web references its wasm next to its module (new URL(..., import.meta.url)); the page loads it from
// ./ort/ instead (ort.env.wasm.wasmPaths, copied by scripts/assets.ts), so the bundled copy is dropped.
const dropOrtWasm = {
  name: "drop-ort-wasm",
  generateBundle(_: unknown, bundle: Record<string, unknown>) {
    for (const k of Object.keys(bundle)) if (/ort-wasm-simd-threaded.*\.wasm$/.test(k)) delete bundle[k];
  },
};

export default defineConfig({
  base: "./",
  resolve: { tsconfigPaths: true },
  server: { port: Number(process.env.PORT ?? 4391), host: "127.0.0.1" },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 2000 },
  plugins: [react(), tailwindcss(), dropOrtWasm],
});

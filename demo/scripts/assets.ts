// Copies the runtime files the page loads at run time into public/ (served from the build itself):
// onnxruntime-web's wasm + its JS glue modules and MediaPipe's vision wasm. The face landmarker model is checked in.
import { cpSync, mkdirSync, readdirSync } from "fs";
import { join } from "path";

const root = join(import.meta.dir, "..");
const ortDist = join(root, "node_modules/onnxruntime-web/dist");
mkdirSync(join(root, "public/ort"), { recursive: true });
for (const f of readdirSync(ortDist)) {
  // the WebGPU build (asyncify) and the plain wasm build; not the JSEP / JSPI variants
  if (/^ort-wasm-simd-threaded(\.asyncify)?\.(wasm|mjs)$/.test(f)) cpSync(join(ortDist, f), join(root, "public/ort", f));
}
cpSync(join(root, "node_modules/@mediapipe/tasks-vision/wasm"), join(root, "public/mediapipe/wasm"), { recursive: true });
console.log("assets copied to public/ort and public/mediapipe/wasm");

// The gaze model in onnxruntime-web (WebGPU when the browser has it, wasm otherwise). Two graphs:
// - xattn.onnx (per frame): eye window + aux + the calibration context's keys / values -> gaze mm, the bin
//   probabilities, and `cal` (this frame's tokens, kept when the frame is a calibration dot)
// - kv.onnx (once per calibration): the K calibration frames' `cal` + their dots -> keys / values + the box
// Files come from the model's Hugging Face repo, are kept in Cache Storage after the first download, and the
// weights (.onnx.data) are handed to ORT as external data.
import * as ort from "onnxruntime-web/webgpu";
import { fileUrl, type ModelEntry } from "./models";
import { CORNERS } from "./calib";
import type { WindowGeom } from "./window";

ort.env.wasm.wasmPaths = new URL("./ort/", document.baseURI).href;

export interface ModelFile { file: string; bytes: number; data?: { file: string; bytes: number } }

/** model.json in the model repo. */
export interface ModelInfo {
  name: string;
  exported: string;
  size: [number, number]; // window the graph takes (H, W)
  window: WindowGeom;
  n_aux: number;
  K: number;
  feat_dim: number;
  cal_tokens: number;
  label_frame: "screen-centre";
  bin_centres: { x: number[]; y: number[] };
  cam_delay_ms: number; // the calibration frame = the one showing the eye this long after the Space press
  files: Partial<Record<Dtype, ModelFile>> & Record<"fp16" | "fp32", ModelFile>;
  kv_files: Partial<Record<Dtype, ModelFile>> & Record<"fp16" | "fp32", ModelFile>;
  /** The log crop-scale input (aux[29]) is relative to the calibration frames: minus their mean (setScaleRef). */
  aux_scale_rel?: boolean;
}

export type Backend = "webgpu" | "wasm";
// q8 / q8.fp32: 8-bit weights (MatMulNBits), fp16 / fp32 compute; about half / a quarter of fp16 / fp32's download
export type Dtype = "fp16" | "fp32" | "q8" | "q8.fp32";

export interface Plan { backend: Backend; dtype: Dtype; why: string; adapter: string; kvWasm?: boolean }

export interface Dist { px: Float32Array; py: Float32Array; xMm: Float32Array; yMm: Float32Array }
export interface GazeOut { mm: [number, number]; cal?: Float32Array; dist: Dist; ms: number }
export interface Context { cal: Float32Array[]; mm: [number, number][] }

export interface GazeModel {
  info: ModelInfo;
  plan: Plan;
  /** One frame with the current context (none until setContext); wantCal: also return the frame's `cal`. */
  run(img: Float32Array, aux: Float32Array, wantCal?: boolean): Promise<GazeOut>;
  setContext(ctx: Context | null): Promise<void>;
  /** aux_scale_rel models: the calibration frames' mean aux[29] (scaleRef), subtracted from every frame's; null = none yet. */
  setScaleRef(ref: number | null): void;
  release(): Promise<void>;
}

/** The reference for aux_scale_rel, as in training: the mean log crop scale (aux[29]) of the corner dots' frames
 * (at least 3 of them), else of all calibration frames. */
export function scaleRef(dots: { grid: number; aux: Float32Array }[]): number {
  const corners = dots.filter((d) => CORNERS.includes(d.grid));
  const use = corners.length >= 3 ? corners : dots;
  return use.reduce((s, d) => s + d.aux[29], 0) / Math.max(1, use.length);
}

export async function fetchInfo(m: ModelEntry): Promise<ModelInfo> {
  const r = await fetch(fileUrl(m, "model.json"), { cache: "no-cache" });
  if (!r.ok) throw new Error(`model.json: HTTP ${r.status}`);
  return r.json();
}

interface Adapter { features: Set<string>; info?: { vendor?: string; architecture?: string; description?: string } }

export async function planModel(info: ModelInfo, want?: { backend?: Backend; dtype?: Dtype }): Promise<Plan> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<Adapter | null> } }).gpu;
  const fallback = new URLSearchParams(location.search).get("gpu") === "fallback";
  const adapter = want?.backend === "wasm" || !gpu ? null : await gpu.requestAdapter(fallback ? { forceFallbackAdapter: true } : { powerPreference: "high-performance" });
  if (adapter) {
    ort.env.webgpu.adapter = adapter as never;
    const f16 = adapter.features.has("shader-f16");
    const dtype = pickDtype(info, f16, want?.dtype);
    const name = [adapter.info?.vendor, adapter.info?.architecture || adapter.info?.description].filter(Boolean).join(" ") || "GPU";
    // the fp32 kv graphs round the calibration features to fp16 inside (a Cast with no shader without shader-f16): on such
    // devices the kv graph (once per calibration) runs on wasm
    return { backend: "webgpu", dtype, why: f16 ? "WebGPU with shader-f16" : "WebGPU without shader-f16", adapter: name, kvWasm: !f16 };
  }
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  const why = want?.backend === "wasm" ? "wasm asked for" : !gpu ? (self.isSecureContext ? "no WebGPU in this browser" : "not a secure context (open it over https)") : "no WebGPU adapter";
  return { backend: "wasm", dtype: pickDtype(info, false, want?.dtype), why, adapter: `CPU, ${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads === 1 ? "" : "s"}` };
}

/** ?dtype= wins if the model has it; else 8-bit weights when published (fp16 compute with shader-f16, else fp32). */
function pickDtype(info: ModelInfo, f16: boolean, want?: Dtype): Dtype {
  const has = (d: Dtype) => !!info.files[d] && !!info.kv_files[d] && (f16 || !d.startsWith("fp16") && d !== "q8");
  if (want && has(want)) return want;
  for (const d of (f16 ? ["q8", "fp16"] : ["q8.fp32", "fp32"]) as Dtype[]) if (has(d)) return d;
  return "fp32";
}

export type Progress = (loaded: number, total: number, what: string) => void;

const CACHE = "gaze-demo-models";

/** One file, from Cache Storage or the network (redirects followed; HF sends CORS headers on the CDN too). */
async function fetchCached(url: string, key: string, bytes: number, onBytes: (n: number) => void): Promise<Uint8Array> {
  const cache = "caches" in self ? await caches.open(CACHE).catch(() => null) : null;
  const hit = await cache?.match(key);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    if (buf.byteLength === bytes) { onBytes(buf.byteLength); return buf; }
    await cache?.delete(key);
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = new Uint8Array(bytes);
  let at = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (at + value.byteLength > bytes) throw new Error(`${url}: more than the expected ${bytes} bytes`);
    buf.set(value, at);
    at += value.byteLength;
    onBytes(value.byteLength);
  }
  if (at !== bytes) throw new Error(`${url}: ${at} of ${bytes} bytes`);
  try { await cache?.put(key, new Response(buf)); } catch { /* quota: downloaded again next time */ }
  return buf;
}

/** Drop cached files of other models / versions. */
async function pruneCache(keep: string[]) {
  if (!("caches" in self)) return;
  const c = await caches.open(CACHE).catch(() => null);
  if (!c) return;
  for (const r of await c.keys()) if (!keep.includes(r.url)) await c.delete(r);
}

export async function loadModel(entry: ModelEntry, info: ModelInfo, plan: Plan, onProgress?: Progress): Promise<GazeModel> {
  const main = info.files[plan.dtype]!, kvf = info.kv_files[plan.dtype]!;
  const parts: { file: string; bytes: number }[] = [main, ...(main.data ? [main.data] : []), kvf];
  const total = parts.reduce((n, p) => n + p.bytes, 0);
  let loaded = 0;
  const key = (f: string) => new URL(`${fileUrl(entry, f)}?v=${encodeURIComponent(info.exported)}`, location.href).href;
  await pruneCache(parts.map((p) => key(p.file)));
  const got: Record<string, Uint8Array> = {};
  for (const p of parts) {
    onProgress?.(loaded, total, p.file);
    got[p.file] = await fetchCached(fileUrl(entry, p.file), key(p.file), p.bytes, (n) => { loaded += n; onProgress?.(loaded, total, p.file); });
  }
  onProgress?.(total, total, "starting the model");
  const gpuOut = plan.backend === "webgpu";
  const opts = {
    executionProviders: [plan.backend],
    graphOptimizationLevel: "all" as const,
    ...(gpuOut ? { preferredOutputLocation: "gpu-buffer" as const } : {}),
  };
  const session = await ort.InferenceSession.create(got[main.file], {
    ...opts, ...(main.data ? { externalData: [{ path: main.data.file, data: got[main.data.file] }] } : {}),
  });
  delete got[main.file];
  if (main.data) delete got[main.data.file];
  const kv = await ort.InferenceSession.create(got[kvf.file], plan.kvWasm ? { executionProviders: ["wasm"], graphOptimizationLevel: "all" as const } : opts);
  return xattnModel(info, plan, session, kv);
}

function xattnModel(info: ModelInfo, plan: Plan, session: ort.InferenceSession, kv: ort.InferenceSession): GazeModel {
  const [H, W] = info.size;
  const K = info.K, D = info.feat_dim, R = info.cal_tokens, nAux = info.n_aux;
  type Packed = { k: ort.Tensor; v: ort.Tensor; keep: ort.Tensor; box: Float32Array; boxT: ort.Tensor };
  const read = async (t: ort.Tensor | undefined, want: boolean): Promise<Float32Array | undefined> => {
    if (!t) return undefined;
    if (t.location !== "gpu-buffer") return want ? Float32Array.from(t.data as Float32Array) : undefined;
    try { return want ? Float32Array.from((await t.getData()) as Float32Array) : undefined; } finally { t.dispose(); }
  };
  const pack = async (ctx: Context | null): Promise<Packed> => {
    const n = Math.min(K, ctx?.cal.length ?? 0);
    const cal = new Float32Array(K * R * D), mm = new Float32Array(K * 2), keep = new Float32Array(K);
    for (let i = 0; i < n; i++) {
      if (ctx!.cal[i].length !== R * D) throw new Error(`calibration dot ${i}: ${ctx!.cal[i].length} numbers, the model wants ${R} x ${D}`);
      cal.set(ctx!.cal[i], i * R * D); mm[2 * i] = ctx!.mm[i][0]; mm[2 * i + 1] = ctx!.mm[i][1]; keep[i] = 1;
    }
    const out = await kv.run({
      ctx_cal: new ort.Tensor("float32", cal, [K, R, D]),
      ctx_mm: new ort.Tensor("float32", mm, [K, 2]),
      ctx_keep: new ort.Tensor("float32", keep, [K]),
    });
    const box = (await read(out.box, true))!; // [cx, cy, hx, hy], network units (100 mm)
    return { k: out.ctx_k, v: out.ctx_v, keep: new ort.Tensor("float32", keep, [1, K]), box, boxT: new ort.Tensor("float32", box, [1, 2, 2]) };
  };
  const bc = info.bin_centres;
  const dist = (px: Float32Array, py: Float32Array, b: Float32Array): Dist => {
    const at = (c: number[], o: number, h: number) => Float32Array.from(c, (u) => (o + h * u) * 100);
    return { px, py, xMm: at(bc.x, b[0], b[2]), yMm: at(bc.y, b[1], b[3]) };
  };
  let none: Packed | null = null, packed: Packed | null = null, sref: number | null = null;
  const free = (p: Packed | null) => { if (p && p !== none) { p.k.dispose?.(); p.v.dispose?.(); } };
  // passes and context swaps on one chain: a context's GPU tensors are released only when no pass can still read them
  let chain: Promise<unknown> = Promise.resolve();
  const enqueue = <T>(f: () => Promise<T>): Promise<T> => { const job = chain.then(f); chain = job.catch(() => {}); return job; };
  return {
    info, plan,
    run: (img, aux, wantCal = false) => enqueue(async () => {
      none ??= await pack(null);
      const p = packed ?? none;
      const t0 = performance.now();
      if (info.aux_scale_rel && sref !== null) { aux = Float32Array.from(aux); aux[29] -= sref; }
      const out = await session.run({
        img: new ort.Tensor("float32", img, [1, H, W, 3]),
        aux: new ort.Tensor("float32", aux.subarray(0, nAux), [1, nAux]),
        ctx_k: p.k, ctx_v: p.v, ctx_keep: p.keep, ctx_box: p.boxT,
      });
      const [m, px, py, cal] = await Promise.all([read(out.mm, true), read(out.px, true), read(out.py, true), read(out.cal, wantCal)]);
      return { mm: [m![0], m![1]], cal, dist: dist(px!, py!, p.box), ms: performance.now() - t0 };
    }),
    setContext: (ctx) => enqueue(async () => {
      const old = packed;
      packed = ctx?.cal.length ? await pack(ctx) : null;
      free(old);
    }),
    setScaleRef: (r) => { sref = r; },
    release: async () => { await chain; free(packed); free(none); none = null; await session.release(); await kv.release(); },
  };
}

// Test mode without a webcam: ?clip=<url of clip.json>. Plays a list of still frames through the same pipeline
// as the camera (landmarks in VIDEO mode over each frame's lead-in frames, window, model, calibration context)
// and shows the predictions. clip.json:
//   { page: { w, h, widthCm },
//     frames: [{ src, lead?: [src...], dot?: { grid, page: [x, y] }, lm?: [[x, y] x 6], ref?: url, aux?: number[] }] }
// Frames with `dot` are calibration dots (grid = row-major 3x3 index, page = the dot in page px); the others are
// predicted after the calibration. ?lm=given uses the clip's landmarks instead of the landmarker's. `ref` (raw RGB
// bytes of an expected window) and `aux` (expected aux values), when given, are compared with what the page computes; ?lmonly=1 runs the landmarker alone (no model).
// The result is also left in window.__clipResult.
import { useEffect, useState } from "react";
import type { ModelEntry } from "./gaze/models";
import { fetchInfo, loadModel, planModel, scaleRef, type Backend, type Dtype } from "./gaze/model";
import { createLandmarker } from "./gaze/landmarks";
import { cutWindow, grabRegion, type Pt } from "./gaze/window";
import { contextOrder } from "./gaze/calib";
import { pageToMm, type Page } from "./gaze/screen";

interface ClipFrame { src: string; lead?: string[]; dot?: { grid: number; page: [number, number] }; lm?: Pt[]; ref?: string; aux?: number[] }
interface Clip { page: Page; frames: ClipFrame[] }

export interface ClipFrameResult {
  src: string; dot: boolean; lm: Pt[] | null; lmGiven: boolean; lmDiffPx?: number; aux?: number[]; auxDiff?: number;
  pixMax?: number; pixMean?: number; mm?: [number, number]; ms?: number;
  /** aux_scale_rel models: aux[29] as the model gets it (minus the calibration reference) and its difference from the clip's */
  aux29?: number; aux29Diff?: number;
}

const AUX_USED = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30];

export function ClipRun({ clipUrl, entry, want }: { clipUrl: string; entry: ModelEntry; want: { backend?: Backend; dtype?: Dtype } }) {
  const [log, setLog] = useState<string[]>([]);
  const [rows, setRows] = useState<ClipFrameResult[]>([]);
  useEffect(() => {
    let alive = true;
    const say = (t: string) => alive && setLog((l) => [...l, t]);
    (async () => {
      try {
        const base = new URL(clipUrl, location.href);
        const clip: Clip = await (await fetch(base)).json();
        const lmOnly = new URLSearchParams(location.search).get("lmonly") === "1";
        const info = await fetchInfo(entry);
        if (lmOnly) {
          const detect = await createLandmarker();
          const load = async (src: string) => { const im = new Image(); im.src = new URL(src, base).href; await im.decode(); return im; };
          let t = 0;
          const frames = [];
          for (const f of clip.frames) {
            for (const l of f.lead ?? []) { const im = await load(l); detect(im, (t += 33), im.naturalWidth, im.naturalHeight); }
            const im = await load(f.src);
            const lm = detect(im, (t += 33), im.naturalWidth, im.naturalHeight);
            frames.push({ src: f.src, dot: !!f.dot, lm, lmGiven: false, lmDiffPx: lm && f.lm ? lm.reduce((s, p, k) => s + Math.hypot(p[0] - f.lm![k][0], p[1] - f.lm![k][1]), 0) / 6 : undefined });
          }
          (window as unknown as { __clipResult: unknown }).__clipResult = { model: info.name, backend: "-", dtype: "-", order: [], frames };
          say("done (landmarks only)");
          return;
        }
        const plan = await planModel(info, want);
        say(`${info.name}: ${plan.backend} ${plan.dtype} (${plan.why}; ${plan.adapter})`);
        const t0 = performance.now();
        const model = await loadModel(entry, info, plan, () => {});
        say(`model loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
        const detect = await createLandmarker();
        const given = new URLSearchParams(location.search).get("lm") === "given";
        const canvas = new OffscreenCanvas(1, 1);
        const load = async (src: string) => { const im = new Image(); im.src = new URL(src, base).href; await im.decode(); return im; };
        let t = 0;
        const out: ClipFrameResult[] = [];
        const cals: { grid: number; mm: [number, number]; cal: Float32Array }[] = [];
        const later: { i: number; img: Float32Array; aux: Float32Array }[] = [];
        const dots: { i: number; grid: number; mm: [number, number]; img: Float32Array; aux: Float32Array }[] = [];
        for (const [i, f] of clip.frames.entries()) {
          for (const l of f.lead ?? []) { const im = await load(l); detect(im, (t += 33), im.naturalWidth, im.naturalHeight); }
          const im = await load(f.src);
          const W = im.naturalWidth, H = im.naturalHeight;
          const found = detect(im, (t += 33), W, H);
          const lm = given && f.lm ? f.lm : found;
          const r: ClipFrameResult = { src: f.src, dot: !!f.dot, lm, lmGiven: given && !!f.lm };
          if (found && f.lm) r.lmDiffPx = found.reduce((s, p, k) => s + Math.hypot(p[0] - f.lm![k][0], p[1] - f.lm![k][1]), 0) / 6;
          out.push(r);
          if (!lm) { say(`${f.src}: no face`); continue; }
          const region = grabRegion(im, W, H, lm, info.window, canvas);
          if (!region) { say(`${f.src}: window outside the frame`); continue; }
          const win = cutWindow(region, lm, info.window, W, H);
          r.aux = Array.from(win.aux);
          // aux_scale_rel: the clip's aux[29] is relative to the calibration; compared once the reference is known
          if (f.aux) r.auxDiff = Math.max(...AUX_USED.filter((k) => !info.aux_scale_rel || k !== 29).map((k) => Math.abs(win.aux[k] - f.aux![k])));
          if (f.ref) {
            const ref = new Uint8Array(await (await fetch(new URL(f.ref, base))).arrayBuffer());
            let mx = 0, s = 0;
            for (let k = 0; k < ref.length; k++) { const d = Math.abs(ref[k] - win.img[k]); s += d; if (d > mx) mx = d; }
            r.pixMax = mx; r.pixMean = s / ref.length;
          }
          if (f.dot) dots.push({ i, grid: f.dot.grid, mm: pageToMm(clip.page, f.dot.page[0], f.dot.page[1]), img: win.img, aux: win.aux });
          else later.push({ i, img: win.img, aux: win.aux });
          if (alive) setRows([...out]);
        }
        // as the live page: the calibration passes after the last dot, with the crop-scale reference from all of them
        const sref = info.aux_scale_rel ? scaleRef(dots) : null;
        model.setScaleRef(sref);
        if (sref !== null) for (const [i, f] of clip.frames.entries()) {
          const r = out[i];
          if (!r.aux) continue;
          r.aux29 = r.aux[29] - sref;
          if (f.aux) { r.aux29Diff = Math.abs(r.aux29 - f.aux[29]); r.auxDiff = Math.max(r.auxDiff ?? 0, r.aux29Diff); }
        }
        for (const d of dots) {
          const o = await model.run(d.img, d.aux, true);
          cals.push({ grid: d.grid, mm: d.mm, cal: o.cal! });
          out[d.i].ms = o.ms;
          say(`dot ${cals.length} (grid ${d.grid}): ${o.ms.toFixed(0)} ms`);
        }
        const order = contextOrder(cals);
        say(`context order (grid): ${order.map((k) => cals[k].grid).join(",")}`);
        await model.setContext({ cal: order.map((k) => cals[k].cal), mm: order.map((k) => cals[k].mm) });
        for (const l of later) {
          const o = await model.run(l.img, l.aux);
          out[l.i].mm = o.mm; out[l.i].ms = o.ms;
          if (alive) setRows([...out]);
        }
        const res = { model: info.name, backend: plan.backend, dtype: plan.dtype, K: info.K, scaleRel: !!info.aux_scale_rel, scaleRef: sref, order: order.map((k) => cals[k].grid), frames: out };
        (window as unknown as { __clipResult: unknown }).__clipResult = res;
        say("done");
        await model.release();
      } catch (e) {
        say(`error: ${e instanceof Error ? e.message : String(e)}`);
        (window as unknown as { __clipResult: unknown }).__clipResult = { error: String(e) };
      }
    })();
    return () => { alive = false; };
  }, [clipUrl, entry, want]);
  return (
    <div className="p-4 font-mono text-xs space-y-2">
      <div className="font-sans text-base font-semibold">Clip test</div>
      {log.map((l, i) => <div key={i}>{l}</div>)}
      <table className="tabular-nums">
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="pr-3">{r.src}</td><td className="pr-3">{r.dot ? "dot" : ""}</td>
              <td className="pr-3">{r.lmDiffPx != null ? `lm ${r.lmDiffPx.toFixed(2)} px` : ""}</td>
              <td className="pr-3">{r.auxDiff != null ? `aux ${r.auxDiff.toExponential(1)}` : ""}</td>
              <td className="pr-3">{r.pixMax != null ? `pix max ${r.pixMax} mean ${r.pixMean!.toFixed(4)}` : ""}</td>
              <td className="pr-3">{r.mm ? `${r.mm[0].toFixed(1)}, ${r.mm[1].toFixed(1)} mm` : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

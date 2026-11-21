// The gaze heat map: the model's two per-axis bin distributions (x and y, each over its calibration box) taken as
// independent, p(x, y) = px(x) py(y), drawn as a coloured overlay. Smoothed over time so it does not flicker.
import type { Dist } from "./model";
import { mmToPage, type Page } from "./screen";

export function createHeat(ema = 0.5) {
  let px: Float32Array | null = null, py: Float32Array | null = null, xMm: Float32Array | null = null, yMm: Float32Array | null = null;
  let img: ImageData | null = null;
  const off = document.createElement("canvas");
  return {
    push(d: Dist) {
      if (!px || px.length !== d.px.length || py!.length !== d.py.length) { px = Float32Array.from(d.px); py = Float32Array.from(d.py); }
      else for (let i = 0; i < px.length; i++) px[i] = ema * px[i] + (1 - ema) * d.px[i];
      if (py && py.length === d.py.length) for (let j = 0; j < py.length; j++) py[j] = ema * py[j] + (1 - ema) * d.py[j];
      xMm = d.xMm; yMm = d.yMm;
    },
    reset() { px = py = xMm = yMm = null; },
    draw(g: CanvasRenderingContext2D, page: Page) {
      if (!px || !py || !xMm || !yMm) return;
      const nx = px.length, ny = py.length;
      if (!img || img.width !== nx || img.height !== ny) { img = new ImageData(nx, ny); off.width = nx; off.height = ny; }
      let mx = 0, my = 0;
      for (const v of px) mx = Math.max(mx, v);
      for (const v of py) my = Math.max(my, v);
      const top = mx * my || 1, a = img.data;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const t = Math.sqrt((px[i] * py[j]) / top); // sqrt: the tails stay visible
        const o = 4 * (j * nx + i);
        // transparent -> yellow -> red
        a[o] = 255; a[o + 1] = Math.round(230 * (1 - t) + 40 * t); a[o + 2] = Math.round(60 * (1 - t));
        a[o + 3] = t < 0.04 ? 0 : Math.round(200 * Math.min(1, t * 1.2));
      }
      off.getContext("2d")!.putImageData(img, 0, 0);
      // bin centres are evenly spaced in mm, so the grid maps to an axis-aligned page rectangle (x is mirrored)
      const [x0, y0] = mmToPage(page, xMm[0], yMm[0]), [x1, y1] = mmToPage(page, xMm[nx - 1], yMm[ny - 1]);
      const sx = (x1 - x0) / (nx - 1), sy = (y1 - y0) / (ny - 1);
      g.save();
      g.imageSmoothingEnabled = true;
      g.translate(x0 - sx / 2, y0 - sy / 2); g.scale(sx, sy);
      g.drawImage(off, 0, 0);
      g.restore();
    },
  };
}

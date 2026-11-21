// The gaze heat map: the model's two per-axis bin distributions (x and y, each over its calibration box) taken as
// independent within one prediction, p(x, y) = px(x) py(y), drawn as a coloured overlay. The 2D map itself is smoothed
// over predictions (smoothing x and y apart would cross one prediction's x with another's y: ghost blobs).
import type { Dist } from "./model";
import { mmToPage, type Page } from "./screen";

export function createHeat(ema = 0.5, floor = 0.15) {
  let map: Float32Array | null = null, nx = 0, ny = 0, xMm: Float32Array | null = null, yMm: Float32Array | null = null;
  let img: ImageData | null = null;
  const off = document.createElement("canvas");
  return {
    push(d: Dist) {
      const fresh = !map || nx !== d.px.length || ny !== d.py.length;
      if (fresh) { nx = d.px.length; ny = d.py.length; map = new Float32Array(nx * ny); }
      const m = map!, w = fresh ? 1 : 1 - ema;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const k = j * nx + i; m[k] = (1 - w) * m[k] + w * d.px[i] * d.py[j]; }
      xMm = d.xMm; yMm = d.yMm;
    },
    reset() { map = xMm = yMm = null; },
    draw(g: CanvasRenderingContext2D, page: Page) {
      if (!map || !xMm || !yMm) return;
      if (!img || img.width !== nx || img.height !== ny) { img = new ImageData(nx, ny); off.width = nx; off.height = ny; }
      let top = 0;
      for (const v of map) top = Math.max(top, v);
      const a = img.data;
      for (let k = 0; k < map.length; k++) {
        const t = map[k] / (top || 1), o = 4 * k;
        // below `floor` of the peak: nothing drawn; above it transparent yellow -> opaque red
        const u = t <= floor ? 0 : (t - floor) / (1 - floor);
        a[o] = 255; a[o + 1] = Math.round(220 * (1 - u) + 30 * u); a[o + 2] = Math.round(50 * (1 - u));
        a[o + 3] = Math.round(210 * u);
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

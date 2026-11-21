// The model's input window, cut level from the camera frame in one resample.
//
// Geometry (all in camera px): the outer eye corners (MediaPipe 33 and 263) are SPAN px apart in a layout of
// stripW x stripH, centred on the mean of the four eye corners and turned so those corners are level.
// The model takes the centred WIN_W x WIN_H window of that layout (moved by dy rows). The camera frame goes to
// the window in ONE warpAffine: bicubic when enlarging; when shrinking, bilinear onto a 2x finer grid and then
// 2x2 means (one anti-aliased interpolation); a region more than 2x too big is first halved (2x2 means).
//
// aux (31 numbers): 12:24 the six landmarks in centred window coordinates, 27:30 the window centre over the frame
// size minus 0.5 (x, y) and the log of the mean scale, 30 the roll taken out. Slots 0-11 and 24-26 are
// unused (0).
import { halveRgba, warpAffine, type Affine, type Rgba } from "./warp";

export type Pt = [number, number];

export const SPAN = 640;
export const LM6 = [33, 133, 362, 263, 61, 291];

export interface WindowGeom {
  stripW: number; stripH: number; // the layout the window sits in (1056 x 368)
  winW: number; winH: number; // the window the model takes (768 x 252)
  dy: number; // window moved down (> 0) / up in layout rows
}

export interface EyeWindow {
  img: Float32Array; // winH x winW x 3, RGB 0-255
  aux: Float32Array; // 31
  roll: number;
  scale: [number, number]; // window px per camera px (x, y)
  centre: Pt; // eye-corner centre, camera px
  lm: Pt[]; // the six landmarks in window px
  M: Affine; // camera px -> window px
}

const f = Math.fround;

/** Python's round(): half to even. */
export function pyRound(v: number): number {
  const fl = Math.floor(v), d = v - fl;
  if (d > 0.5) return fl + 1;
  if (d < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

/** Window placement for these landmarks (no pixels). */
export function windowPlacement(lm6: Pt[], g: WindowGeom) {
  let vx = lm6[3][0] - lm6[0][0], vy = lm6[3][1] - lm6[0][1];
  if (vx < 0) { vx = -vx; vy = -vy; } // left to right in the image, whatever the landmark order
  const roll = f(Math.atan2(vy, vx)); // float32
  const ex = lm6[3][0] - lm6[0][0], ey = lm6[3][1] - lm6[0][1];
  const span = Math.sqrt(ex * ex + ey * ey);
  const k = (SPAN * (g.stripW - 32)) / 1024 / span;
  let cx = 0, cy = 0;
  for (let i = 0; i < 4; i++) { cx += lm6[i][0]; cy += lm6[i][1]; }
  cx /= 4; cy /= 4;
  const sw = Math.max(1, pyRound(g.stripW / k)), sh = Math.max(1, pyRound(g.stripH / k));
  const co = Math.cos(roll), si = Math.sin(roll);
  const kx = g.stripW / sw, ky = g.stripH / sh;
  const ox = Math.floor((g.stripW - g.winW) / 2);
  const oy = Math.min(Math.max(Math.floor((g.stripH - g.winH) / 2) + g.dy, 0), g.stripH - g.winH);
  // camera px -> level box px: R(-roll) (p - c) + (sw, sh) / 2; -> layout px (pixel centres kept); -> window px
  const tx = sw / 2 - (co * cx + si * cy), ty = sh / 2 - (-si * cx + co * cy);
  const M: Affine = [
    kx * co, kx * si, kx * tx + 0.5 * kx - 0.5 - ox,
    -ky * si, ky * co, ky * ty + 0.5 * ky - 0.5 - oy,
  ];
  return { roll, k, cx, cy, sw, sh, co, si, kx, ky, ox, oy, M };
}

/** Camera px rectangle (integer) the warp reads, taps included. */
export function sourceRect(lm6: Pt[], g: WindowGeom) {
  const { M } = windowPlacement(lm6, g);
  // window corners -> camera px (inverse of M)
  const D = 1 / (M[0] * M[4] - M[1] * M[3]);
  const inv = (u: number, v: number): Pt => {
    const a = u - M[2], b = v - M[5];
    return [(M[4] * a - M[1] * b) * D, (-M[3] * a + M[0] * b) * D];
  };
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [u, v] of [[-0.5, -0.5], [g.winW - 0.5, -0.5], [-0.5, g.winH - 0.5], [g.winW - 0.5, g.winH - 0.5]]) {
    const [x, y] = inv(u, v);
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  // even origin (a halved region then keeps whole camera pixels in each 2x2 block)
  const ex = (Math.floor(x0) - 4) & ~1, ey = (Math.floor(y0) - 4) & ~1;
  return { x0: ex, y0: ey, x1: Math.ceil(x1) + 5, y1: Math.ceil(y1) + 5 };
}

/** The window from a frame region (src.ox / oy = its place in the frame), with the aux vector.
 * frameW / frameH: the whole camera frame (the window centre is given as a fraction of it). */
export function cutWindow(src: Rgba, lm6: Pt[], g: WindowGeom, frameW: number, frameH: number): EyeWindow {
  const p = windowPlacement(lm6, g);
  let M = p.M;
  const sMax = Math.max(Math.abs(M[0]) + Math.abs(M[1]), Math.abs(M[3]) + Math.abs(M[4]));
  let s = src;
  let red = sMax <= 0.25 ? 4 : sMax <= 0.5 ? 2 : 1;
  if (red > 1) {
    // halve the region (2x2 means; twice for 4x), then the warp reads half-size px: full = red * half + (red - 1) / 2
    const ox = s.ox ?? 0, oy = s.oy ?? 0;
    let h = { data: s.data, width: s.width, height: s.height } as Rgba;
    for (let r = red; r > 1; r >>= 1) h = halveRgba(h);
    s = { ...h, ox: 0, oy: 0 };
    const a = red, b = 0.5 * red - 0.5;
    // M' = M . [[a, 0, b + ox], [0, a, b + oy]]
    M = [M[0] * a, M[1] * a, M[0] * (b + ox) + M[1] * (b + oy) + M[2], M[3] * a, M[4] * a, M[3] * (b + ox) + M[4] * (b + oy) + M[5]];
  }
  const sMin = Math.min(Math.abs(M[0]) + Math.abs(M[1]), Math.abs(M[3]) + Math.abs(M[4]));
  let rgb: Uint8Array;
  if (sMin >= 1) rgb = warpAffine(s, M, g.winW, g.winH, "cubic");
  else {
    const M2: Affine = [2 * M[0], 2 * M[1], 2 * M[2] + 0.5, 2 * M[3], 2 * M[4], 2 * M[5] + 0.5];
    const big = warpAffine(s, M2, 2 * g.winW, 2 * g.winH, "linear");
    rgb = halveRgb(big, 2 * g.winW, 2 * g.winH);
  }
  const img = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) img[i] = rgb[i];
  // aux (float32 where the model expects float32)
  const { cx, cy, sw, sh, co, si, kx, ky, ox, oy, roll } = p;
  const lmS: Pt[] = lm6.map(([x, y]) => {
    const dx = x - cx, dy = y - cy;
    return [f((co * dx + si * dy + sw / 2) * kx), f((-si * dx + co * dy + sh / 2) * ky)];
  });
  const cxy: Pt = [f(cx - sw / 2), f(cy - sh / 2)], ks: Pt = [f(kx), f(ky)];
  const aux = new Float32Array(31);
  lmS.forEach(([x, y], i) => {
    aux[12 + 2 * i] = (x - ox) / g.winW - 0.5;
    aux[13 + 2 * i] = (y - oy) / g.winH - 0.5;
  });
  const winX = cxy[0] + (ox + g.winW / 2) / ks[0], winY = cxy[1] + (oy + g.winH / 2) / ks[1];
  aux[27] = winX / frameW - 0.5;
  aux[28] = winY / frameH - 0.5;
  aux[29] = Math.log(f(f(ks[0] + ks[1]) / 2));
  aux[30] = roll;
  return { img, aux, roll, scale: [kx, ky], centre: [cx, cy], lm: lmS.map(([x, y]) => [x - ox, y - oy] as Pt), M: p.M };
}

function halveRgb(img: Uint8Array, w: number, h: number): Uint8Array {
  const ow = w >> 1, oh = h >> 1, out = new Uint8Array(ow * oh * 3);
  for (let y = 0; y < oh; y++) {
    const r0 = 2 * y * w * 3, r1 = (2 * y + 1) * w * 3;
    for (let x = 0; x < ow; x++) {
      const o = (y * ow + x) * 3, a = 2 * x * 3;
      for (let c = 0; c < 3; c++) out[o + c] = (img[r0 + a + c] + img[r0 + a + 3 + c] + img[r1 + a + c] + img[r1 + a + 3 + c] + 2) >> 2;
    }
  }
  return out;
}

/** The frame region the window needs, read from a video / image at 1:1 through a canvas. */
export function grabRegion(src: CanvasImageSource, frameW: number, frameH: number, lm6: Pt[], g: WindowGeom, canvas: OffscreenCanvas): Rgba | null {
  const b = sourceRect(lm6, g);
  const sx0 = Math.max(0, b.x0), sy0 = Math.max(0, b.y0);
  const sx1 = Math.min(frameW, b.x1), sy1 = Math.min(frameH, b.y1);
  if (sx1 <= sx0 || sy1 <= sy0) return null;
  const w = sx1 - sx0, h = sy1 - sy0;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(src, sx0, sy0, w, h, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h);
  return { data: d.data, width: w, height: h, ox: sx0, oy: sy0 };
}

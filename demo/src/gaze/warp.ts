// OpenCV's cv2.warpAffine (8-bit, BORDER_CONSTANT 0) and the 2x2 INTER_AREA shrink, ported so the browser's eye
// window matches OpenCV's output pixel for pixel (OpenCV 5's warp kernels: float32 source coordinates,
// float32 interpolation, one rounding at the end). Checked against cv2 5.0: INTER_LINEAR and INTER_CUBIC within one
// level on a handful of pixels in 10^5 (float summation order), INTER_AREA 2x exact.

export interface Rgba {
  data: Uint8ClampedArray | Uint8Array; // RGBA, row-major
  width: number;
  height: number;
  ox?: number; oy?: number; // frame px of data's top-left, when it is only part of the frame
}

/** 2x3 affine, source px -> destination px, row-major [a, b, c, d, e, f]. */
export type Affine = [number, number, number, number, number, number];

/** lrint: round half to even. */
export const cvRound = (v: number) => { const r = Math.round(v); return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r; };
const f = Math.fround;
const A = -0.75;

/** OpenCV's cubic coefficients (a = -0.75) for a fractional offset x, in float32. */
function cubic(x: number, out: Float64Array) {
  const c0 = f(f(f(f(f(f(A * f(x + 1)) - f(5 * A)) * f(x + 1)) + f(8 * A)) * f(x + 1)) - f(4 * A));
  const c1 = f(f(f(f(f(f(A + 2) * x) - f(A + 3)) * x) * x) + 1);
  const c2 = f(f(f(f(f(f(A + 2) * f(1 - x)) - f(A + 3)) * f(1 - x)) * f(1 - x)) + 1);
  out[0] = c0; out[1] = c1; out[2] = c2; out[3] = f(f(f(1 - c0) - c1) - c2);
}

/** OpenCV's inversion of the forward map (destination px -> source px). */
function invert(m: Affine): Affine {
  let D = m[0] * m[4] - m[1] * m[3];
  D = D !== 0 ? 1 / D : 0;
  const A11 = m[4] * D, A22 = m[0] * D, A12 = -m[1] * D, A21 = -m[3] * D;
  const b1 = -A11 * m[2] - A12 * m[5], b2 = -A21 * m[2] - A22 * m[5];
  return [A11, A12, b1, A21, A22, b2];
}

/** cv2.warpAffine(src, M, (outW, outH), INTER_CUBIC | INTER_LINEAR, BORDER_CONSTANT 0) -> RGB uint8 (outH x outW x 3).
 * M: source (full-frame) px -> output px; src may be a part of the frame (ox, oy). */
export function warpAffine(src: Rgba, M: Affine, outW: number, outH: number, mode: "cubic" | "linear"): Uint8Array {
  const [M0, M1, M2, M3, M4, M5] = invert(M);
  const { data, width: fw, height: fh } = src;
  const sox = src.ox ?? 0, soy = src.oy ?? 0;
  const out = new Uint8Array(outW * outH * 3);
  if (mode === "linear") {
    // src = M^-1 dst, bilinear in float, one rounding; taps outside the image are the border value 0
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        const sxf = f(f(f(M0) * x) + f(f(f(M1) * y) + f(M2))) - sox, syf = f(f(f(M3) * x) + f(f(f(M4) * y) + f(M5))) - soy;
        const x0 = Math.floor(sxf), y0 = Math.floor(syf);
        const ax = f(sxf - x0), ay = f(syf - y0);
        const o = (y * outW + x) * 3;
        if (x0 >= fw || x0 + 1 < 0 || y0 >= fh || y0 + 1 < 0) continue;
        for (let c = 0; c < 3; c++) {
          const px = (xx: number, yy: number) => (xx >= 0 && xx < fw && yy >= 0 && yy < fh ? data[(yy * fw + xx) * 4 + c] : 0);
          const p00 = px(x0, y0), p01 = px(x0 + 1, y0), p10 = px(x0, y0 + 1), p11 = px(x0 + 1, y0 + 1);
          const top = f(p00 + f(ax * f(p01 - p00))), bot = f(p10 + f(ax * f(p11 - p10)));
          const v = f(top + f(ay * f(bot - top)));
          out[o + c] = v <= 0 ? 0 : v >= 255 ? 255 : cvRound(v);
        }
      }
    }
    return out;
  }
  const wx = new Float64Array(4), wy = new Float64Array(4);
  for (let y = 0; y < outH; y++) {
    const by = f(f(f(M1) * y) + f(M2)), by2 = f(f(f(M4) * y) + f(M5));
    for (let x = 0; x < outW; x++) {
      const sxf = f(f(f(M0) * x) + by) - sox, syf = f(f(f(M3) * x) + by2) - soy;
      const x0 = Math.floor(sxf), y0 = Math.floor(syf);
      const o = (y * outW + x) * 3;
      if (x0 - 1 >= fw || x0 + 3 <= 0 || y0 - 1 >= fh || y0 + 3 <= 0) continue; // all taps outside: border 0
      cubic(f(sxf - x0), wx); cubic(f(syf - y0), wy);
      const inside = x0 >= 1 && y0 >= 1 && x0 + 3 <= fw && y0 + 3 <= fh;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let i = 0; i < 4; i++) {
          const yy = y0 - 1 + i;
          let row = 0;
          if (inside) {
            const p = (yy * fw + x0 - 1) * 4 + c;
            row = f(f(f(f(wx[0] * data[p]) + f(wx[1] * data[p + 4])) + f(wx[2] * data[p + 8])) + f(wx[3] * data[p + 12]));
          } else if (yy >= 0 && yy < fh) {
            for (let j = 0; j < 4; j++) {
              const xx = x0 - 1 + j;
              row = f(row + f(wx[j] * (xx >= 0 && xx < fw ? data[(yy * fw + xx) * 4 + c] : 0)));
            }
          }
          v = f(v + f(wy[i] * row));
        }
        out[o + c] = v <= 0 ? 0 : v >= 255 ? 255 : cvRound(v);
      }
    }
  }
  return out;
}

/** cv2.resize(img, (w / 2, h / 2), INTER_AREA) on RGB uint8: each output pixel the rounded mean of a 2x2 block. */
export function halve(img: Uint8Array, w: number, h: number): Uint8Array {
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

/** The same 2x2 mean straight from RGBA (a frame region decoded at half size). */
export function halveRgba(src: Rgba): Rgba {
  const { data, width: w, height: h } = src;
  const ow = w >> 1, oh = h >> 1, out = new Uint8ClampedArray(ow * oh * 4);
  for (let y = 0; y < oh; y++) {
    const r0 = 2 * y * w * 4, r1 = (2 * y + 1) * w * 4;
    for (let x = 0; x < ow; x++) {
      const o = (y * ow + x) * 4, a = 2 * x * 4;
      for (let c = 0; c < 3; c++) out[o + c] = (data[r0 + a + c] + data[r0 + a + 4 + c] + data[r1 + a + c] + data[r1 + a + 4 + c] + 2) >> 2;
      out[o + 3] = 255;
    }
  }
  return { data: out, width: ow, height: oh };
}

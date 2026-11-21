// Gaze in mm <-> page CSS px. The model's labels are mm from the centre of the page, in the screen plane:
// x mirrored (the camera faces the user: + x is to the user's left on the screen), y down.

export interface Page { w: number; h: number; widthCm: number }

export function pageToMm(g: Page, x: number, y: number): [number, number] {
  const mmPerPx = (g.widthCm * 10) / g.w;
  return [-(x - g.w / 2) * mmPerPx, (y - g.h / 2) * mmPerPx];
}

export function mmToPage(g: Page, mx: number, my: number): [number, number] {
  const mmPerPx = (g.widthCm * 10) / g.w;
  return [g.w / 2 - mx / mmPerPx, g.h / 2 + my / mmPerPx];
}

// One-euro filter (Casiez et al. 2012) on a 2D point; t in ms.
export function createOneEuro(minCutoff = 0.6, beta = 0.004, dCutoff = 1) {
  let last: { t: number; x: number[]; dx: number[] } | null = null;
  const alpha = (cutoff: number, dt: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
  return {
    filter(t: number, p: [number, number]): [number, number] {
      if (!last || t - last.t > 1000) { last = { t, x: [...p], dx: [0, 0] }; return p; }
      const dt = Math.max(1e-3, (t - last.t) / 1000);
      const ad = alpha(dCutoff, dt);
      const out: number[] = [];
      for (let i = 0; i < 2; i++) {
        const dx = ad * ((p[i] - last.x[i]) / dt) + (1 - ad) * last.dx[i];
        const a = alpha(minCutoff + beta * Math.abs(dx), dt);
        out.push(a * p[i] + (1 - a) * last.x[i]);
        last.dx[i] = dx;
      }
      last = { t, x: out, dx: last.dx };
      return [out[0], out[1]];
    },
    reset() { last = null; },
  };
}

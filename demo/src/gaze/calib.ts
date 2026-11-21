// The calibration: 4 corners (default) or a 9-dot 3x3 grid at the page's edges (the dot's radius + 4 px in from each edge), shown
// clockwise from the top-left corner with the centre last. The model gets the dots as its context in a fixed
// order: the 4 corners first (grid order), then farthest-point from those.

export const EDGE_PX = 14;
export const SHOW_ORDER = [0, 1, 2, 5, 8, 7, 6, 3, 4]; // row-major grid index
/** The 4-dot calibration: the corners only, clockwise from the top-left. */
export const SHOW_ORDER_4 = [0, 2, 8, 6];
const CORNERS = [0, 2, 6, 8];

/** Page px of grid point i (row-major 3x3). */
export function gridPoint(i: number, w: number, h: number): [number, number] {
  const u = [0, 0.5, 1][i % 3], v = [0, 0.5, 1][Math.floor(i / 3)];
  const at = (f: number, s: number) => Math.min(Math.max(f * s, EDGE_PX), s - EDGE_PX);
  return [at(u, w), at(v, h)];
}

/** Context order for dots given by grid index and label (mm): corners first (grid order), then farthest-point from
 * those, ties to the earlier dot; distances in float32. Returns indices into `dots`. */
export function contextOrder(dots: { grid: number; mm: [number, number] }[]): number[] {
  const f = Math.fround;
  const byGrid = dots.map((_, i) => i).sort((a, b) => dots[a].grid - dots[b].grid);
  const lab = dots.map((d) => [f(d.mm[0]), f(d.mm[1])]);
  const pick = byGrid.filter((i) => CORNERS.includes(dots[i].grid));
  if (!pick.length && byGrid.length) pick.push(byGrid[0]);
  const dist = (a: number, b: number) => { const dx = f(lab[a][0] - lab[b][0]), dy = f(lab[a][1] - lab[b][1]); return f(Math.sqrt(f(f(dx * dx) + f(dy * dy)))); };
  const d = byGrid.map((i) => Math.min(...pick.map((p) => dist(i, p))));
  while (pick.length < byGrid.length) {
    let j = 0;
    for (let q = 1; q < d.length; q++) if (d[q] > d[j]) j = q;
    pick.push(byGrid[j]);
    for (let q = 0; q < d.length; q++) d[q] = Math.min(d[q], dist(byGrid[q], byGrid[j]));
  }
  return pick;
}

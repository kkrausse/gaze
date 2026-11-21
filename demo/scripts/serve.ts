// bun scripts/serve.ts <dir> [port]: serve a static directory (e.g. the build) for a local look.
// Extra mounts for local checks: GAZE_MOUNTS="/hf/=<dir>,/clips/=<dir>". Cross-origin isolated (wasm threads).
import { existsSync, statSync } from "fs";
import { join, normalize } from "path";

const root = process.argv[2] ?? "dist";
const port = Number(process.argv[3] ?? process.env.PORT ?? 4393);
const mounts: [string, string][] = (process.env.GAZE_MOUNTS ?? "").split(",").filter(Boolean).map((m) => m.split("=") as [string, string]);
const iso = { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp", "cross-origin-resource-policy": "same-origin" };

Bun.serve({
  port, hostname: "127.0.0.1",
  fetch(req) {
    const p = decodeURIComponent(new URL(req.url).pathname);
    let dir = root, rel = p;
    for (const [pre, d] of mounts) if (p.startsWith(pre)) { dir = d; rel = p.slice(pre.length); break; }
    let f = normalize(join(dir, rel));
    if (!f.startsWith(normalize(dir))) return new Response("no", { status: 403 });
    if (existsSync(f) && statSync(f).isDirectory()) f = join(f, "index.html");
    if (!existsSync(f)) return new Response("not found", { status: 404, headers: iso });
    return new Response(Bun.file(f), { headers: iso });
  },
});
console.log(`serving ${root} on http://127.0.0.1:${port}/`);

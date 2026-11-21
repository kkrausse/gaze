import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { HF, initialModel, MODELS, type ModelEntry } from "./gaze/models";
import { fetchInfo, loadModel, planModel, type Backend, type Dtype, type GazeModel } from "./gaze/model";
import { createLandmarker, type Detect } from "./gaze/landmarks";
import { cutWindow, grabRegion, type Pt } from "./gaze/window";
import type { Rgba } from "./gaze/warp";
import { contextOrder, gridPoint, SHOW_ORDER } from "./gaze/calib";
import { createOneEuro, mmToPage, pageToMm, type Page } from "./gaze/screen";
import { ClipRun } from "./ClipRun";

const params = new URLSearchParams(location.search);
const WANT = { backend: (params.get("backend") as Backend) || undefined, dtype: (params.get("dtype") as Dtype) || undefined };
const BG = "#7a7a7a"; // mid grey behind the dots
// A calibration dot's frame: the one closest to press + camera delay, at most PRESS_MAX_MS off; when the page
// skipped frames around then (busy main thread, slow camera), the nearest one up to PRESS_SLACK_MS off.
const PRESS_MAX_MS = 50, PRESS_SLACK_MS = 100;
const MOVE_MS = 600; // a calibration dot glides this long to its place; Space counts once it has stopped

// The model works in its calibration box's units, so the page's physical size cancels out (calibration dots in and
// predictions out use the same mm-per-px); the nominal CSS pixel (1/96 inch) is used for the mm in between.
const CSS_CM_PER_PX = 2.54 / 96;

async function listCameras() {
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
}

async function openCamera(deviceId?: string) {
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 60 } },
  });
}

export function App() {
  const clip = params.get("clip");
  const [model, setModel] = useState<ModelEntry>(initialModel);
  const [cams, setCams] = useState<MediaDeviceInfo[]>([]);
  const [cam, setCam] = useState(localStorage.getItem("gaze-demo-cam") ?? "");
  const [running, setRunning] = useState(false);
  const runRef = useRef<HTMLDivElement>(null);

  useEffect(() => { listCameras().then(setCams).catch(() => {}); }, []);
  useEffect(() => { localStorage.setItem("gaze-demo-cam", cam); }, [cam]);

  if (clip) return <ClipRun clipUrl={clip} entry={model} want={WANT} />;

  const start = async () => {
    setRunning(true);
    await document.documentElement.requestFullscreen?.().catch(() => {});
  };

  return (
    <>
      <div className="mx-auto max-w-xl p-6 space-y-6">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold">Gaze demo</h1>
          <p className="text-sm text-muted-foreground">
            Webcam gaze tracking in the browser. Everything runs on this device: camera frames never leave the page. The model
            (several hundred MB) is downloaded once and kept in the browser's cache. After a 9-dot calibration a red dot shows
            where you are looking.
          </p>
        </header>
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="model">Model</Label>
            <select id="model" className="h-8 w-full rounded-lg border border-input bg-transparent px-2 text-sm"
              value={model.id} onChange={(e) => setModel(MODELS.find((m) => m.id === e.target.value)!)}>
              {MODELS.map((m) => <option key={m.id} value={m.id} disabled={!m.available}>{m.label}{m.available ? "" : " (not published yet)"}</option>)}
            </select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cam">Camera</Label>
            <select id="cam" className="h-8 w-full rounded-lg border border-input bg-transparent px-2 text-sm" value={cam}
              onChange={(e) => setCam(e.target.value)} onFocus={() => listCameras().then(setCams)}>
              <option value="">Default camera</option>
              {cams.filter((c) => c.deviceId).map((c) => <option key={c.deviceId} value={c.deviceId}>{c.label || c.deviceId.slice(0, 8)}</option>)}
            </select>
          </div>
          <p className="text-xs text-muted-foreground">
            Put the camera at the top of the screen, sit as you normally would, keep the face well lit.
          </p>
          <Button size="lg" onClick={start} disabled={!model.available}>Start</Button>
        </div>
        <footer className="space-y-1 border-t pt-4 text-xs text-muted-foreground">
          <p>
            Model weights: <a className="underline" href={`${HF}${model.repo}`}>{model.repo}</a> on Hugging Face, under the {model.licence}.
            The page's code is MIT licensed.
          </p>
          <p>Face landmarks by MediaPipe; inference by ONNX Runtime Web.</p>
        </footer>
      </div>
      <div ref={runRef} className={running ? "fixed inset-0 z-50" : "hidden"} style={{ background: BG }}>
        {running && <Run entry={model} camId={cam} onExit={() => { setRunning(false); if (document.fullscreenElement) document.exitFullscreen().catch(() => {}); }} />}
      </div>
    </>
  );
}

type Phase = { kind: "loading"; text: string; frac: number } | { kind: "error"; text: string } | { kind: "intro" } | { kind: "calib"; dot: number } | { kind: "live" };

interface FrameGrab { t: number; lm: Pt[]; region: Rgba; fw: number; fh: number }

function Run({ entry, camId, onExit }: { entry: ModelEntry; camId: string; onExit: () => void }) {
  const [phase, setPhase] = useState<Phase>({ kind: "loading", text: "Loading", frac: 0 });
  const [hud, setHud] = useState("");
  const [note, setNote] = useState("");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const st = useRef({
    model: null as GazeModel | null, detect: null as Detect | null,
    face: null as Pt[] | null, faceAt: 0,
    dotFrom: [0, 0] as [number, number], dotTo: [0, 0] as [number, number], dotT0: 0, dotI: 0,
    press: null as null | { tp: number; best: FrameGrab | null; bestD: number },
    cals: [] as { grid: number; mm: [number, number]; img: Float32Array; aux: Float32Array }[],
    busy: false, gaze: null as [number, number] | null, raw: null as [number, number] | null, ms: 0, n: 0, t0: 0,
    phase: "loading" as Phase["kind"], filter: createOneEuro(),
  });
  const page = (): Page => ({ w: innerWidth, h: innerHeight, widthCm: innerWidth * CSS_CM_PER_PX });

  // model, landmarker, camera
  useEffect(() => {
    let alive = true, stream: MediaStream | null = null, vfc = 0;
    const s = st.current;
    (async () => {
      try {
        setPhase({ kind: "loading", text: "Model info", frac: 0 });
        const info = await fetchInfo(entry);
        const plan = await planModel(info, WANT);
        const model = await loadModel(entry, info, plan, (l, t, what) => alive && setPhase({ kind: "loading", text: `Downloading ${what}: ${(l / 1e6).toFixed(0)} of ${(t / 1e6).toFixed(0)} MB`, frac: t ? l / t : 0 }));
        if (!alive) { await model.release(); return; }
        s.model = model;
        setPhase({ kind: "loading", text: "Face landmarks", frac: 1 });
        s.detect = await createLandmarker();
        setPhase({ kind: "loading", text: "Camera", frac: 1 });
        stream = await openCamera(camId || undefined);
        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play();
        setHud(`${plan.backend} ${plan.dtype} (${plan.why}; ${plan.adapter}) · camera ${v.videoWidth}x${v.videoHeight}`);
        const tick = (now: number, md: VideoFrameCallbackMetadata) => {
          if (!alive) return;
          onFrame(md.captureTime ?? now, v, md.mediaTime);
          vfc = v.requestVideoFrameCallback(tick);
        };
        vfc = v.requestVideoFrameCallback(tick);
        if (alive) setPhase({ kind: "intro" });
      } catch (e) {
        if (alive) setPhase({ kind: "error", text: String(e instanceof Error ? e.message : e) });
      }
    })();
    return () => {
      alive = false;
      videoRef.current?.cancelVideoFrameCallback(vfc);
      stream?.getTracks().forEach((t) => t.stop());
      s.model?.release();
      s.model = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    st.current.phase = phase.kind;
    // state for automated checks
    (window as unknown as { __gazeDemo: unknown }).__gazeDemo = { phase: phase.kind, dot: phase.kind === "calib" ? phase.dot : null };
  }, [phase]);

  const canvas = useMemo(() => new OffscreenCanvas(1, 1), []);

  function onFrame(t: number, v: HTMLVideoElement, mediaTime: number) {
    const s = st.current;
    if (!s.detect || !s.model) return;
    const W = v.videoWidth, H = v.videoHeight;
    const lm = s.detect(v, t, W, H);
    s.face = lm; if (lm) s.faceAt = t;
    const g = s.model.info.window, delay = s.model.info.cam_delay_ms;
    if (s.press && lm) {
      const d = Math.abs(t - (s.press.tp + delay));
      if (d <= PRESS_SLACK_MS && d < s.press.bestD) {
        const region = grabRegion(v, W, H, lm, g, canvas);
        if (region) { s.press.best = { t, lm, region, fw: W, fh: H }; s.press.bestD = d; }
      }
    }
    if (s.press && (t > s.press.tp + delay + PRESS_SLACK_MS || (t > s.press.tp + delay + PRESS_MAX_MS && s.press.bestD <= PRESS_MAX_MS))) finishPress();
    if (s.phase === "live" && lm && !s.busy) {
      const region = grabRegion(v, W, H, lm, g, canvas);
      if (!region) return;
      const win = cutWindow(region, lm, g, W, H);
      const vt = mediaTime;
      s.busy = true;
      s.model.run(win.img, win.aux).then((o) => {
        // the latest prediction, for automated checks (gaze mm, the frame's media time)
        (window as unknown as { __gaze: unknown }).__gaze = { mm: o.mm, vt, n: s.n + 1 };
        s.raw = mmToPage(page(), o.mm[0], o.mm[1]);
        s.gaze = s.filter.filter(performance.now(), s.raw);
        s.ms = o.ms; s.n++;
      }).catch((e) => console.warn(e)).finally(() => { s.busy = false; });
    }
  }

  // calibration
  const showDot = (i: number, now: number) => {
    const s = st.current, p = page();
    s.dotFrom = i === 0 ? [p.w / 2, p.h / 2] : s.dotTo;
    s.dotTo = gridPoint(SHOW_ORDER[i], p.w, p.h);
    s.dotT0 = now; s.dotI = i;
    setPhase({ kind: "calib", dot: i });
  };
  const startCalib = () => {
    const s = st.current;
    s.cals = []; s.press = null; s.gaze = s.raw = null; s.filter.reset();
    s.model?.setContext(null);
    showDot(0, performance.now());
  };
  function finishPress() {
    const s = st.current, pr = s.press!;
    s.press = null;
    if (!pr.best) {
      setNote("No face in the frame at the press: press Space again");
      (window as unknown as { __gazeDemo: { retry?: number } }).__gazeDemo.retry = performance.now();
      return;
    }
    setNote("");
    const p = page(), grid = SHOW_ORDER[s.dotI];
    const mm = pageToMm(p, s.dotTo[0], s.dotTo[1]);
    const b = pr.best, win = cutWindow(b.region, b.lm, s.model!.info.window, b.fw, b.fh);
    // the model runs on the dots' frames after the last dot: on a CPU backend a pass blocks the page for seconds
    s.cals.push({ grid, mm, img: win.img, aux: win.aux });
    if (s.dotI + 1 < SHOW_ORDER.length) showDot(s.dotI + 1, performance.now());
    else finishCalib();
  }
  async function finishCalib() {
    const s = st.current;
    setPhase({ kind: "loading", text: "Calibrating", frac: 1 });
    const cal: (Float32Array | null)[] = [];
    for (const [i, c] of s.cals.entries()) {
      setPhase({ kind: "loading", text: `Calibrating: dot ${i + 1} of ${s.cals.length}`, frac: i / s.cals.length });
      cal.push(await s.model!.run(c.img, c.aux, true).then((o) => o.cal ?? null).catch(() => null));
    }
    const ok = s.cals.map((c, i) => ({ grid: c.grid, mm: c.mm, cal: cal[i] })).filter((c) => c.cal);
    const order = contextOrder(ok);
    await s.model!.setContext({ cal: order.map((i) => ok[i].cal!), mm: order.map((i) => ok[i].mm) });
    s.t0 = performance.now(); s.n = 0;
    setPhase({ kind: "live" });
  }

  // keys
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = st.current;
      if (e.key === "Escape") { onExit(); return; }
      if (e.code === "Space") {
        e.preventDefault();
        if (s.phase === "intro") startCalib();
        // the key's own time stamp: the moment of the press even if the page handles it late
        else if (s.phase === "calib" && !s.press && e.timeStamp - s.dotT0 > MOVE_MS) s.press = { tp: e.timeStamp, best: null, bestD: Infinity };
      }
      if (e.key === "r" || e.key === "R") if (s.phase === "live") startCalib();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // drawing
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const c = canvasRef.current, s = st.current;
      if (c) {
        const dpr = devicePixelRatio || 1;
        if (c.width !== innerWidth * dpr) { c.width = innerWidth * dpr; c.height = innerHeight * dpr; }
        const g = c.getContext("2d")!;
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, innerWidth, innerHeight);
        const now = performance.now();
        if (s.phase === "calib") {
          const a = Math.min(1, (now - s.dotT0) / MOVE_MS), e = a * a * (3 - 2 * a);
          const x = s.dotFrom[0] + (s.dotTo[0] - s.dotFrom[0]) * e, y = s.dotFrom[1] + (s.dotTo[1] - s.dotFrom[1]) * e;
          g.fillStyle = "#fff"; g.beginPath(); g.arc(x, y, a < 1 ? 10 : 7, 0, Math.PI * 2); g.fill();
          if (a >= 1) { g.fillStyle = "#000"; g.beginPath(); g.arc(x, y, 2, 0, Math.PI * 2); g.fill(); }
        }
        if (s.phase === "live") {
          if (s.raw) { g.strokeStyle = "rgba(255,255,255,0.5)"; g.lineWidth = 1.5; g.beginPath(); g.arc(s.raw[0], s.raw[1], 6, 0, Math.PI * 2); g.stroke(); }
          if (s.gaze) { g.fillStyle = "#e11d48"; g.beginPath(); g.arc(s.gaze[0], s.gaze[1], 12, 0, Math.PI * 2); g.fill(); }
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  const [, force] = useState(0);
  useEffect(() => { const id = setInterval(() => force((n) => n + 1), 500); return () => clearInterval(id); }, []);
  const s = st.current;
  const faceOk = s.face && performance.now() - s.faceAt < 500;
  const rate = s.n && s.t0 ? (s.n / ((performance.now() - s.t0) / 1000)).toFixed(1) : "-";

  return (
    <div className="absolute inset-0 select-none text-white">
      <video ref={videoRef} className="hidden" muted playsInline />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      {phase.kind === "loading" && (
        <div className="absolute inset-0 grid place-items-center">
          <div className="w-96 space-y-2 text-center text-sm">
            <div>{phase.text}</div>
            <div className="h-2 w-full overflow-hidden rounded bg-white/20"><div className="h-full bg-white" style={{ width: `${(phase.frac * 100).toFixed(1)}%` }} /></div>
          </div>
        </div>
      )}
      {phase.kind === "error" && (
        <div className="absolute inset-0 grid place-items-center p-6 text-center">
          <div className="max-w-lg space-y-3"><div className="font-medium">Could not start</div><div className="text-sm">{phase.text}</div><Button variant="secondary" onClick={onExit}>Back</Button></div>
        </div>
      )}
      {phase.kind === "intro" && (
        <div className="absolute inset-0 grid place-items-center text-center">
          <div className="max-w-md space-y-2">
            <div className="text-lg">Calibration: 9 dots</div>
            <div className="text-sm">Look at each dot. When it has stopped, press Space while looking at its centre.</div>
            <div className="text-sm">{faceOk ? "Face found." : "No face found yet: check the camera and the light."}</div>
            <div className="text-sm font-medium">Press Space to begin · Esc to leave</div>
          </div>
        </div>
      )}
      {phase.kind === "calib" && (note || (phase.dot === 0 && performance.now() - s.dotT0 < 2500)) && (
        <div className="absolute bottom-8 w-full text-center text-sm">{note || "Look at the dot, press Space"}</div>
      )}
      {(phase.kind === "live" || phase.kind === "calib") && (
        <div className="absolute bottom-2 left-3 text-xs text-white/80 tabular-nums">
          {hud}{phase.kind === "live" ? ` · ${s.ms.toFixed(0)} ms per prediction · ${rate}/s` : ""}{faceOk ? "" : " · no face"}
        </div>
      )}
      {phase.kind === "live" && (
        <div className="absolute bottom-2 right-3 flex gap-2">
          <Button size="sm" variant="secondary" onClick={startCalib}>Recalibrate (R)</Button>
          <Button size="sm" variant="secondary" onClick={onExit}>Exit (Esc)</Button>
        </div>
      )}
    </div>
  );
}

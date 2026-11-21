// MediaPipe Face Landmarker (VIDEO mode: tracking across frames, so it runs on every camera frame), served from
// the build: ./mediapipe/wasm and ./mediapipe/face_landmarker.task. CPU delegate by default: the model expects
// MediaPipe's CPU-path landmarks; the GPU delegate's land ~2-3 px up and to the left of them (a constant offset).
// ?delegate=GPU switches.
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { LM6, type Pt } from "./window";

export type Detect = (src: HTMLVideoElement | HTMLImageElement | ImageBitmap | HTMLCanvasElement, t: number, w: number, h: number) => Pt[] | null;

export async function createLandmarker(delegate: "GPU" | "CPU" = (new URLSearchParams(location.search).get("delegate") === "GPU" ? "GPU" : "CPU")): Promise<Detect> {
  const fileset = await FilesetResolver.forVisionTasks(new URL("./mediapipe/wasm", document.baseURI).href);
  const lm = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: new URL("./mediapipe/face_landmarker.task", document.baseURI).href, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  let last = -1;
  return (src, t, w, h) => {
    if (t <= last) t = last + 0.001;
    last = t;
    const f = lm.detectForVideo(src as HTMLVideoElement, t).faceLandmarks?.[0];
    if (!f) return null;
    return LM6.map((k) => [f[k].x * w, f[k].y * h] as Pt);
  };
}

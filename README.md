# gaze

Webcam gaze estimation in the browser. Calibrate with 9 dots, then a dot follows where you look on the screen.
Everything runs on your machine: the camera never leaves the page, and the only download is the model.

**Demo:** https://kkrausse.github.io/gaze/ (Chrome or Edge with WebGPU; a desktop GPU is recommended)

## Models

| model | weights | licence |
|---|---|---|
| `gaze` (m69) | [kkrausse/gaze](https://huggingface.co/kkrausse/gaze) | Sapiens2 License |
| `gaze-nc` (m68) | [kkrausse/gaze-nc](https://huggingface.co/kkrausse/gaze-nc) | Sapiens2 License, non-commercial use only |

Both use Meta's [Sapiens2](https://github.com/facebookresearch/sapiens2) 0.4B as the image backbone, followed by
transformer blocks that cross-attend to the 9 calibration frames and their dot positions. `gaze-nc` was
also trained on MPIIFaceGaze, UT Multiview and EVE, which are non-commercial datasets; `gaze` was not.

Pick a model in the demo, or with `?model=m69` / `?model=m68`.

## Accuracy

Mean error of the gaze point after the 9-dot calibration, in mm on the screen and in degrees. The same frames,
calibration and protocol for both models.

| held-out set | n | `gaze` (m69) | `gaze-nc` (m68) |
|---|---|---|---|
| Webcam sessions, locked test | 55 sessions, 1068 dots | 21.1 mm / 2.32° | 18.0 mm / 1.99° |
| Webcam sessions, standard test | 19 sessions, 480 dots | 19.8 mm / 2.20° | 17.3 mm / 1.93° |
| MPIIFaceGaze, 3 held-out people | 761 frames | 29.2 mm / 3.26° | 20.4 mm / 2.31° |
| UT Multiview, 5 held-out people | 612 frames | 41.0 mm / 3.83° | 20.8 mm / 1.95° |
| EVE, held-out person val01 (4 cameras) | 110 frames | 30.1 mm / 2.84° | 19.6 mm / 1.85° |

- The webcam sessions are one person at one laptop-size screen, about 50 cm away, with MacBook, phone and Sony
  cameras. The locked test was never used to pick a model.
- For `gaze`, MPIIFaceGaze, UT Multiview and EVE are unseen datasets (cross-dataset), calibrated with 9 frames of
  the same recording. `gaze-nc` saw other people from them.
- `gaze` is an early version and will be replaced.
- Without calibration both are far off (75–110 mm): calibration is required.

## How it works

1. MediaPipe Face Landmarker (CPU) finds the eye corners in each camera frame.
2. A 768 × 252 window around both eyes is cut from the frame in one resample, levelled so the eye corners are
   horizontal, at a fixed eye-corner distance (`demo/src/gaze/window.ts`). 31 numbers go with it: the six
   landmarks in window coordinates, where the window sits in the frame, its scale, and the roll taken out.
3. Calibration: for each of the 9 dots, the frame ~100 ms after you press Space is run once; its tokens and the
   dot's position become the calibration context (`kv.onnx`, once per calibration).
4. Every frame then runs `xattn.onnx` with that context: per-axis probabilities over positions relative to the
   calibration box, whose mean is the gaze point in mm from the centre of the screen
   (`demo/src/gaze/model.ts`, `screen.ts`).

Graph inputs and outputs:

| graph | inputs | outputs |
|---|---|---|
| `kv.onnx` | `ctx_cal` [K, R, D] calibration tokens, `ctx_mm` [K, 2] dots (mm / 100), `ctx_keep` [K] | `ctx_k`, `ctx_v` (keys / values), `box` [1, 2, 2] (centre, half-size) |
| `xattn.onnx` | `img` [1, 252, 768, 3] RGB 0–255, `aux` [1, 31], `ctx_k`, `ctx_v`, `ctx_keep` [1, K], `ctx_box` | `mm` [1, 2] (mm / 100), `px` / `py` (bin probabilities), `cal` (this frame's tokens) |

K = 9, and R and D are in each model's `model.json`.

## Run it locally

```sh
cd demo
bun install
bun run dev      # http://127.0.0.1:4391
bun run build    # static site in demo/dist
```

## Licence

- Code in this repository: MIT (`LICENSE`).
- Model weights: the Sapiens2 License (Meta), a copy of which ships with each model; `gaze-nc` is also
  non-commercial only.
- `demo/public/mediapipe/face_landmarker.task` and the MediaPipe Tasks wasm: Google MediaPipe, Apache 2.0.
  onnxruntime-web (JS and wasm): MIT.

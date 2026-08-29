# Facial Gesture Keyboard — Step 1: Camera + Landmark Tracking

This is the first working piece of the hackathon project: a browser page
that streams your webcam to a local Python server, which runs face
landmark detection (MediaPipe) and sends the landmarks back to be drawn
live on screen.

**What this does:** proves the camera → tracking pipeline works.
**What this doesn't do yet:** turn tracking into the 6 gestures
(up/down/left/right/confirm/backspace) or drive the on-screen keyboard —
that's the next step once this is confirmed working on your machines and
webcams.

## 1. Setup

Requires Python 3.10–3.12.

```bash
cd facial-gesture-keyboard
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run

```bash
uvicorn main:app --reload
```

Then open **http://localhost:8000** in Chrome or Edge (Safari's WebSocket
+ getUserMedia support is patchier — use Chrome/Edge if you hit issues).

On the very first run, the server downloads a small (~4MB) face-tracking
model file automatically — you'll see a `[setup] Downloading...` line in
the terminal. This needs internet access once; after that it's cached
locally in `face_landmarker.task` and works offline.

Your browser will ask for camera permission — allow it. You should see
your video feed with colored dots tracking your eyes, mouth, eyebrows,
cheeks, and nose, plus a telemetry panel on the right showing FPS and
round-trip latency.

## 3. What to check as a team

- **Detection reliability:** does tracking stay stable as you move,
  turn your head, or change lighting? Note where it breaks — that
  directly informs how forgiving the gesture classifier needs to be.
- **Latency:** the telemetry panel shows round-trip ms per frame. If
  it's consistently above ~150-200ms, that's worth addressing before
  building gesture logic on top (see "If it's too slow" below).
- **Multiple machines/webcams:** worth testing on all 3 laptops now,
  since webcam quality and lighting varies and this is the foundation
  everything else builds on.

## 4. If it's too slow (latency > ~200ms)

The current setup sends JPEG frames over a WebSocket to a Python
backend. If that round trip is too slow for responsive gesture control,
the fallback discussed in the project plan is to run MediaPipe's
JavaScript build directly in the browser instead, avoiding the network
hop entirely. Flag this early rather than late — it's a meaningful
rework, not a tweak.

## 5. Project layout

```
facial-gesture-keyboard/
  main.py              FastAPI server: WebSocket endpoint + MediaPipe inference
  requirements.txt
  static/
    index.html          Page structure (camera preview, telemetry panel)
    style.css            Visual design
    app.js                Camera capture, WebSocket client, landmark drawing
  face_landmarker.task  (auto-downloaded on first run, not checked into git)
```

## 6. Troubleshooting

- **`ModuleNotFoundError: mediapipe`** — make sure the venv is activated
  before running `uvicorn`.
- **Camera permission denied / black video** — check the browser's site
  settings for localhost, and check no other app (Zoom, another browser
  tab) is already holding the camera.
- **Model download fails** — check your network allows
  `storage.googleapis.com`; some corporate/venue wifi blocks it. If
  that's the case on hackathon wifi, download the model beforehand on
  a different network and drop `face_landmarker.task` directly into the
  project folder.
- **"no_face" even when clearly facing the camera** — try better,
  more even lighting first; this is the single biggest factor in
  detection reliability.

## 7. Next step

Once tracking looks solid: build the calibration flow (record a few
reps of each of the 6 gestures from the target user) and the lightweight
classifier that turns landmark movement into discrete gesture events —
covered in the project plan document.

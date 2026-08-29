# Facial Gesture Keyboard — Camera Tracking + Gesture Calibration

This covers the first two working pieces of the hackathon project:

1. **Tracking** — a browser page streams your webcam to a local Python
   server, which runs face landmark detection (MediaPipe) and draws the
   landmarks live on screen.
2. **Calibration + gesture classification** — record a few seconds of
   each of the 6 target gestures (up/down/left/right/confirm/backspace)
   plus a neutral face, train a lightweight classifier on the spot, and
   see discrete gesture events fire live as you repeat them.

**What this doesn't do yet:** drive an actual on-screen keyboard from
those fired events — that's the next step once gesture recognition
feels reliable.

## How the gesture mapping works

There's no hardcoded rule like "eyebrow raise = up". Instead, you pick
*any* facial movement you can reliably repeat for each action — it
doesn't need to resemble the real gesture at all — and the calibration
step records what that looks like in landmark-feature space. A small
k-nearest-neighbors classifier (`gestures.py`) then learns to tell your
7 recorded patterns apart. This matters for the target users: which
facial movements are reliably controllable varies a lot person to
person, so the mapping is learned per-user rather than assumed.

## 1. Setup

Requires Python 3.10 or newer (3.13 confirmed to work).

**macOS / Linux:**
```bash
cd facial-gesture-keyboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```powershell
cd facial-gesture-keyboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
> Windows doesn't have a `python3` command by default — use `python` (or
> `py`). If `python --version` says "not found" and opens the Microsoft
> Store, Python isn't installed yet: get it from
> [python.org/downloads](https://python.org/downloads) and check **"Add
> python.exe to PATH"** during install, then reopen your terminal.
>
> If `.venv\Scripts\activate` gets blocked by PowerShell's execution
> policy, either run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy
> Bypass` first, or use Command Prompt instead, where
> `.venv\Scripts\activate.bat` just works.

## 2. Run

```bash
uvicorn main:app --reload
```

Then open **http://localhost:8000**. On first run the server downloads
a small (~4MB) face-tracking model automatically (needs internet once,
then cached locally). Allow camera access when prompted.

## 3. Calibrate

Scroll down to the **Calibration** panel. For each of the 7 rows
(`neutral`, `up`, `down`, `left`, `right`, `confirm`, `backspace`):

1. Decide what facial movement you'll use for that action.
2. Click **Record**, hold/repeat that movement for a few seconds
   (watch the counter climb — aim for well past the minimum shown).
3. Click **Record** again to stop.

For `neutral`, just hold a relaxed, resting face.

Once every row is past its minimum, click **Train Classifier**. If it
succeeds, stop recording and just make your gestures naturally — fired
events will appear in the **Live Events** panel in the sidebar, and the
**live prediction** telemetry row shows the raw per-frame classification
before debouncing.

If a gesture isn't firing reliably: click **Clear** on that row and
re-record with a more exaggerated or more consistent version of the
movement — the classifier is only as good as how repeatable the
recorded samples are.

## 4. What to check as a team

- **Detection reliability:** does tracking stay stable as you move,
  turn your head, or change lighting?
- **Gesture separability:** are the 6 gestures + neutral easy to tell
  apart in practice, or do some get confused with each other? If two
  gestures keep firing as each other, they're probably too similar in
  landmark-feature space — pick more distinct movements for them.
- **Latency:** the telemetry panel shows round-trip ms per frame. If
  it's consistently above ~150-200ms, that's worth addressing before
  wiring gestures to the keyboard (see "If it's too slow" below).
- **Multiple machines/webcams:** worth testing on all 3 laptops now,
  since webcam quality and lighting varies.

## 5. If it's too slow (latency > ~200ms)

The current setup sends JPEG frames over a WebSocket to a Python
backend. If that round trip is too slow for responsive gesture control,
the fallback discussed in the project plan is to run MediaPipe's
JavaScript build directly in the browser instead, avoiding the network
hop entirely. Flag this early rather than late — it's a meaningful
rework, not a tweak.

## 6. Project layout

```
facial-gesture-keyboard/
  main.py               FastAPI server: WebSocket endpoint, MediaPipe inference,
                         calibration protocol handling
  gestures.py            Feature extraction, calibration storage, classifier
                          training/prediction, and gesture-event debouncing
                          (unit-testable without a camera or the ML model)
  requirements.txt
  static/
    index.html            Page structure (tracking view, telemetry, calibration panel)
    style.css              Visual design
    app.js                  Camera capture, WebSocket client, landmark drawing,
                            calibration UI wiring
  face_landmarker.task   (auto-downloaded on first run, not checked into git)
```

## 7. Troubleshooting

- **`ModuleNotFoundError`** — make sure the venv is activated before
  running `uvicorn`.
- **Camera permission denied / black video** — check the browser's site
  settings for localhost, and check no other app already holds the
  camera.
- **Model download fails** — check your network allows
  `storage.googleapis.com`; some corporate/venue wifi blocks it. If
  that's the case on hackathon wifi, download the model beforehand on a
  different network and drop `face_landmarker.task` directly into the
  project folder.
- **"no_face" even when clearly facing the camera** — try better, more
  even lighting first; this is the single biggest factor in detection
  reliability.
- **"Train Classifier" says you need more samples** — the message
  lists exactly which labels are short; go record more for those.
- **A gesture won't stop firing / fires constantly** — this usually
  means its recorded samples overlap too much with neutral. Re-record
  that label with a more deliberate, exaggerated movement.

## 8. Next step

Wire fired gesture events to an actual on-screen keyboard: a
directional-scanning grid UI where up/down/left/right move a
highlighted cursor, confirm selects a key, and backspace deletes —
covered in the project plan document.

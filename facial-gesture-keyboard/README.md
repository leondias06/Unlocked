# Facial Gesture Keyboard — Camera Tracking + Gesture Calibration

This covers the first two working pieces of the hackathon project:

1. **Tracking** — a browser page streams your webcam to a local Python
   server, which runs face landmark detection (MediaPipe) and draws the
   landmarks live on screen.
2. **Calibration + gesture classification** — record a few seconds of
   each of the 6 target gestures (up/down/left/right/confirm/backspace)
   plus a neutral face, train a lightweight classifier on the spot, and
   see discrete gesture events fire live as you repeat them.
3. **On-screen keyboard** — a compact, translucent directional-scanning
   keyboard (toggle with the button in the corner, or press **K**).
   up/down/left/right move the highlighted cursor, confirm types the
   highlighted key, backspace deletes. It types **real OS-level
   keystrokes** (via `pynput`), so it drives whatever window actually
   has focus — not just the browser tab. See "On-screen keyboard"
   below for platform notes.

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
python run.py
```

Then open **http://localhost:8000**. On first run the server downloads
a small (~4MB) face-tracking model automatically (needs internet once,
then cached locally). Allow camera access when prompted.

Use `run.py` instead of `uvicorn main:app --reload` directly - plain
`--reload` watches every file in the project **including `.venv`**,
which has thousands of files from mediapipe/scipy/etc. Anything that
touches one (pip, antivirus, search indexing) looks like a code change
and restarts the whole server, wiping calibration progress and
sometimes crashing with a `BrokenPipeError`. `run.py` excludes `.venv`
from the watch. (This can't be fixed by adding `--reload-exclude
".venv/*"` to the CLI command on Windows - Click, uvicorn's CLI
framework, auto-expands `*` in command-line arguments against real
files before uvicorn ever sees them, so the pattern silently turns into
several unrelated filenames and the command fails to even parse.
`run.py` calls uvicorn's Python API directly instead, which has no
such issue.)

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

## 4. On-screen keyboard

Toggle it with the button in the bottom-right corner, or press **K**.
Once it's open, up/down/left/right move the cursor over the grid,
confirm types the highlighted key, and backspace deletes.

It doesn't just update the on-page display — it uses `pynput` on the
*server* to synthesize real keystrokes at the OS level, so it types
into whichever window currently has focus (a text editor, another
browser tab, anything). This only works because the typing happens in
the Python backend, which has OS-level access; JavaScript in a browser
tab is sandboxed and cannot do this on its own.

Platform notes:
- **Windows:** works out of the box.
- **macOS:** the terminal (or whatever process runs `uvicorn`) needs
  **Accessibility** permission — System Settings → Privacy & Security →
  Accessibility — or keystrokes silently do nothing.
- **Linux:** requires an X11 session; synthetic input generally isn't
  supported under Wayland.

Arrow keys / Enter / Backspace on your physical keyboard drive the same
code path as the gesture events, so you can test the whole keyboard
(including real OS typing) without a trained classifier.

## 5. What to check as a team

- **Detection reliability:** does tracking stay stable as you move,
  turn your head, or change lighting?
- **Gesture separability:** are the 6 gestures + neutral easy to tell
  apart in practice, or do some get confused with each other? If two
  gestures keep firing as each other, they're probably too similar in
  landmark-feature space — pick more distinct movements for them.
- **Latency:** the telemetry panel shows round-trip ms per frame. If
  it's consistently above ~150-200ms, that's worth addressing (see "If
  it's too slow" below) — it directly affects how responsive the
  on-screen keyboard feels.
- **Multiple machines/webcams:** worth testing on all 3 laptops now,
  since webcam quality and lighting varies.

## 6. If it's too slow (latency > ~200ms)

The current setup sends JPEG frames over a WebSocket to a Python
backend. If that round trip is too slow for responsive gesture control,
the fallback discussed in the project plan is to run MediaPipe's
JavaScript build directly in the browser instead, avoiding the network
hop entirely. Flag this early rather than late — it's a meaningful
rework, not a tweak.

## 7. Project layout

```
facial-gesture-keyboard/
  run.py                 Dev server launcher (use this, not `uvicorn --reload` -
                          see "Run" above for why)
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
  calibration_data.json  (created after your first recorded sample, not checked into git)
```

## 8. Troubleshooting

- **`ModuleNotFoundError`** — make sure the venv is activated before
  running `uvicorn`, and that you're consistently using either the venv
  or a global Python install, not switching between them — installing a
  package into one doesn't make it visible to the other.
- **You see `WatchFiles detected changes in '.venv\...'` in the logs,
  possibly followed by a crash/traceback** — you're running
  `uvicorn main:app --reload` directly instead of `python run.py`. See
  "Run" above; `run.py` exists specifically to stop this.
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
- **Calibration progress looked like it reset** — recorded samples are
  saved to `calibration_data.json` after every sample, so a reconnect
  or an `--reload` restart shouldn't lose them; reopening the page
  should show the real counts again. If it actually looks empty, check
  that `calibration_data.json` exists in the project folder and that
  nothing (e.g. a "Reset All" click) deleted its contents.
- **On-screen keyboard doesn't actually type anywhere** — check the
  platform notes in "On-screen keyboard" above (macOS needs
  Accessibility permission granted to whatever process runs `uvicorn`;
  Wayland on Linux generally doesn't support synthetic keystrokes at
  all). Also make sure the window you want to type into has OS focus —
  the keyboard types wherever focus currently is, not into the browser
  specifically.

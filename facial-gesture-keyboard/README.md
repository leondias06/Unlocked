# Facial Gesture Keyboard — Camera Tracking + Gesture Calibration

This covers the first two working pieces of the hackathon project:

1. **Tracking** — a browser page streams your webcam to a local Python
   server, which runs face landmark detection (MediaPipe) and draws the
   landmarks live on screen.
2. **Calibration + gesture classification** — record a few seconds of
   each of the 9 target gestures plus a neutral face, train a
   lightweight classifier on the spot, and see discrete gesture events
   fire live as you repeat them. Six are for **keyboard mode**
   (up/down/left/right/confirm/backspace); three are for **eye/mouse
   mode** (left_click/right_click/switch_to_keyboard). Only one mode's
   gestures are ever listened to at a time - see "Modes" below.
3. **On-screen keyboard** — matches the reference layout (11x7 grid,
   esc/tab/caps/enter/backspace, a numbers/symbols column, function
   row). It types **real OS-level keystrokes** (via `pynput`), so it
   drives whatever window actually has focus - not just the browser
   tab. See "On-screen keyboard" below for platform notes.
4. **Three modes** (setup / keyboard / eye) - a dashboard you calibrate
   from, a Confirm button that switches into keyboard mode, an
   on-screen key that switches to eye/mouse mode, and a dedicated
   gesture to switch back. See "Modes" below - this is the part most
   worth reading carefully before changing anything here.
5. **Standalone desktop app** — `build.spec` packages the whole thing
   into a single `.exe`: a real app window (no browser, no address
   bar), no Python install required to run it, and no network access
   needed at runtime at all. See "Standalone desktop app" below.

## How the gesture mapping works

There's no hardcoded rule like "eyebrow raise = up". Instead, you pick
*any* facial movement you can reliably repeat for each action — it
doesn't need to resemble the real gesture at all — and the calibration
step records what that looks like in landmark-feature space. A small
k-nearest-neighbors classifier (`gestures.py`) then learns to tell your
10 recorded patterns apart. This matters for the target users: which
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

From the dashboard, click **Calibrate**. For each of the 10 rows -
`neutral`, then the 6 **keyboard-mode** gestures
(`up`, `down`, `left`, `right`, `confirm`, `backspace`), then the 3
**eye-mode** gestures (`left_click`, `right_click`,
`switch_to_keyboard`):

1. Decide what facial movement you'll use for that action.
2. Click **Record**, hold/repeat that movement for a few seconds
   (watch the counter climb — aim for well past the minimum shown).
3. Click **Record** again to stop.

For `neutral`, just hold a relaxed, resting face. Keyboard-mode and
eye-mode gestures are never listened to at the same time (see "Modes"
below), so it's fine if a movement you pick for one feels similar to
one you picked for the other - they'll never be classified against
each other in practice. `switch_to_keyboard` is the *only* gesture
eye mode listens for besides the two clicks; there's no gesture to
leave keyboard mode - that's the on-screen `toggle` key instead.

Once every row is past its minimum, click **Train Classifier**. If it
succeeds, stop recording and just make your gestures naturally — fired
events will appear in the **Live Events** panel in the sidebar, and the
**live prediction** telemetry row shows the raw per-frame classification
before debouncing.

If a gesture isn't firing reliably: click **Clear** on that row and
re-record with a more exaggerated or more consistent version of the
movement — the classifier is only as good as how repeatable the
recorded samples are.

## 4. Modes

There are three modes, and this is the part of the whole project most
likely to be misread, so it's worth being precise about it:

- **setup** - the dashboard/calibration window is visible; nothing
  else is. The app starts here. Reachable from either other mode via
  the small tab docked to the left screen edge.
- **keyboard** - the on-screen keyboard overlay is visible and
  receives the 6 keyboard-mode gestures. Entered from **setup** by
  clicking **Confirm** (which turns the keyboard on *and* minimizes
  the dashboard to the left-edge tab, in one step). The *only* way out
  of keyboard mode is the on-screen **`toggle`** key inside the
  keyboard grid itself (navigate to it, confirm, like any other key) -
  there is deliberately no gesture for this direction.
- **eye** - eye/gaze cursor mode (the actual gaze-to-cursor tracking is
  a separate, later piece of work - this mode already exists and
  already routes its gestures correctly, it just doesn't move the
  cursor with your eyes yet). Receives `left_click`/`right_click` (real
  OS mouse clicks) and `switch_to_keyboard`, which is the *only* way
  back to keyboard mode - deliberately a gesture, not a button, since
  there's no keyboard visible to click one on.

Keyboard-mode gestures and eye-mode gestures are never listened to at
the same time - firing a keyboard-mode gesture while in eye mode (or
vice versa) is simply ignored. See `windows.DesktopWindows` for the
actual state machine; `on_gesture()` there is the one place that
enforces this.

## 5. On-screen keyboard

The keyboard is its own page/window (`static/keyboard.html` +
`keyboard.js`), separate from the dashboard/calibration page - in the
standalone desktop app it's a real, always-on-top overlay so it can
float over whatever app you're actually typing into. In plain-browser
dev mode there's no window management to speak of - open
`/keyboard.html` directly in a second tab to exercise it standalone.

up/down/left/right move the cursor over the grid, confirm types the
highlighted key, backspace deletes. Typed text also lands in a real,
selectable/copyable textbox right in the keyboard panel - that part is
plain client-side JavaScript, so it works identically no matter how
this is hosted or run.

**Held-gesture auto-repeat:** up/down/left/right and backspace behave
like a held key on a physical keyboard - hold the gesture and, after a
short initial delay (~450ms), it keeps firing on its own every ~150ms
until you release, instead of requiring a full return-to-neutral
between every single step. This is what actually makes scanning across
an 11x7 grid fast enough to use; without it, moving from one corner to
the other means holding-and-releasing the same gesture a dozen times.
confirm and the eye-mode gestures (left/right click,
`switch_to_keyboard`) deliberately stay one-shot-per-hold - repeating
those while held would mean an accidental extra keystroke, click, or
mode flicker. See `GestureDebouncer` in `gestures.py`.

The layout is an 11x7 grid matching the reference design: plain A-Z
reading order (not QWERTY) with a numbers/symbols column, `caps` and
`enter` each spanning two rows, `esc`/`tab`/`backspace` as real
keystrokes, and a bottom function row. **caps** is a local shift-style
toggle - it doesn't touch the real OS caps lock state, it just decides
whether confirming a letter/number key types the upper or lower
variant (numbers show their shifted symbol, e.g. `1`/`!`, the same
way). The top row's 5 gradient cells are real predictive-text
suggestions now: a bundled offline word list (no network calls, same as
everything else here) is prefix-matched against whatever you're
currently typing (the run of letters since the last space), ranked
most-common-first, and confirming one backspaces out the in-progress
word and types the full suggestion plus a trailing space - both in the
on-page textbox and, via the same real keystrokes as everything else,
into whatever window has OS focus. It's a small hand-picked ~1000-word
pool (common English words plus everyday-needs vocabulary like
"hungry"/"bathroom"/"hurt", since this app exists for people who may
have no other way to communicate a basic need) rather than a proper
frequency corpus, so treat it as a solid demo-quality baseline, not a
finished autocomplete engine - swapping in a real frequency-ranked word
list (or per-user learning) is a natural next step. The bottom row's
plain cells are intentionally blank custom keys
(navigable, selectable, do nothing - no setup UI, per spec); `sound
up/down` send real OS media-key presses, `brightness up/down` adjust
the display via WMI where the hardware supports it (most laptop
panels; many external monitors don't) and silently no-op otherwise.
`toggle` switches to eye mode (see "Modes" above) - this is real and
wired up, not a placeholder. `on | off` still is.

Separately, it *also* uses `pynput` on the *server* to synthesize real
keystrokes at the OS level, so it can type into whichever window
currently has focus on that machine (a text editor, another browser
tab, anything) - not just the on-page textbox. This only works because
the typing happens in the Python backend, which has OS-level access;
JavaScript in a browser tab is sandboxed and cannot do this on its own.
**This is inherently local-machine-only** - it types wherever the
*server's* OS focus is, so it only makes sense when the server and the
person using it are on the same computer (which is exactly the
standalone desktop app below). It cannot and never will work for a
visitor connecting to a server hosted elsewhere - no web technology
allows a remote server to control a visitor's OS keyboard, for the
obvious security reason.

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

## 6. Standalone desktop app

`python run.py` is great for development, but it's still "open a
terminal, run a command, open a browser tab." `desktop_app.py` +
`build.spec` package the whole app into three real native windows (via
[pywebview](https://pywebview.flowrite.com/)) instead:

The app is a small state machine with three modes - `setup`, `keyboard`
and `eye` (see "Modes" above) - and the three windows just reflect
whichever mode is active:

1. The **launch/calibration window** starts in `setup` mode - camera +
   calibration UI (now behind a dashboard, see "Modes") + a **Confirm**
   button.
2. Clicking Confirm hides that window and switches to `keyboard` mode:
   the **keyboard overlay** appears, and a small **tab docked to the
   left screen edge** appears with it. That tab's only job now is to
   reopen calibration (back to `setup` mode) - it does not toggle the
   keyboard.
3. From `keyboard` mode, the keyboard's own on-screen **toggle key**
   switches to `eye` mode (keyboard hides). From `eye` mode, the
   `switch_to_keyboard` gesture switches back to `keyboard` mode (no
   on-screen button for this direction, by design - see "Modes" above
   for why the split is deliberate). Gestures are mode-gated: the six
   keyboard-navigation gestures only act while in `keyboard` mode, and
   `left_click`/`right_click`/`switch_to_keyboard` only act while in
   `eye` mode, so a stray gesture from the "wrong" mode is silently
   ignored rather than doing something unexpected.

The keyboard overlay is engineered to never steal OS keyboard focus
when it appears, so gesture-typed keystrokes keep landing in whatever
app (Word, a browser, ...) you were actually using - see `windows.py`
for exactly how (`WS_EX_NOACTIVATE` plus an immediate focus-restore,
since `Show()` always activates a window once regardless of that
style; verified this holds even with the overlaid page actively
re-rendering, and separately verified that hiding a window does **not**
throttle its camera/JS loop the way a backgrounded browser tab would,
so gesture recognition keeps working while a window is hidden).

No Python install is needed to run the built `.exe`, and it makes **no
network calls at all** at runtime - the MediaPipe model is bundled into
the build instead of downloaded on first run.

Known follow-up: the overlay windows currently use solid (not truly
see-through) translucent-styled backgrounds - real OS-level window
transparency (`transparent=True`) wasn't confirmed working reliably in
testing, so it was left for a later pass rather than risk a broken
window.

Build it (from an activated venv):
```powershell
pip install -r requirements-build.txt
pyinstaller build.spec
```
This produces `dist/FacialGestureKeyboard.exe` (~200MB - mediapipe,
opencv and scikit-learn are large; this is normal). Copy that one file
anywhere and double-click it.

**First launch will show a real camera-permission prompt** (the same
kind a browser shows) - click **Allow**. This is expected, not a bug.

Calibration data for the packaged app is saved to
`%APPDATA%\FacialGestureKeyboard\calibration_data.json` rather than
next to the script, since a `.exe` built with PyInstaller's "onefile"
mode re-extracts itself to a fresh temp folder on every launch - saving
next to the script would silently lose all calibration every time you
closed the app.

If something goes wrong and you need to see errors: edit `build.spec`,
set `console=True` in the `EXE(...)` call, rebuild, and run it from a
terminal instead of double-clicking - the console window will show
tracebacks that a windowed app otherwise swallows.

## 7. What to check as a team

- **Detection reliability:** does tracking stay stable as you move,
  turn your head, or change lighting?
- **Gesture separability:** are the 9 gestures + neutral easy to tell
  apart in practice, or do some get confused with each other? If two
  gestures keep firing as each other, they're probably too similar in
  landmark-feature space — pick more distinct movements for them.
- **Latency:** the telemetry panel shows round-trip ms per frame. If
  it's consistently above ~150-200ms, that's worth addressing (see "If
  it's too slow" below) — it directly affects how responsive the
  on-screen keyboard feels.
- **Multiple machines/webcams:** worth testing on all 3 laptops now,
  since webcam quality and lighting varies.

## 8. If it's too slow (latency > ~200ms)

The current setup sends JPEG frames over a WebSocket to a Python
backend. If that round trip is too slow for responsive gesture control,
the fallback discussed in the project plan is to run MediaPipe's
JavaScript build directly in the browser instead, avoiding the network
hop entirely. Flag this early rather than late — it's a meaningful
rework, not a tweak.

## 9. Project layout

```
facial-gesture-keyboard/
  run.py                 Dev server launcher (use this, not `uvicorn --reload` -
                          see "Run" above for why)
  desktop_app.py          Standalone-app entry point: runs the server in a
                          background thread, creates the 3 native windows
  windows.py              Window orchestration: confirm/minimize, the
                          left-edge toggle tab, and the focus-preserving
                          keyboard overlay show/hide
  build.spec              PyInstaller build config for the standalone .exe
  main.py               FastAPI server: WebSocket endpoint, MediaPipe inference,
                         calibration protocol handling
  gestures.py            Feature extraction, calibration storage, classifier
                          training/prediction, and gesture-event debouncing
                          (unit-testable without a camera or the ML model)
  requirements.txt
  requirements-build.txt  Adds pyinstaller, only needed to build the .exe
  static/
    index.html            Launch/calibration page (tracking, telemetry,
                          calibration panel, Confirm button)
    keyboard.html/.js      The on-screen keyboard - its own page/window
    toggle.html             The small left-edge "reopen calibration" tab
    style.css              Visual design (shared by all three pages)
    app.js                  Camera capture, WebSocket client, landmark drawing,
                            calibration UI wiring
  face_landmarker.task   (auto-downloaded on first run, not checked into git)
  calibration_data.json  (created after your first recorded sample, not checked into git)
  build/, dist/           PyInstaller output (not checked into git - see
                          "Standalone desktop app" above)
```

## 10. Troubleshooting

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
  specifically. The textbox inside the keyboard panel itself always
  works regardless, since that part doesn't depend on OS-level access.
- **You changed `static/app.js` or `style.css` and don't see the
  change** — those don't trigger a server reload (only `.py` edits do),
  and the browser can cache them. Hard-refresh (Ctrl+Shift+R). The dev
  server sends `Cache-Control: no-store` specifically so this class of
  confusion shouldn't recur, but a very old already-open tab may still
  have stale JS in memory from before that was added.
- **The standalone `.exe` won't build / fails with a missing-module
  error** — `pyinstaller build.spec` collects mediapipe, pynput and
  webview's data files automatically, but if a newer version of one of
  those packages changes how it loads resources, PyInstaller may miss
  something new. Rebuild with `console=True` (see "Standalone desktop
  app" above) to see the actual traceback instead of a silent exit.
- **`pyinstaller build.spec` fails with `PermissionError: [WinError 5]
  Access is denied`** — this means a previous build of
  `FacialGestureKeyboard.exe` is still running (including a copy you
  launched yourself to test) and Windows has the file locked. Close
  every running instance first, then rebuild.
- **The `.exe` opens but the window closes immediately / never shows
  the page** — same fix: flip `console=True` in `build.spec`, rebuild,
  and run from a terminal to see what actually failed.

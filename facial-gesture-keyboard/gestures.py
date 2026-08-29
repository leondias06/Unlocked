"""
gestures.py - turns raw MediaPipe face landmarks into discrete gesture
events (up / down / left / right / confirm / backspace).

Deliberately does NOT hardcode "eyebrow raise = up" or similar rules.
Instead: during calibration, the user performs whatever facial movement
they want to represent each action, we record the resulting feature
vectors, and a small classifier learns to tell them apart. This matters
for the target users (stroke, locked-in syndrome, paralysis) - which
facial movements are reliably controllable varies a lot per person, so
the mapping needs to be learned per user, not assumed by us.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------- config

KEYBOARD_MODE_GESTURES = ["up", "down", "left", "right", "confirm", "backspace"]
EYE_MODE_GESTURES = ["left_click", "right_click", "switch_to_keyboard"]

# Keyboard-mode and eye-mode gestures are mutually exclusive by design
# (only one mode's gestures are ever listened to at a time - see
# windows.DesktopWindows.on_gesture) but are still calibrated as
# distinct labels here, so the same physical facial movement isn't
# forced to mean two different things.
GESTURE_LABELS = [*KEYBOARD_MODE_GESTURES, *EYE_MODE_GESTURES]
NEUTRAL_LABEL = "neutral"
ALL_LABELS = [NEUTRAL_LABEL, *GESTURE_LABELS]

MIN_SAMPLES_PER_LABEL = 15   # ~0.75s of frames at 20fps; calibration UI collects more
K_NEIGHBORS = 5
CONFIDENCE_THRESHOLD = 0.7   # predictions below this are treated as "unsure" -> neutral
HOLD_FRAMES_TO_FIRE = 5      # consecutive matching predictions needed before a gesture fires
NEUTRAL_FRAMES_TO_REARM = 3  # consecutive neutral frames needed before the next gesture can fire

# These three were tuned up together (from 0.6 / 3 / 2) after reports of
# wrong gestures firing - a single noisy/borderline frame (classifier
# blips happen especially near two gestures' decision boundary) used to
# be enough to fire the wrong action. Requiring more consecutive matching
# frames plus a higher confidence floor filters out that kind of blip
# without meaningfully hurting responsiveness: at ~20fps, 5 frames is
# ~250ms, still fast for a deliberate hold. If misfires are still
# happening after this, the next lever is re-calibrating with more
# distinct/exaggerated movements per gesture (see README troubleshooting)
# rather than pushing these numbers even higher, since past a point that
# just makes every gesture feel sluggish instead of fixing the real
# separability problem.

# Gestures that auto-repeat while held, instead of requiring a return to
# neutral between each fire - navigation and backspace behave like a
# held arrow/delete key on a physical keyboard, which matters a lot for
# scanning speed across an 11x7 grid. confirm/click/mode-switch
# deliberately stay one-shot-per-hold: repeating those while held would
# mean an accidental extra keystroke, extra click, or a mode flicker.
REPEATABLE_LABELS = {"up", "down", "left", "right", "backspace"}
REPEAT_INITIAL_DELAY_S = 0.45  # time held before auto-repeat kicks in, like OS key-repeat
REPEAT_INTERVAL_S = 0.15       # time between repeats once it's kicked in

# ---------------------------------------------------------------- landmark indices
#
# Standard MediaPipe FaceMesh canonical topology (468 points, +iris if
# refined). These groupings are also used for the tracking-view overlay.

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW = [70, 63, 105]
RIGHT_BROW = [300, 293, 334]
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_UPPER = 13
MOUTH_LOWER = 14
LEFT_CHEEK = 116
RIGHT_CHEEK = 345
NOSE_TIP = 1

FEATURE_NAMES = [
    "left_eye_open", "right_eye_open",
    "left_brow_raise", "right_brow_raise",
    "mouth_open", "mouth_width", "mouth_corner_asym",
    "head_yaw", "head_roll",
    "cheek_left", "cheek_right",
]


def _pt(landmarks: list[dict], i: int) -> np.ndarray:
    p = landmarks[i]
    return np.array([p["x"], p["y"]], dtype=np.float64)


def _center(landmarks: list[dict], indices: list[int]) -> np.ndarray:
    pts = np.array([[landmarks[i]["x"], landmarks[i]["y"]] for i in indices])
    return pts.mean(axis=0)


def landmarks_to_features(landmarks: list[dict]) -> list[float]:
    """
    Convert one frame of raw landmarks into an ~11-dim feature vector,
    normalized by inter-eye distance so it's roughly invariant to how
    close the person is sitting to the camera.
    """
    left_eye_c = _center(landmarks, LEFT_EYE)
    right_eye_c = _center(landmarks, RIGHT_EYE)
    scale = float(np.linalg.norm(left_eye_c - right_eye_c)) or 1e-6

    left_eye_open = np.linalg.norm(_pt(landmarks, LEFT_EYE[1]) - _pt(landmarks, LEFT_EYE[5])) / scale
    right_eye_open = np.linalg.norm(_pt(landmarks, RIGHT_EYE[1]) - _pt(landmarks, RIGHT_EYE[5])) / scale

    left_brow_c = _center(landmarks, LEFT_BROW)
    right_brow_c = _center(landmarks, RIGHT_BROW)
    left_brow_raise = np.linalg.norm(left_brow_c - left_eye_c) / scale
    right_brow_raise = np.linalg.norm(right_brow_c - right_eye_c) / scale

    mouth_open = np.linalg.norm(_pt(landmarks, MOUTH_UPPER) - _pt(landmarks, MOUTH_LOWER)) / scale
    mouth_l = _pt(landmarks, MOUTH_LEFT_CORNER)
    mouth_r = _pt(landmarks, MOUTH_RIGHT_CORNER)
    mouth_width = np.linalg.norm(mouth_l - mouth_r) / scale
    mouth_corner_asym = (mouth_l[1] - mouth_r[1]) / scale  # captures smirk/asymmetric corner movement

    nose = _pt(landmarks, NOSE_TIP)
    eye_mid = (left_eye_c + right_eye_c) / 2
    head_yaw = (nose[0] - eye_mid[0]) / scale

    eye_line = right_eye_c - left_eye_c
    head_roll = float(np.arctan2(eye_line[1], eye_line[0]))

    cheek_left = np.linalg.norm(_pt(landmarks, LEFT_CHEEK) - nose) / scale
    cheek_right = np.linalg.norm(_pt(landmarks, RIGHT_CHEEK) - nose) / scale

    return [
        float(left_eye_open), float(right_eye_open),
        float(left_brow_raise), float(right_brow_raise),
        float(mouth_open), float(mouth_width), float(mouth_corner_asym),
        float(head_yaw), head_roll,
        float(cheek_left), float(cheek_right),
    ]


def head_pose(landmarks: list[dict]) -> tuple[float, float]:
    """
    Continuous head yaw/pitch for cursor steering in eye mode.

    Deliberately kept separate from landmarks_to_features() above: that
    vector is fixed-shape and calibrated per-user (changing its
    dimensionality would break every saved calibration_data.json), while
    this needs no calibration at all and runs every frame regardless of
    what's been trained. Duplicating the yaw math is a small price for
    keeping cursor steering fully decoupled from the gesture classifier.

    Returns (yaw, pitch), normalized by inter-eye distance so it's
    roughly invariant to how close you're sitting to the camera. Both
    are near 0 for a centered/neutral head pose; positive yaw = nose
    shifted right of the eyes (head turned/tilted right), positive pitch
    = nose shifted below the eyes (head tilted/nodded down).
    """
    left_eye_c = _center(landmarks, LEFT_EYE)
    right_eye_c = _center(landmarks, RIGHT_EYE)
    scale = float(np.linalg.norm(left_eye_c - right_eye_c)) or 1e-6
    eye_mid = (left_eye_c + right_eye_c) / 2
    nose = _pt(landmarks, NOSE_TIP)
    yaw = (nose[0] - eye_mid[0]) / scale
    pitch = (nose[1] - eye_mid[1]) / scale
    return float(yaw), float(pitch)


# ---------------------------------------------------------------- calibration store

def _default_calibration_path() -> Path:
    # When frozen (PyInstaller), __file__ resolves inside the onefile
    # build's temp extraction dir, which is wiped and recreated on every
    # launch - saving there would silently lose all calibration data
    # every time the app closes. Use a stable per-user location instead.
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("APPDATA", Path.home())) / "FacialGestureKeyboard"
        base.mkdir(parents=True, exist_ok=True)
        return base / "calibration_data.json"
    return Path(__file__).parent / "calibration_data.json"


DEFAULT_CALIBRATION_PATH = _default_calibration_path()


class CalibrationStore:
    """
    Holds labeled feature-vector samples collected during calibration,
    and the classifier trained from them.

    Persisted to a JSON file next to this module so recorded samples
    survive a server restart - including `uvicorn --reload` restarting
    the process whenever a file changes, which otherwise silently wipes
    an in-memory-only store mid-calibration.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CALIBRATION_PATH
        self.samples: dict[str, list[list[float]]] = {label: [] for label in ALL_LABELS}
        self.classifier: KNeighborsClassifier | None = None
        self.ready: bool = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return  # corrupt or unreadable - start fresh rather than crash startup

        for label in ALL_LABELS:
            samples = data.get(label)
            if isinstance(samples, list):
                self.samples[label] = samples

        counts = self.counts()
        if all(c >= MIN_SAMPLES_PER_LABEL for c in counts.values()):
            self.train()  # picks up where a previously-trained session left off

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.samples))
        except OSError:
            pass  # persistence is a nice-to-have; don't break calibration over it

    def add_sample(self, label: str, features: list[float]) -> int:
        if label not in self.samples:
            return 0
        self.samples[label].append(features)
        self._save()
        return len(self.samples[label])

    def clear_label(self, label: str) -> None:
        if label in self.samples:
            self.samples[label] = []
            self.ready = False
            self._save()

    def clear_all(self) -> None:
        for label in self.samples:
            self.samples[label] = []
        self.classifier = None
        self.ready = False
        self._save()

    def counts(self) -> dict[str, int]:
        return {label: len(s) for label, s in self.samples.items()}

    def train(self) -> dict:
        counts = self.counts()
        missing = [label for label, c in counts.items() if c < MIN_SAMPLES_PER_LABEL]
        if missing:
            self.ready = False
            return {
                "status": "error",
                "message": f"Need at least {MIN_SAMPLES_PER_LABEL} samples for: {', '.join(missing)}",
                "counts": counts,
            }

        X: list[list[float]] = []
        y: list[str] = []
        for label, samples in self.samples.items():
            X.extend(samples)
            y.extend([label] * len(samples))

        k = min(K_NEIGHBORS, min(counts.values()))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        clf.fit(X, y)

        self.classifier = clf
        self.ready = True
        return {"status": "ok", "message": "Classifier trained.", "counts": counts}

    def predict(
        self, features: list[float], allowed_labels: set[str] | None = None
    ) -> tuple[str | None, float]:
        """
        allowed_labels restricts which labels can win, e.g. while the
        desktop app is in keyboard mode only the 6 keyboard-mode labels
        (+ neutral) are ever candidates - left_click/right_click/
        switch_to_keyboard aren't just ignored downstream, they're not
        competing for probability mass at all. This matters beyond
        tidiness: without it, a frame that's genuinely a clean "confirm"
        can still lose to a superficially-similar out-of-mode gesture in
        the raw argmax, so keyboard-mode presses get eaten by an eye-
        mode gesture that could never legally fire anyway. Restricting
        first means the confidence returned reflects how well the frame
        matches the gestures that are actually reachable right now.
        None (the default - dev/browser mode with no concept of "mode",
        or the setup/calibration screen, which needs to keep recognizing
        every label) means no restriction, matching the old behavior.
        """
        if not self.ready or self.classifier is None:
            return None, 0.0
        proba = self.classifier.predict_proba([features])[0]
        classes = self.classifier.classes_

        if allowed_labels is None:
            idx = int(np.argmax(proba))
            return classes[idx], float(proba[idx])

        allowed_mask = np.array([c in allowed_labels for c in classes])
        if not allowed_mask.any():
            return None, 0.0
        idx = int(np.argmax(np.where(allowed_mask, proba, -1.0)))
        return classes[idx], float(proba[idx])


# ---------------------------------------------------------------- debouncing

class GestureDebouncer:
    """
    Turns a raw per-frame (label, confidence) prediction stream into
    discrete "fire" events. Without this, a held expression would spam
    repeated keystrokes; this makes it behave like a key press instead
    of a key that auto-repeats while held.

    One instance per live connection/session - it's stateful.
    """

    def __init__(self) -> None:
        self.hold_label: str | None = None
        self.hold_count: int = 0
        self.neutral_count: int = 0
        self.armed: bool = True
        self.fired_at: float | None = None       # monotonic time of this hold's first fire
        self.last_repeat_at: float | None = None

    def update(self, pred: str | None, confidence: float) -> str | None:
        """Feed one prediction in. Returns a gesture label if one just fired."""
        is_neutral_ish = pred is None or pred == NEUTRAL_LABEL or confidence < CONFIDENCE_THRESHOLD

        if is_neutral_ish:
            self.neutral_count += 1
            self.hold_label, self.hold_count = None, 0
            self.fired_at, self.last_repeat_at = None, None
            if self.neutral_count >= NEUTRAL_FRAMES_TO_REARM:
                self.armed = True
            return None

        self.neutral_count = 0
        if pred == self.hold_label:
            self.hold_count += 1
        else:
            self.hold_label, self.hold_count = pred, 1
            self.fired_at, self.last_repeat_at = None, None

        if self.armed and self.hold_count >= HOLD_FRAMES_TO_FIRE:
            self.armed = False
            self.hold_count = 0
            now = time.monotonic()
            self.fired_at, self.last_repeat_at = now, now
            return pred

        # Still holding the same gesture past its initial fire: for
        # repeatable labels, keep re-firing at a fixed interval instead
        # of waiting for a return to neutral, like a held OS key.
        if (
            not self.armed
            and pred == self.hold_label
            and pred in REPEATABLE_LABELS
            and self.fired_at is not None
            and self.last_repeat_at is not None
        ):
            now = time.monotonic()
            if (
                now - self.fired_at >= REPEAT_INITIAL_DELAY_S
                and now - self.last_repeat_at >= REPEAT_INTERVAL_S
            ):
                self.last_repeat_at = now
                return pred

        return None
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
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------- config

GESTURE_LABELS = ["up", "down", "left", "right", "confirm", "backspace"]
NEUTRAL_LABEL = "neutral"
ALL_LABELS = [NEUTRAL_LABEL, *GESTURE_LABELS]

MIN_SAMPLES_PER_LABEL = 15   # ~0.75s of frames at 20fps; calibration UI collects more
K_NEIGHBORS = 5
CONFIDENCE_THRESHOLD = 0.6   # predictions below this are treated as "unsure" -> neutral
HOLD_FRAMES_TO_FIRE = 3      # consecutive matching predictions needed before a gesture fires
NEUTRAL_FRAMES_TO_REARM = 2  # consecutive neutral frames needed before the next gesture can fire

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


# ---------------------------------------------------------------- calibration store

DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "calibration_data.json"


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

    def predict(self, features: list[float]) -> tuple[str | None, float]:
        if not self.ready or self.classifier is None:
            return None, 0.0
        pred = self.classifier.predict([features])[0]
        proba = self.classifier.predict_proba([features])[0]
        confidence = float(max(proba))
        return pred, confidence


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

    def update(self, pred: str | None, confidence: float) -> str | None:
        """Feed one prediction in. Returns a gesture label if one just fired."""
        is_neutral_ish = pred is None or pred == NEUTRAL_LABEL or confidence < CONFIDENCE_THRESHOLD

        if is_neutral_ish:
            self.neutral_count += 1
            self.hold_label, self.hold_count = None, 0
            if self.neutral_count >= NEUTRAL_FRAMES_TO_REARM:
                self.armed = True
            return None

        self.neutral_count = 0
        if pred == self.hold_label:
            self.hold_count += 1
        else:
            self.hold_label, self.hold_count = pred, 1

        if self.armed and self.hold_count >= HOLD_FRAMES_TO_FIRE:
            self.armed = False
            self.hold_count = 0
            return pred

        return None
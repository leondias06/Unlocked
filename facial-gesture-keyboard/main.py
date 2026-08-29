"""
Facial Gesture Keyboard - CV pipeline server.

Handles two things now:
  1. Camera tracking: MediaPipe Face Mesh landmark detection (step 1).
  2. Gesture calibration + classification: turning landmark movement
     into discrete up/down/left/right/confirm/backspace events (step 2).

See gestures.py for the feature extraction, calibration storage, and
debouncing logic - kept separate so it can be unit-tested without a
camera or the MediaPipe model.
"""

import base64
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import gestures

# --- MediaPipe Face Landmarker setup -----------------------------------
#
# Note for the team: MediaPipe's older `mp.solutions.face_mesh` API
# (what most tutorials/blog posts show) has been replaced by the newer
# Tasks API in current MediaPipe releases (pip installs 1.0.x now).
# Same underlying model quality, different Python interface - this file
# uses the current one. It needs a small model file (face_landmarker.task,
# ~3-4MB) which is downloaded automatically the first time you run this
# server (requires internet on first run only; cached locally after that).

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"[setup] Downloading face landmark model to {MODEL_PATH.name} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[setup] Model downloaded.")


ensure_model()

_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.FaceLandmarker.create_from_options(_options)

app = FastAPI()

# Single shared calibration store - this is a single-user prototype, so
# calibration data is intentionally global rather than per-connection.
calibration_store = gestures.CalibrationStore()

# Landmark index groups we'll care about once we build gesture
# classification. Kept here now so the frontend can already highlight
# these regions distinctly during tracking. Indices are MediaPipe's
# fixed FaceMesh topology (0-467).
KEY_REGIONS = {
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    "mouth": [61, 291, 13, 14, 78, 308],
    "left_cheek": [116, 123],
    "right_cheek": [345, 352],
    "nose_tip": [1],
    "eyebrows": [70, 63, 105, 300, 293, 334],
}


def decode_data_url(data_url: str) -> np.ndarray | None:
    """Decode a 'data:image/jpeg;base64,...' string into a BGR frame."""
    try:
        _, encoded = data_url.split(",", 1)
        binary = base64.b64decode(encoded)
        arr = np.frombuffer(binary, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def extract_landmarks(frame: np.ndarray) -> list[dict] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]
    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face]


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Tell the client what gestures/thresholds exist so the calibration
    # UI can be built from this single source of truth instead of
    # duplicating label names and constants in JS.
    await websocket.send_text(json.dumps({
        "type": "config",
        "all_labels": gestures.ALL_LABELS,
        "neutral_label": gestures.NEUTRAL_LABEL,
        "gesture_labels": gestures.GESTURE_LABELS,
        "min_samples_per_label": gestures.MIN_SAMPLES_PER_LABEL,
    }))

    # Per-connection state: which label (if any) is currently being
    # recorded during calibration, and the debouncer that turns live
    # predictions into discrete fired gesture events.
    capture_label: str | None = None
    debouncer = gestures.GestureDebouncer()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            # --- calibration control messages -----------------------

            if msg_type == "set_label":
                capture_label = msg.get("label")  # a gesture name, or None to stop capturing
                count = len(calibration_store.samples.get(capture_label, [])) if capture_label else 0
                await websocket.send_text(json.dumps({
                    "type": "label_set", "label": capture_label, "count": count,
                }))
                continue

            if msg_type == "reset_label_samples":
                calibration_store.clear_label(msg.get("label", ""))
                await websocket.send_text(json.dumps({
                    "type": "label_set", "label": msg.get("label"), "count": 0,
                }))
                continue

            if msg_type == "reset_all":
                calibration_store.clear_all()
                await websocket.send_text(json.dumps({"type": "reset_ok"}))
                continue

            if msg_type == "train":
                result = calibration_store.train()
                await websocket.send_text(json.dumps({"type": "train_result", **result}))
                continue

            # --- video frame: tracking + (capture or live classify) --

            if msg_type == "frame":
                frame = decode_data_url(msg.get("frame", ""))
                if frame is None:
                    await websocket.send_text(json.dumps({"type": "tracking", "status": "bad_frame"}))
                    continue

                landmarks = extract_landmarks(frame)
                if landmarks is None:
                    await websocket.send_text(json.dumps({"type": "tracking", "status": "no_face"}))
                    continue

                response = {
                    "type": "tracking",
                    "status": "ok",
                    "landmark_count": len(landmarks),
                    "landmarks": landmarks,
                    "key_regions": KEY_REGIONS,
                }

                features = gestures.landmarks_to_features(landmarks)

                if capture_label:
                    # Calibration mode: just record the sample, no classification.
                    count = calibration_store.add_sample(capture_label, features)
                    response["capture_label"] = capture_label
                    response["capture_count"] = count
                elif calibration_store.ready:
                    # Live mode: classify, then debounce into a discrete event.
                    pred, confidence = calibration_store.predict(features)
                    response["prediction"] = pred
                    response["confidence"] = round(confidence, 2)

                    fired = debouncer.update(pred, confidence)
                    if fired:
                        await websocket.send_text(json.dumps({
                            "type": "gesture", "label": fired, "confidence": round(confidence, 2),
                        }))

                await websocket.send_text(json.dumps(response))
                continue

    except WebSocketDisconnect:
        pass


# Serve the frontend (index.html, style.css, app.js) from /static at the
# site root, so opening http://localhost:8000 works out of the box and
# the camera runs in a browser "secure context" (localhost counts as one).
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

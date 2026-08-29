// Facial Gesture Keyboard - CV pipeline frontend.
//
// Responsibilities:
//   1. Get webcam access and show a live preview.
//   2. Stream downscaled JPEG frames to the backend over a WebSocket.
//   3. Draw the returned landmarks on top of the video as visual proof
//      the tracking pipeline is working.
//   4. Drive the calibration UI: record samples per gesture, train the
//      classifier, and show fired gesture events live.
//
// Gesture label names, ordering, and the minimum-samples threshold all
// come from the server's "config" message (single source of truth in
// gestures.py) rather than being duplicated here.

const REGION_COLORS = {
  left_eye: "#4CC9F0",
  right_eye: "#7DE2D1",
  mouth: "#FF6B6B",
  eyebrows: "#F5A623",
  left_cheek: "#B892FF",
  right_cheek: "#F783AC",
  nose_tip: "#FFFFFF",
};

const SEND_WIDTH = 480;
const SEND_HEIGHT = 360;
const SEND_INTERVAL_MS = 50; // ~20 fps to the server - lower this and gestures take longer to register

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const stageHint = document.getElementById("stageHint");

const connIndicator = document.getElementById("connIndicator");
const connLabel = document.getElementById("connLabel");

const mFace = document.getElementById("mFace");
const mPoints = document.getElementById("mPoints");
const mFps = document.getElementById("mFps");
const mLatency = document.getElementById("mLatency");
const mPrediction = document.getElementById("mPrediction");

const eventLog = document.getElementById("eventLog");
const eventLogEmpty = document.getElementById("eventLogEmpty");

const calRows = document.getElementById("calRows");
const trainBtn = document.getElementById("trainBtn");
const resetAllBtn = document.getElementById("resetAllBtn");
const trainStatus = document.getElementById("trainStatus");

const dashboardView = document.getElementById("dashboardView");
const calibrationView = document.getElementById("calibrationView");
const goToCalibrationBtn = document.getElementById("goToCalibrationBtn");
const dashCalibratedCount = document.getElementById("dashCalibratedCount");
const dashTrainedStatus = document.getElementById("dashTrainedStatus");

const sendCanvas = document.createElement("canvas");
sendCanvas.width = SEND_WIDTH;
sendCanvas.height = SEND_HEIGHT;
const sendCtx = sendCanvas.getContext("2d");

let ws = null;
let lastLandmarks = null;   // most recent landmark array from the server
let keyRegions = null;      // region-name -> [landmark indices], sent once by server
let sendTimestamps = [];    // queue for round-trip latency measurement

let msgCount = 0;
let fpsWindowStart = performance.now();
let scanPhase = 0;

// calibration/gesture state
let allLabels = [];          // e.g. ["neutral","up","down","left","right","confirm","backspace"]
let neutralLabel = "neutral";
let keyboardModeGestures = [];
let eyeModeGestures = [];
let minSamplesPerLabel = 15; // overwritten by server config
let activeCaptureLabel = null;
let rowEls = {};             // label -> { root, fill, count, recordBtn }
let latestCounts = {};       // label -> sample count, kept for gating checks outside handleConfig

// ---------------------------------------------------------------- camera

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;

    stageHint.classList.add("is-hidden");
    requestAnimationFrame(renderLoop);
    setInterval(sendFrame, SEND_INTERVAL_MS);
  } catch (err) {
    stageHint.textContent =
      "Camera access failed: " + err.message + " — check browser permissions.";
  }
}

// ---------------------------------------------------------------- websocket

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => setConnected(true);
  ws.onclose = () => {
    setConnected(false);
    setTimeout(connectWS, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (evt) => handleMessage(JSON.parse(evt.data));
}

function setConnected(isLive) {
  connIndicator.classList.toggle("is-live", isLive);
  connLabel.textContent = isLive ? "connected" : "connecting";
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || video.readyState < 2) return;
  sendCtx.drawImage(video, 0, 0, SEND_WIDTH, SEND_HEIGHT);
  const dataUrl = sendCanvas.toDataURL("image/jpeg", 0.6);
  sendTimestamps.push(performance.now());
  ws.send(JSON.stringify({ type: "frame", frame: dataUrl }));
}

function send(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(obj));
}

function handleMessage(msg) {
  switch (msg.type) {
    case "config":
      handleConfig(msg);
      break;
    case "tracking":
      handleTracking(msg);
      break;
    case "label_set":
      handleLabelSet(msg);
      break;
    case "train_result":
      handleTrainResult(msg);
      break;
    case "gesture":
      handleGestureEvent(msg);
      break;
    case "reset_ok":
      handleResetOk();
      break;
    default:
      // unknown message type - ignore rather than throw, servers may
      // add fields/messages over time
      break;
  }
}

function handleTracking(msg) {
  // latency + fps bookkeeping: exactly one "tracking" message per frame
  // sent, so this stays correctly paired with sendTimestamps.
  const sentAt = sendTimestamps.shift();
  if (sentAt !== undefined) {
    const rtt = Math.round(performance.now() - sentAt);
    mLatency.textContent = `${rtt} ms`;
    mLatency.classList.toggle("is-alert", rtt > 250);
  }

  msgCount++;
  const now = performance.now();
  if (now - fpsWindowStart > 1000) {
    mFps.textContent = msgCount.toString();
    msgCount = 0;
    fpsWindowStart = now;
  }

  if (msg.status === "ok") {
    lastLandmarks = msg.landmarks;
    keyRegions = msg.key_regions;
    mFace.textContent = "yes";
    mFace.classList.remove("is-idle", "is-alert");
    mPoints.textContent = msg.landmark_count;

    if (msg.capture_label) {
      // currently recording calibration samples for this label
      updateRowProgress(msg.capture_label, msg.capture_count);
    } else if (msg.prediction !== undefined) {
      // live classification readout
      const conf = msg.confidence !== undefined ? ` (${Math.round(msg.confidence * 100)}%)` : "";
      mPrediction.textContent = (msg.prediction || "—") + conf;
    } else {
      mPrediction.textContent = "not calibrated";
    }
  } else if (msg.status === "no_face") {
    lastLandmarks = null;
    mFace.textContent = "no";
    mFace.classList.add("is-alert");
    mFace.classList.remove("is-idle");
    mPoints.textContent = "0";
  } else {
    // bad_frame or anything unexpected - don't spam state, just note it
    mFace.textContent = "—";
    mFace.classList.add("is-idle");
  }
}

// ---------------------------------------------------------------- calibration UI

function handleConfig(msg) {
  allLabels = msg.all_labels;
  neutralLabel = msg.neutral_label;
  keyboardModeGestures = msg.keyboard_mode_gestures || [];
  eyeModeGestures = msg.eye_mode_gestures || [];
  minSamplesPerLabel = msg.min_samples_per_label;
  buildCalibrationRows();

  // Restore real progress from the server rather than assuming 0 - the
  // calibration store outlives any one connection, so a reconnect
  // (or a dev server --reload) must not make finished work look wiped.
  latestCounts = msg.counts || {};
  if (msg.counts) {
    for (const [label, count] of Object.entries(msg.counts)) {
      updateRowProgress(label, count);
    }
  }
  if (msg.ready) {
    trainStatus.textContent = "Classifier already trained from saved samples.";
    trainStatus.classList.add("is-ok");
  }
  updateDashboardStatus(msg.ready, msg.counts);
  updateGatingUI(msg.ready);
}

// ---------------------------------------------------------------- dashboard

function updateDashboardStatus(ready, counts) {
  const total = allLabels.length;
  const readyCount = counts
    ? Object.values(counts).filter((c) => c >= minSamplesPerLabel).length
    : 0;
  dashCalibratedCount.textContent = total ? `${readyCount} / ${total}` : "—";
  dashTrainedStatus.textContent = ready ? "trained" : "not trained yet";
  dashTrainedStatus.classList.toggle("is-ok", !!ready);
}

goToCalibrationBtn.addEventListener("click", () => {
  dashboardView.hidden = true;
  calibrationView.hidden = false;
});

function buildCalibrationRows() {
  calRows.innerHTML = "";
  rowEls = {};

  // Grouped by mode rather than one flat list - which gesture belongs
  // to which mode isn't obvious from the name alone (e.g. "confirm" vs
  // "left_click" look like they could both be clicks), and the two
  // groups are never active at the same time in real use (see "Modes"
  // in the README), so calibrating them as visibly separate sets
  // reinforces that instead of presenting 9 gestures as one big
  // undifferentiated pile.
  const groups = [
    { title: "Neutral", hint: "Your resting/relaxed face - recorded as its own label so the classifier has a baseline for \"no gesture\".", labels: [neutralLabel] },
    { title: "Keyboard mode", hint: "Used while the on-screen keyboard is active: scanning the grid and confirming keys.", labels: keyboardModeGestures },
    { title: "Eye / mouse mode", hint: "Used once real cursor control is active: clicking and switching back to the keyboard.", labels: eyeModeGestures },
  ];

  for (const group of groups) {
    if (!group.labels.length) continue;

    const header = document.createElement("div");
    header.className = "cal-group-header";
    const title = document.createElement("h3");
    title.textContent = group.title;
    const hint = document.createElement("p");
    hint.textContent = group.hint;
    header.append(title, hint);
    calRows.appendChild(header);

    for (const label of group.labels) {
      const root = document.createElement("div");
      root.className = "cal-row";
      root.dataset.label = label;

      const name = document.createElement("span");
      name.className = "cal-row__name";
      // Display-only: "switch_to_keyboard" etc. read better with spaces,
      // and - more importantly - underscores aren't a CSS line-break
      // opportunity, so a long one-word label like that would overflow
      // its fixed-width column and visually overlap the bar next to it.
      // The raw label (with underscores) still goes out over the wire.
      name.textContent = label.replace(/_/g, " ");

      const badge = document.createElement("span");
      badge.className = "cal-row__badge";
      badge.textContent = "✓";
      badge.title = "Enough samples recorded";
      name.appendChild(badge);

      const bar = document.createElement("div");
      bar.className = "cal-row__bar";
      const fill = document.createElement("div");
      fill.className = "cal-row__fill";
      bar.appendChild(fill);

      const count = document.createElement("span");
      count.className = "cal-row__count";
      count.textContent = `0 / ${minSamplesPerLabel}`;

      const recordBtn = document.createElement("button");
      recordBtn.className = "btn cal-row__record";
      recordBtn.textContent = "Record";
      recordBtn.addEventListener("click", () => toggleRecording(label));

      const clearBtn = document.createElement("button");
      clearBtn.className = "btn btn--ghost cal-row__clear";
      clearBtn.textContent = "Clear";
      clearBtn.addEventListener("click", () => {
        send({ type: "reset_label_samples", label });
      });

      root.append(name, bar, count, recordBtn, clearBtn);
      calRows.appendChild(root);
      rowEls[label] = { root, fill, count, recordBtn };
    }
  }
}

function toggleRecording(label) {
  if (activeCaptureLabel === label) {
    // stop recording this label
    send({ type: "set_label", label: null });
  } else {
    // switching to (or starting) recording this label
    send({ type: "set_label", label });
  }
}

function handleLabelSet(msg) {
  // reflect which row (if any) is now actively recording
  if (activeCaptureLabel && rowEls[activeCaptureLabel]) {
    rowEls[activeCaptureLabel].root.classList.remove("is-recording");
    rowEls[activeCaptureLabel].recordBtn.classList.remove("is-active");
    rowEls[activeCaptureLabel].recordBtn.textContent = "Record";
  }

  activeCaptureLabel = msg.label;

  if (activeCaptureLabel && rowEls[activeCaptureLabel]) {
    rowEls[activeCaptureLabel].root.classList.add("is-recording");
    rowEls[activeCaptureLabel].recordBtn.classList.add("is-active");
    rowEls[activeCaptureLabel].recordBtn.textContent = "Recording… (click to stop)";
  }

  updateRowProgress(msg.label, msg.count || 0);
}

function updateRowProgress(label, count) {
  latestCounts[label] = count;
  const row = rowEls[label];
  if (!row) return;
  const pct = Math.min(100, (count / minSamplesPerLabel) * 100);
  row.fill.style.width = `${pct}%`;
  row.count.textContent = `${count} / ${minSamplesPerLabel}`;
  row.root.classList.toggle("is-ready", count >= minSamplesPerLabel);
  updateGatingUI();
}

// Keeps Train/Confirm from being clicked before they can actually
// succeed, instead of letting you find out via an error message (Train)
// or, worse, silently stranding you in keyboard mode with an untrained
// classifier where literally no gesture can ever fire (Confirm).
function updateGatingUI(ready) {
  const allCalibrated = allLabels.length > 0 &&
    allLabels.every((label) => (latestCounts[label] || 0) >= minSamplesPerLabel);

  trainBtn.disabled = !allCalibrated;
  trainBtn.title = allCalibrated
    ? ""
    : `Record at least ${minSamplesPerLabel} samples for every gesture above first.`;

  const isReady = ready !== undefined ? ready : confirmSetupBtn.dataset.ready === "true";
  confirmSetupBtn.dataset.ready = isReady ? "true" : "false";
  confirmSetupBtn.disabled = !isReady;
  confirmSetupBtn.title = isReady
    ? "Confirm calibration and switch to keyboard mode"
    : "Train the classifier below first - otherwise no gesture will do anything in keyboard mode.";
}

function handleTrainResult(msg) {
  trainStatus.textContent = msg.message;
  trainStatus.classList.toggle("is-ok", msg.status === "ok");
  trainStatus.classList.toggle("is-error", msg.status === "error");

  if (msg.counts) {
    for (const [label, count] of Object.entries(msg.counts)) {
      updateRowProgress(label, count);
    }
  }
  updateDashboardStatus(msg.status === "ok", msg.counts);
  updateGatingUI(msg.status === "ok");
}

function handleGestureEvent(msg) {
  eventLogEmpty?.remove();

  const row = document.createElement("div");
  row.className = "event-row";

  const label = document.createElement("span");
  label.className = "event-row__label";
  label.textContent = msg.label;

  const meta = document.createElement("span");
  meta.className = "event-row__meta";
  const time = new Date().toLocaleTimeString([], { hour12: false });
  meta.textContent = `${Math.round(msg.confidence * 100)}% · ${time}`;

  row.append(label, meta);
  eventLog.prepend(row);

  // keep the log short
  while (eventLog.children.length > 8) {
    eventLog.removeChild(eventLog.lastChild);
  }

  // Gesture routing (show/hide the keyboard overlay, forward moves to
  // it) is handled in Python (desktop_app.py), since the keyboard is a
  // separate window from this one. In plain-browser dev mode (no
  // pywebview), there's nothing to forward to.
  if (window.pywebview?.api?.on_gesture) {
    window.pywebview.api.on_gesture(msg.label);
  }
}

function handleResetOk() {
  activeCaptureLabel = null;
  trainStatus.textContent = "";
  trainStatus.classList.remove("is-ok", "is-error");
  mPrediction.textContent = "—";
  eventLog.innerHTML = "";
  const empty = document.createElement("p");
  empty.className = "note";
  empty.id = "eventLogEmpty";
  empty.textContent = "Calibrate and train below to start seeing live gesture events here.";
  eventLog.appendChild(empty);
  latestCounts = {};
  buildCalibrationRows();
  updateDashboardStatus(false, {});
  updateGatingUI(false);
}

trainBtn.addEventListener("click", () => send({ type: "train" }));
resetAllBtn.addEventListener("click", () => send({ type: "reset_all" }));

// ---------------------------------------------------------------- render

function renderLoop() {
  const w = overlay.width;
  const h = overlay.height;

  ctx.clearRect(0, 0, w, h);

  // Mirror everything (video + landmarks together) for a natural selfie
  // view. The frame sent to the backend is NOT mirrored, so landmark
  // coordinates line up correctly once we apply the same flip here.
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);

  ctx.drawImage(video, 0, 0, w, h);

  if (lastLandmarks) {
    drawLandmarks(lastLandmarks, w, h);
  } else {
    drawScanLine(w, h);
  }

  ctx.restore();

  requestAnimationFrame(renderLoop);
}

function drawLandmarks(points, w, h) {
  // Build a lookup of "this point index -> region name" from the key
  // regions the server sent us, so we can color important points
  // distinctly and dim the rest of the mesh.
  const indexToRegion = {};
  if (keyRegions) {
    for (const [region, indices] of Object.entries(keyRegions)) {
      for (const i of indices) indexToRegion[i] = region;
    }
  }

  // dim full mesh first
  ctx.fillStyle = "rgba(201, 209, 217, 0.28)";
  for (let i = 0; i < points.length; i++) {
    if (indexToRegion[i]) continue;
    const p = points[i];
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 1, 0, Math.PI * 2);
    ctx.fill();
  }

  // highlighted key regions on top, slightly larger
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;

    const region = indexToRegion[i];
    if (!region) continue;
    ctx.fillStyle = REGION_COLORS[region] || "#4CC9F0";
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  drawCornerBrackets(minX * w, minY * h, maxX * w, maxY * h);
}

function drawCornerBrackets(x0, y0, x1, y1) {
  const pad = 14;
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  const len = Math.min(24, (x1 - x0) / 3);

  ctx.strokeStyle = "#4CC9F0";
  ctx.lineWidth = 2;

  const corners = [
    [[x0, y0 + len], [x0, y0], [x0 + len, y0]],
    [[x1 - len, y0], [x1, y0], [x1, y0 + len]],
    [[x0, y1 - len], [x0, y1], [x0 + len, y1]],
    [[x1 - len, y1], [x1, y1], [x1, y1 - len]],
  ];
  for (const path of corners) {
    ctx.beginPath();
    ctx.moveTo(path[0][0], path[0][1]);
    ctx.lineTo(path[1][0], path[1][1]);
    ctx.lineTo(path[2][0], path[2][1]);
    ctx.stroke();
  }
}

function drawScanLine(w, h) {
  // Gentle vertical sweep while no face is detected, signals "searching"
  // rather than "broken".
  scanPhase = (scanPhase + 0.01) % 1;
  const y = scanPhase * h;
  const gradient = ctx.createLinearGradient(0, y - 40, 0, y + 40);
  gradient.addColorStop(0, "rgba(76, 201, 240, 0)");
  gradient.addColorStop(0.5, "rgba(76, 201, 240, 0.25)");
  gradient.addColorStop(1, "rgba(76, 201, 240, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, y - 40, w, 80);
}

// ---------------------------------------------------------------- confirm / minimize
//
// The on-screen keyboard itself lives in a separate window now (see
// keyboard.html/keyboard.js) so it can float over other apps. This page
// is just the calibration/launch UI: confirming here hands off to
// Python (desktop_app.py) to hide this window and show the small
// left-edge toggle tab instead.

const confirmSetupBtn = document.getElementById("confirmSetupBtn");
confirmSetupBtn.addEventListener("click", () => {
  if (window.pywebview?.api?.confirm_calibration) {
    window.pywebview.api.confirm_calibration();
  } else {
    // Plain-browser dev mode: nothing to minimize to, just no-op.
    console.log("confirm_calibration: no pywebview API available (dev mode)");
  }
});

// ---------------------------------------------------------------- boot

startCamera();
connectWS();

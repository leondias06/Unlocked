// Facial Gesture Keyboard - CV pipeline frontend.
//
// Responsibilities:
//   1. Get webcam access and show a live preview.
//   2. Stream downscaled JPEG frames to the backend over a WebSocket.
//   3. Draw the returned landmarks on top of the video as visual proof
//      the tracking pipeline is working.
//
// No gesture logic lives here yet - this is purely the tracking step.

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
const SEND_INTERVAL_MS = 100; // ~10 fps to the server

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
  ws.send(JSON.stringify({ frame: dataUrl }));
}

function handleMessage(msg) {
  // latency: pair this response with the oldest outstanding send timestamp
  const sentAt = sendTimestamps.shift();
  if (sentAt !== undefined) {
    const rtt = Math.round(performance.now() - sentAt);
    mLatency.textContent = `${rtt} ms`;
    mLatency.classList.toggle("is-alert", rtt > 250);
  }

  // server fps (message rate)
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

// ---------------------------------------------------------------- boot

startCamera();
connectWS();

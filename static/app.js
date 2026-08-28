const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const fps = document.getElementById("fps");
const count = document.getElementById("count");
const zoneState = document.getElementById("zone-state");
const runtimeStatus = document.getElementById("runtime-status");
const alertBox = document.getElementById("alert");

let points = [];
let detections = [];
let frameSize = { width: 640, height: 480 };
let dragIndex = -1;
let lastAlertSeq = 0;
let alertHideTimer = null;
let zoneDirty = false;
let audioContext = null;
let audioUnlocked = false;
let filterLayers = {};

function scaleX() {
  return canvas.width / frameSize.width;
}

function scaleY() {
  return canvas.height / frameSize.height;
}

function toCanvasPoint(point) {
  return { x: point.x * scaleX(), y: point.y * scaleY() };
}

function toFramePoint(point) {
  return { x: point.x / scaleX(), y: point.y / scaleY() };
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  draw();
}

function drawZone() {
  if (points.length !== 4) return;

  const scaled = points.map(toCanvasPoint);
  ctx.beginPath();
  ctx.moveTo(scaled[0].x, scaled[0].y);
  scaled.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
  ctx.closePath();
  ctx.fillStyle = "rgba(255, 192, 67, 0.18)";
  ctx.strokeStyle = "#ffc043";
  ctx.lineWidth = 3;
  ctx.fill();
  ctx.stroke();

  scaled.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 9, 0, Math.PI * 2);
    ctx.fillStyle = "#101828";
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(index + 1), point.x, point.y);
  });
}

function drawDetections() {
  detections.forEach((det) => {
    const [x1, y1, x2, y2] = det.bbox;
    const x = x1 * scaleX();
    const y = y1 * scaleY();
    const w = (x2 - x1) * scaleX();
    const h = (y2 - y1) * scaleY();
    ctx.strokeStyle = "#18c37e";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "rgba(24, 195, 126, 0.9)";
    ctx.fillRect(x, Math.max(0, y - 24), 116, 24);
    ctx.fillStyle = "#042015";
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`person ${Math.round(det.confidence * 100)}%`, x + 8, Math.max(12, y - 12));
  });
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawZone();
  drawDetections();
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function nearestPointIndex(pos) {
  let best = -1;
  let bestDistance = 20;
  points.map(toCanvasPoint).forEach((point, index) => {
    const distance = Math.hypot(point.x - pos.x, point.y - pos.y);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

canvas.addEventListener("pointerdown", (event) => {
  unlockAudio();
  dragIndex = nearestPointIndex(pointerPosition(event));
  if (dragIndex >= 0) canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (dragIndex < 0) return;
  const pos = pointerPosition(event);
  const clamped = {
    x: Math.max(0, Math.min(canvas.width, pos.x)),
    y: Math.max(0, Math.min(canvas.height, pos.y)),
  };
  points[dragIndex] = toFramePoint(clamped);
  zoneDirty = true;
  draw();
});

canvas.addEventListener("pointerup", () => {
  dragIndex = -1;
});

canvas.addEventListener("pointercancel", () => {
  dragIndex = -1;
});

async function saveZone() {
  const response = await fetch("/api/zone", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
  const data = await response.json();
  if (data.points) points = data.points;
  zoneDirty = false;
  draw();
}

async function resetZone() {
  const response = await fetch("/api/zone/reset", { method: "POST" });
  const data = await response.json();
  points = data.points;
  zoneDirty = false;
  draw();
}

function readFilterLayerControls() {
  const next = {};
  document.querySelectorAll("[data-filter-layer]").forEach((input) => {
    next[input.dataset.filterLayer] = input.checked;
  });
  return next;
}

function syncFilterLayerControls(layers) {
  filterLayers = layers || filterLayers;
  document.querySelectorAll("[data-filter-layer]").forEach((input) => {
    const value = filterLayers[input.dataset.filterLayer];
    if (typeof value === "boolean") input.checked = value;
  });
}

async function saveFilterLayers() {
  filterLayers = readFilterLayerControls();
  const response = await fetch("/api/filter-layers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filter_layers: filterLayers }),
  });
  const data = await response.json();
  syncFilterLayerControls(data.filter_layers);
}

async function unlockAudio() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return false;
  if (!audioContext) audioContext = new AudioContext();
  if (audioContext.state === "suspended") {
    try {
      await audioContext.resume();
    } catch (error) {
      return false;
    }
  }
  audioUnlocked = audioContext.state === "running";
  return audioUnlocked;
}

async function beep() {
  if (!(await unlockAudio())) return;

  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.frequency.value = 880;
  gain.gain.setValueAtTime(0.001, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.2, audioContext.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.25);
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.28);
}

function pulseAlert() {
  alertBox.classList.add("visible");
  beep();
}

function showTestAlert() {
  pulseAlert();
  if (alertHideTimer) window.clearTimeout(alertHideTimer);
  alertHideTimer = window.setTimeout(() => alertBox.classList.remove("visible"), 1200);
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    frameSize = data.frame_size || frameSize;
    if (!zoneDirty && dragIndex < 0) points = data.zone || points;
    syncFilterLayerControls(data.filter_layers);
    detections = data.detections || [];
    fps.textContent = `${data.fps ?? "0.0"} / ${data.capture_fps ?? "0.0"}`;
    count.textContent = `${detections.length} (${data.raw_detection_count ?? 0}/${data.filtered_detection_count ?? 0})`;

    const hasError = data.camera_error || data.detector_error || data.hid?.error;
    runtimeStatus.textContent = hasError || `${data.mode || "CAMERA"} live`;

    zoneState.textContent = data.collision ? "Active" : "Clear";
    zoneState.className = data.collision ? "state active" : "state clear";

    if (data.alert_seq > lastAlertSeq) {
      lastAlertSeq = data.alert_seq;
      pulseAlert();
    }
    if (data.collision) {
      if (alertHideTimer) {
        window.clearTimeout(alertHideTimer);
        alertHideTimer = null;
      }
      alertBox.classList.add("visible");
    } else {
      alertBox.classList.remove("visible");
    }
    draw();
  } catch (error) {
    runtimeStatus.textContent = "Flask status unavailable";
  }
}

document.getElementById("save-zone").addEventListener("click", saveZone);
document.getElementById("reset-zone").addEventListener("click", resetZone);
document.getElementById("test-alert").addEventListener("click", showTestAlert);
document.querySelectorAll("[data-filter-layer]").forEach((input) => {
  input.addEventListener("change", saveFilterLayers);
});
document.addEventListener("pointerdown", unlockAudio, { once: true });
document.addEventListener("keydown", unlockAudio, { once: true });

video.addEventListener("load", resizeCanvas);
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
pollStatus();
window.setInterval(pollStatus, 300);

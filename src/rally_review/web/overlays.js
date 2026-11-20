"use strict";

// MediaPipe Pose landmarks in the same order as src/pose_estimator.py.
const poseEdges = [
  [0,1], [1,2], [2,3], [3,7], [0,4], [4,5], [5,6], [6,8], [9,10],
  [11,12], [11,13], [13,15], [15,17], [15,19], [15,21], [17,19],
  [12,14], [14,16], [16,18], [16,20], [16,22], [18,20],
  [11,23], [12,24], [23,24], [23,25], [24,26], [25,27], [26,28],
  [27,29], [28,30], [29,31], [30,32], [27,31], [28,32]
];
const hiddenPoint = -32768;
const playerCanvas = el("player-overlay");
const playerContext = playerCanvas.getContext("2d");
const boxesToggle = el("boxes-toggle");
const posesToggle = el("poses-toggle");

let overlays = null;
let presentedFrame = null;
let callbackId = null;

function overlayStatus(message = "") {
  const node = el("overlay-status");
  node.textContent = message;
  node.hidden = !message;
}

function configureOverlays(query, chunkSize) {
  overlays?.controller.abort();
  if (callbackId != null) video.cancelVideoFrameCallback?.(callbackId);
  callbackId = null;
  overlays = query && chunkSize ? {
    query, chunkSize,
    controller: new AbortController(),
    chunks: new Map(),
    pending: new Set(),
  } : null;
  boxesToggle.disabled = posesToggle.disabled = !overlays;
  overlayStatus(query && !overlays ? "No saved player observations for this run." : "");
  presentedFrame = overlays && video.readyState >= 2 ? frameAt(video.currentTime) : null;
  watchVideoFrames();
  drawPlayerOverlay();
}

function watchVideoFrames() {
  if (callbackId != null || !overlays || !video.requestVideoFrameCallback) return;
  callbackId = video.requestVideoFrameCallback((_, metadata) => {
    callbackId = null;
    if (!overlays || !data) return;
    presentedFrame = frameAt(metadata.mediaTime + .000005);
    drawPlayerOverlay();
    watchVideoFrames();
  });
}

async function requestOverlayChunk(start) {
  const session = overlays;
  if (!session || start >= data.frameCount || session.chunks.has(start) || session.pending.has(start)) return;
  session.pending.add(start);
  try {
    const query = new URLSearchParams(session.query);
    query.set("start", start);
    const response = await fetch("/api/overlays?" + query, { signal: session.controller.signal });
    if (!response.ok) throw new Error("Could not load the saved player observations.");
    const chunk = await response.json();
    if (overlays !== session) return;
    session.chunks.set(start, chunk);
    while (session.chunks.size > 6) session.chunks.delete(session.chunks.keys().next().value);
    drawPlayerOverlay();
  } catch (error) {
    if (overlays !== session || error.name === "AbortError") return;
    session.chunks.set(start, null);
    overlayStatus(error.message);
  } finally {
    session.pending.delete(start);
  }
}

function drawPlayerOverlay() {
  const bounds = playerCanvas.getBoundingClientRect();
  const ctx = playerContext;
  const ratio = Math.min(window.devicePixelRatio || 1, 3);
  const width = Math.round(bounds.width * ratio);
  const height = Math.round(bounds.height * ratio);
  if (playerCanvas.width !== width) playerCanvas.width = width;
  if (playerCanvas.height !== height) playerCanvas.height = height;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  if (!overlays || !data || presentedFrame == null || !width || !height ||
      !video.videoWidth || !video.videoHeight || (!boxesToggle.checked && !posesToggle.checked)) return;

  const first = Math.floor(presentedFrame / overlays.chunkSize) * overlays.chunkSize;
  const chunk = overlays.chunks.get(first);
  if (chunk === undefined) requestOverlayChunk(first);
  if (!video.paused && presentedFrame - first >= overlays.chunkSize / 2) requestOverlayChunk(first + overlays.chunkSize);
  if (!chunk) return;
  const imageSize = chunk.imageSize;
  if (imageSize && (imageSize[0] !== video.videoWidth || imageSize[1] !== video.videoHeight)) {
    overlayStatus("Saved player observations have different image dimensions from this video.");
    return;
  }
  overlayStatus();
  const roles = chunk.frames[presentedFrame - first];
  if (!roles) return;

  const scale = Math.min(bounds.width / video.videoWidth, bounds.height / video.videoHeight);
  const displayedWidth = video.videoWidth * scale;
  const displayedHeight = video.videoHeight * scale;
  const left = (bounds.width - displayedWidth) / 2;
  const top = (bounds.height - displayedHeight) / 2;
  const x = value => left + value * scale;
  const y = value => top + value * scale;

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, displayedWidth, displayedHeight);
  ctx.clip();
  roles.forEach((item, role) => {
    if (!item) return;
    const [box, points] = item;
    const color = playerColors[role];
    if (boxesToggle.checked && box) {
      const [a,b,c,d,track] = box;
      ctx.lineJoin = "round";
      if (!points) ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#111418";
      ctx.lineWidth = 3.5;
      ctx.strokeRect(x(a), y(b), (c - a) * scale, (d - b) * scale);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.strokeRect(x(a), y(b), (c - a) * scale, (d - b) * scale);
      ctx.setLineDash([]);

      const label = playerNames[role] + (track == null ? "" : " · " + track);
      ctx.font = "600 11px system-ui";
      const labelWidth = ctx.measureText(label).width + 9;
      const labelX = clamp(x(a), left, Math.max(left, left + displayedWidth - labelWidth));
      const labelY = Math.max(top, y(b) - 17);
      ctx.fillStyle = color;
      ctx.fillRect(labelX, labelY, labelWidth, 16);
      ctx.fillStyle = "#111418";
      ctx.fillText(label, labelX + 4, labelY + 11.5);
    }
    if (!posesToggle.checked || !points) return;
    const visible = joint => points[joint * 2] !== hiddenPoint;
    ctx.beginPath();
    for (const [a,b] of poseEdges) {
      if (!visible(a) || !visible(b)) continue;
      ctx.moveTo(x(points[a * 2]), y(points[a * 2 + 1]));
      ctx.lineTo(x(points[b * 2]), y(points[b * 2 + 1]));
    }
    ctx.lineCap = "round";
    ctx.strokeStyle = "#111418";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.45;
    ctx.stroke();
    ctx.fillStyle = color;
    for (let i = 0; i < points.length / 2; i++) {
      if (!visible(i)) continue;
      ctx.beginPath();
      ctx.arc(x(points[i * 2]), y(points[i * 2 + 1]), 1.55, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  ctx.restore();
}

function setOverlayFrameFromVideo() {
  if (!overlays || !data || video.readyState < 2) return;
  presentedFrame = frameAt(video.currentTime + .000005);
  drawPlayerOverlay();
}

video.addEventListener("loadeddata", setOverlayFrameFromVideo);
video.addEventListener("seeking", () => { presentedFrame = null; drawPlayerOverlay(); });
video.addEventListener("seeked", setOverlayFrameFromVideo);
video.addEventListener("emptied", () => { presentedFrame = null; drawPlayerOverlay(); });
if (!video.requestVideoFrameCallback) video.addEventListener("timeupdate", setOverlayFrameFromVideo);
boxesToggle.onchange = posesToggle.onchange = drawPlayerOverlay;
new ResizeObserver(drawPlayerOverlay).observe(el("video-stage"));

"use strict";

const el = id => document.getElementById(id);
const video = el("video");
const canvas = el("timeline");
const ctx = canvas.getContext("2d");
const backdrop = document.createElement("canvas");
const colors = ["#68d6cc", "#f4ae6f", "#acb6ff", "#f597ad"];
const playerColors = ["#58caff", "#ee94fa"];
const playerNames = ["P1 near", "P2 far"];
const neutral = "#c6d0db";
const muted = "#aab7c4";
const lineColor = "#35404b";
const cursorColor = "#ffe0ad";

let data = null;
let events = [];
let failureCases = {};
let frame = 0;
let focus = 0;
let bounds = { low: 0, high: 0, left: 76, right: 0, width: 0, height: 0 };
let layers = [];
let signature = "";
let animation = 0;
let durationWarning = "";
let requestedFrame = null;

const probability = p => typeof p === "number" && p >= 0 && p <= 1;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const time = seconds => {
  const cs = Math.round(Math.max(seconds, 0) * 100);
  return String(Math.floor(cs / 6000)).padStart(2, "0") + ":" +
    String(Math.floor(cs / 100) % 60).padStart(2, "0") + "." + String(cs % 100).padStart(2, "0");
};
const spanLength = n => ((secondsAt(n[1] + 1) - secondsAt(n[0]))).toFixed(2) + " s";
const secondsAt = f => data.timestamps?.[f] ??
  (data.timestamps ? data.timestamps.at(-1) + (f - data.frameCount + 1) / data.fps : f / data.fps);
const frameAt = seconds => {
  if (!data.timestamps) return clamp(Math.floor(seconds * data.fps), 0, data.frameCount - 1);
  let a = 0, b = data.frameCount;
  while (a < b) {
    const middle = Math.floor((a + b) / 2);
    if (data.timestamps[middle] <= seconds) a = middle + 1;
    else b = middle;
  }
  return Math.max(0, a - 1);
};

function validate(input) {
  const { sourceId, frameCount, fps, timestamps, models, labels, coverage, players } = input ?? {};
  const need = (isValid, message) => { if (!isValid) throw new Error(message); };
  need(input?.schema === "baddievision-rally-diagnostics-v1", "Choose a BaddieVision rally diagnostics JSON.");
  need(typeof sourceId === "string" && sourceId.trim() && Number.isInteger(frameCount) && frameCount > 0 &&
       Number.isFinite(fps) && fps > 0, "The source, frame count, or frame rate is missing.");
  const series = (value, predicate) => Array.isArray(value) && value.length === frameCount && value.every(predicate);
  need(timestamps == null || (series(timestamps, (t, i) => Number.isFinite(t) && t >= 0 &&
       (i === 0 ? t === 0 : t > timestamps[i - 1]))), "Frame timestamps must be increasing and start at zero.");
  need(Array.isArray(models) && models.length >= 1 && models.length <= colors.length, "Provide one to four prediction runs.");
  for (const model of models) {
    need(typeof model?.name === "string" && model.name.trim() && (model.threshold == null || probability(model.threshold)) &&
         series(model.inplay, x => x === null || probability(x)) &&
         typeof model.decoded === "string" && model.decoded.length === frameCount && /^[01?]+$/.test(model.decoded),
         "Each run needs a name and source-aligned InPlay and decoded values.");
    for (const key of ["start", "end"]) {
      need(model[key] == null || series(model[key], x => x === null || probability(x)),
           "Boundary scores must be source-aligned probabilities.");
    }
  }
  const intervals = value => Array.isArray(value) && value.every((item, i) =>
    Array.isArray(item) && item.length === 2 && item.every(Number.isInteger) && item[0] >= 0 &&
    item[0] <= item[1] && item[1] < frameCount && (!i || value[i - 1][1] < item[0]));
  need(labels == null || intervals(labels), "Labeled rallies must be sorted, disjoint frame intervals.");
  need(coverage == null || (intervals(coverage) && labels != null &&
       labels.every(([a,b]) => coverage.some(([lo,hi]) => lo <= a && hi >= b))),
       "Reviewed coverage must contain every labeled rally.");
  need(players == null || (Array.isArray(players) && players.length === 2 && players.every(s =>
       typeof s === "string" && s.length === frameCount && /^[012?]+$/.test(s))), "Player tracks must have two source-aligned role series.");
  return { sourceId, frameCount, fps, timestamps, models, labels, coverage, players };
}

function spans(series, value) {
  const result = [];
  let start = null;
  for (let i = 0; i <= series.length; i++) {
    if (series[i] === value && start === null) start = i;
    if (series[i] !== value && start !== null) {
      result.push([start, i - 1]);
      start = null;
    }
  }
  return result;
}

function buildEvents() {
  const mode = el("mode").value;
  events = failureCases[mode];
  const dropdown = el("event");
  const placeholder = events.length ? "Choose an event" : "No events in this category";
  dropdown.replaceChildren(new Option(placeholder, ""), ...events.map(({ span, models }, i) =>
    new Option(time(secondsAt(span[0])) + " · " + spanLength(span) +
      (models.length ? " · " + models.map(m => data.models[m].name).join(" + ") : ""), i)));
  dropdown.disabled = events.length === 0;
  const unsupported = !data.coverage?.length && ["false", "earlyStart", "lateEnd", "merged"].includes(mode);
  el("mode-help").textContent = data.labels == null ? "Load metrics with rally labels to review failures." :
    unsupported ? "This category needs reviewed footage outside the labeled rallies." :
    failureModes[mode].help + (boundaryModes.includes(mode) ? " Matches need one prediction, one label, and IoU ≥ 0.50." : "");
  showEventDetail();
}

function showEventDetail() {
  const node = el("event-detail");
  const selected = events[el("event").value];
  node.hidden = !selected;
  if (!selected) return;
  const mode = failureModes[el("mode").value];
  const stamps = ([a,b]) => time(secondsAt(a)) + "–" + time(secondsAt(b));
  const duration = selected.marks.reduce((sum,[a,b]) => sum + secondsAt(b+1) - secondsAt(a), 0);
  const details = [mode.name, selected.models.map(m => data.models[m].name).join(" + "),
    duration.toFixed(2) + " s highlighted"];
  if (selected.label && selected.prediction) {
    details.push("Labeled " + stamps(selected.label), "Decoded " + stamps(selected.prediction),
      "IoU " + selected.iou.toFixed(2));
  }
  node.textContent = details.filter(Boolean).join(" · ");
  node.style.setProperty("--event-color", mode.color);
}

function setup(input) {
  data = input;
  requestedFrame = null;
  if (video.readyState >= 1) frame = frameAt(video.currentTime);
  else frame = 0;
  focus = frame;
  signature = "";
  layers = [];
  if (data.labels) layers.push({ label: "Labeled", color: neutral, spans: data.labels });
  data.models.forEach((model, i) => layers.push({
    label: "Run " + (i + 1), color: colors[i], spans: spans(model.decoded, "1"), model: i
  }));
  if (data.players) data.players.forEach((role,i) => layers.push({
    label: playerNames[i], color: playerColors[i], spans: spans(role,"2"),
    mutedSpans: spans(role,"1"), player: true
  }));
  const legend = el("legend");
  const legendItems = data.models.map(({name, threshold}, i) => [
    "Run " + (i + 1) + " · " + name + (threshold == null ? "" : " · threshold " + threshold.toFixed(2)), colors[i]
  ]);
  if (data.players) playerNames.forEach((name, i) => legendItems.push([name, playerColors[i]]));
  legend.replaceChildren(...legendItems.map(([name, color]) => {
    const key = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.setProperty("--swatch", color);
    key.append(swatch, document.createTextNode(name));
    return key;
  }));
  el("source").textContent = data.sourceId + " · " + data.frameCount.toLocaleString() + " frames · " + data.fps.toFixed(2) + " fps";
  el("summary").textContent = data.players ? "Solid player bars show a valid pose. Dim bars show an assigned box." : "";
  el("metrics").hidden = false;
  el("metrics-empty").hidden = true;
  el("heads").disabled = !data.models.some(model => model.start || model.end);
  if (el("heads").disabled) el("heads").checked = false;
  el("follow").disabled = false;
  el("overview").max = data.frameCount - 1;
  canvas.setAttribute("aria-valuemax", data.frameCount - 1);
  failureCases = collectFailures(data, layers.filter(layer => layer.model !== undefined));
  const mode = el("mode");
  const previous = mode.value;
  mode.replaceChildren(...Object.entries(failureModes).map(([key, definition]) =>
    new Option(definition.name + " (" + failureCases[key].length + ")", key)));
  mode.disabled = data.labels == null;
  mode.value = data.labels == null ? "label" : previous || (data.coverage?.length ? "false" : "label");
  buildEvents();
  render();
}

function plot() {
  const w = Math.round(canvas.getBoundingClientRect().width);
  if (!w) return;
  const mode = el("mode").value;
  const category = failureModes[mode];
  const failureLane = mode !== "label";
  const h = 126 + (layers.length + Number(failureLane)) * 17;
  const scale = Math.min(window.devicePixelRatio || 1, 3);
  const left = 74, right = w - 9, top = 10, bottom = 96;
  const length = el("window").value === "all" ? data.frameCount : Math.min(data.frameCount, Math.round(Number(el("window").value) * data.fps));
  const low = clamp(focus - Math.floor(length / 2), 0, data.frameCount - length);
  const high = low + length - 1;
  const start = secondsAt(low), stop = secondsAt(high + 1);
  const x = f => left + (secondsAt(f) - start) * (right - left) / (stop - start);
  const y = value => bottom - value * (bottom - top);
  bounds = { low, high, left, right, width: w, height: h, x };
  const selected = el("event").value;
  const nextSignature = [low, high, w, h, scale, el("heads").checked, selected, el("mode").value].join("/");
  if (signature === nextSignature) return;
  signature = nextSignature;
  canvas.width = backdrop.width = Math.round(w * scale);
  canvas.height = backdrop.height = Math.round(h * scale);
  canvas.style.height = h + "px";
  const plotter = backdrop.getContext("2d");
  plotter.scale(scale, scale);
  const rule = (x1, y1, x2, y2, color) => {
    plotter.beginPath();
    plotter.strokeStyle = color;
    plotter.moveTo(x1, y1);
    plotter.lineTo(x2, y2);
    plotter.stroke();
  };
  const text = (value, px, py, align = "right", color = muted) => {
    plotter.fillStyle = color;
    plotter.font = "12px system-ui";
    plotter.textAlign = align;
    plotter.fillText(value, px, py);
  };
  const bar = ([a,b], py, height, color, opacity = 1) => {
    if (b < low || a > high) return;
    plotter.globalAlpha = opacity;
    plotter.fillStyle = color;
    const begin = x(Math.max(a, low));
    const end = x(Math.min(b, high) + 1);
    plotter.fillRect(begin, py, Math.max(1, end - begin), height);
    plotter.globalAlpha = 1;
  };
  if (selected && events[Number(selected)]) {
    for (const mark of events[Number(selected)].marks) bar(mark, top, h - top - 24, category.color, .18);
  }
  for (const p of [0, .5, 1]) {
    rule(left, y(p), right, y(p), lineColor);
    text(p.toFixed(1), left - 8, y(p) + 4);
  }
  for (const [i, model] of data.models.entries()) {
    const color = colors[i];
    plotter.globalAlpha = .52;
    plotter.setLineDash([2, 4]);
    if (model.threshold != null) rule(left, y(model.threshold), right, y(model.threshold), color);
    plotter.globalAlpha = 1;
    for (const [key, dash] of [["inplay", []], ["start", [5, 3]], ["end", [1, 3]]]) {
      const values = model[key];
      if (!values || (key !== "inplay" && !el("heads").checked)) continue;
      plotter.setLineDash(dash);
      plotter.lineWidth = key === "inplay" ? 1.55 : 1;
      plotter.globalAlpha = key === "inplay" ? 1 : .7;
      plotter.beginPath();
      let continuous = false;
      for (let f = low; f <= high; f++) {
        const v = values[f];
        if (v === null) { continuous = false; continue; }
        if (continuous) plotter.lineTo(x(f), y(v));
        else plotter.moveTo(x(f), y(v));
        continuous = true;
      }
      plotter.strokeStyle = color;
      plotter.stroke();
    }
    plotter.globalAlpha = 1;
    plotter.lineWidth = 1;
    plotter.setLineDash([]);
  }
  if (failureLane) {
    const py = 112;
    text("Failures", left - 8, py + 4);
    rule(left, py, right, py, lineColor);
    for (const [i, event] of events.entries()) {
      for (const mark of event.marks) bar(mark, py - 4, 8, category.color, selected === String(i) ? 1 : .65);
    }
  }
  layers.forEach((lane, i) => {
    const py = 112 + 17 * (i + Number(failureLane));
    text(lane.label, left - 8, py + 4, "right", lane.player ? lane.color : muted);
    rule(left, py, right, py, lineColor);
    if (lane.label === "Labeled") data.coverage?.forEach(span => bar(span, py - 5, 10, neutral, .13));
    lane.mutedSpans?.forEach(span => bar(span, py - 4, 8, lane.color, .36));
    lane.spans.forEach(span => bar(span, py - 4, 8, lane.color));
  });
  const ticks = Math.max(1, Math.floor((right - left) / 97));
  for (let i = 0; i <= ticks; i++) {
    const progress = i / ticks;
    text(time(start + progress * (stop - start)), left + progress * (right - left), h - 6,
         i === 0 ? "left" : i === ticks ? "right" : "center");
  }
}

function render() {
  const seconds = video.readyState >= 1 ? video.currentTime : data ? secondsAt(frame) : 0;
  el("clock").textContent = time(seconds);
  if (!data) return;
  el("head-guide").hidden = !el("heads").checked;
  plot();
  const {width, height, low, high, x} = bounds;
  if (width) {
    ctx.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(backdrop, 0, 0, width, height);
    if (frame >= low && frame <= high) {
      ctx.strokeStyle = cursorColor;
      ctx.beginPath();
      ctx.moveTo(x(frame), 9);
      ctx.lineTo(x(frame), height - 21);
      ctx.stroke();
    }
  }
  const select = el("overview");
  select.value = frame;
  canvas.setAttribute("aria-valuenow", frame);
  canvas.setAttribute("aria-valuetext", time(secondsAt(frame)) + ", frame " + frame);
  const label = data.labels?.some(([a,b]) => a <= frame && frame <= b) ? "label: in play" :
    data.coverage?.some(([a,b]) => a <= frame && frame <= b) ? "label: out" : "label: unreviewed";
  const format = v => v == null ? "—" : v.toFixed(2);
  const readout = ["Frame " + frame.toLocaleString() + " · " + label];
  for (const [i, model] of data.models.entries()) {
    const decoded = { "1": "in", "0": "out", "?": "unscored" }[model.decoded[frame]];
    let value = "Run " + (i+1) + " · " + format(model.inplay[frame]) + " · decoded " + decoded;
    if (el("heads").checked) value += " · start " + format(model.start?.[frame]) + " · end " + format(model.end?.[frame]);
    readout.push(value);
  }
  if (data.players) readout.push(data.players.map((role,i) =>
    "P" + (i+1) + " " + {"?":"unscored","0":"missing","1":"assigned","2":"valid pose"}[role[frame]]).join(" · "));
  const readoutNode = el("readout");
  readoutNode.replaceChildren(...readout.map(value => {
    const span = document.createElement("span");
    span.textContent = value;
    return span;
  }));
}

function seek(target, recenter = false) {
  if (!data) return;
  frame = clamp(target, 0, data.frameCount - 1);
  if (recenter || el("follow").checked && (frame < bounds.low || frame > bounds.high)) focus = frame;
  if (video.readyState >= 1) {
    requestedFrame = frame;
    const middle = (secondsAt(frame) + secondsAt(frame+1)) / 2;
    video.currentTime = clamp(middle, 0, video.duration);
  }
  render();
}

function sync() {
  if (data && video.readyState >= 1) {
    frame = requestedFrame ?? frameAt(video.currentTime);
    if (el("follow").checked && (frame < bounds.low || frame > bounds.high)) focus = frame;
  }
  render();
}

function updateNotice(message = "") {
  const notice = el("notice");
  notice.textContent = message || durationWarning;
  notice.hidden = !notice.textContent;
}

function matchDuration() {
  durationWarning = "";
  if (data && Number.isFinite(video.duration)) {
    const expected = secondsAt(data.frameCount);
    if (Math.abs(video.duration - expected) > Math.max(0.25, 3 / data.fps)) {
      durationWarning = "Duration differs from the metrics (" + time(video.duration) + " video vs. " +
        time(expected) + " metrics). Use the source-local clip for accurate alignment.";
    }
  }
  updateNotice();
}

function clearReview() {
  data = null;
  requestedFrame = null;
  focus = frame = 0;
  el("metrics").hidden = true;
  el("metrics-empty").hidden = false;
  el("heads").disabled = el("follow").disabled = true;
  el("summary").textContent = "Choose a run to see its model scores and rally boundaries.";
}

function loadSourceVideo(url) {
  if (video.getAttribute("src") === url) return;
  video.pause();
  if (url) video.src = url;
  else {
    video.removeAttribute("src");
    video.load();
  }
  el("video-stage").hidden = !url;
  el("video-empty").hidden = Boolean(url);
  durationWarning = "";
  updateNotice();
}

video.addEventListener("loadedmetadata", () => {
  matchDuration();
  if (data && frame > 0) seek(frame);
  else sync();
});
video.addEventListener("error", () => {
  if (!video.src) return;
  updateNotice("This video could not be played. Check that this browser supports the source video codec.");
  el("video-stage").hidden = true;
  el("video-empty").hidden = false;
});
video.addEventListener("timeupdate", sync);
video.addEventListener("seeked", () => { requestedFrame = null; sync(); });
video.addEventListener("play", () => {
  cancelAnimationFrame(animation);
  let prior = -1;
  const tick = () => {
    if (video.paused || video.ended) return;
    const next = data ? frameAt(video.currentTime) : Math.floor(video.currentTime * 25);
    if (next !== prior) { sync(); prior = next; }
    animation = requestAnimationFrame(tick);
  };
  animation = requestAnimationFrame(tick);
});
video.addEventListener("pause", () => { cancelAnimationFrame(animation); sync(); });
el("window").onchange = () => { focus = frame; render(); };
el("heads").onchange = render;
el("follow").onchange = () => { if (el("follow").checked) focus = frame; render(); };
el("mode").onchange = () => { buildEvents(); signature = ""; render(); };
el("event").onchange = () => {
  showEventDetail();
  const event = events[el("event").value];
  if (event) {
    const target = event.span[0];
    seek(target, true);
  } else render();
};
el("overview").oninput = event => seek(Number(event.target.value), true);
canvas.addEventListener("pointerdown", event => {
  if (event.button !== 0 || !data) return;
  const {left, right, low, high} = bounds;
  const progress = clamp((event.offsetX - left) / (right - left), 0, 1);
  seek(frameAt(secondsAt(low) + progress * (secondsAt(high+1) - secondsAt(low))));
});
canvas.addEventListener("keydown", event => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  event.preventDefault();
  const direction = event.key === "ArrowLeft" ? -1 : 1;
  seek(frame + direction * (event.shiftKey ? Math.max(1, Math.round(data.fps)) : 1));
});
new ResizeObserver(() => { if (data) render(); }).observe(canvas);

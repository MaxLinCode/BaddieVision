"use strict";

const experimentSelect = el("experiment");
const recordingSelect = el("recording");
const primarySelect = el("primary-run");
const compareSelect = el("compare-run");
const status = el("load-status");
const refresh = el("refresh-experiments");

let catalog = [];
let metricsRequest;
let installedKey = "";

function options(select, values, value) {
  select.replaceChildren(...values.map(([name, id, disabled]) => {
    const option = new Option(name, id);
    option.disabled = Boolean(disabled);
    return option;
  }));
  if (values.some(([, id, disabled]) => id === value && !disabled)) select.value = value;
  else select.selectedIndex = values.findIndex(([, , disabled]) => !disabled);
  select.disabled = !values.some(([, , disabled]) => !disabled);
}

function selectedRecording() {
  return catalog.find(recording => recording.id === recordingSelect.value);
}

function updateRuns(recording, primary, compare) {
  const runs = recording?.runs ?? [];
  const variants = runs.map(run => [run.name + (run.playable ? "" : " (video missing)"), run.id, !run.playable]);
  options(primarySelect, variants, primary);
  if (compare === undefined && variants.length === 2) {
    compare = variants.find(([,id,disabled]) => id !== primarySelect.value && !disabled)?.[1];
  }
  options(compareSelect, [["No comparison", ""], ...variants.filter(([, id]) => id !== primarySelect.value)], compare);
  compareSelect.disabled = !variants.some(([,id,disabled]) => id !== primarySelect.value && !disabled);
}

function updateRecordings(recording, primary, compare) {
  const choices = catalog.filter(item => item.experiment === experimentSelect.value);
  options(recordingSelect, choices.map(item => [item.source, item.id]), recording);
  updateRuns(selectedRecording(), primary, compare);
}

async function readJSON(url, signal) {
  const response = await fetch(url, { signal });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "The local viewer could not read this artifact.");
  return payload;
}

async function refreshCatalog(initial = false) {
  refresh.disabled = true;
  let before = { experiment: experimentSelect.value, recording: recordingSelect.value,
    primary: primarySelect.value, compare: compareSelect.value };
  if (initial) {
    try { before = JSON.parse(localStorage.getItem("rally-review-selection")) ?? before; }
    catch { /* The first load does not require saved preferences. */ }
  }
  status.textContent = "Scanning local experiment outputs…";
  try {
    const result = await readJSON("/api/catalog");
    catalog = result.recordings;
    const names = [...new Set(catalog.map(item => item.experiment))];
    options(experimentSelect, names.map(name => [name.replace(/^experiments\//, ""), name]), before.experiment);
    updateRecordings(before.recording, before.primary, initial && !before.recording ? undefined : before.compare);
    if (!catalog.length) {
      status.textContent = "No saved rally prediction files found in the BaddieVision workspace.";
      el("source").textContent = "No local rally experiments";
      return;
    }
    await loadRecording(!initial);
  } catch (error) {
    status.textContent = error.message;
  } finally {
    refresh.disabled = false;
  }
}

async function loadRecording(force = false) {
  const item = selectedRecording();
  const primary = primarySelect.value;
  const compare = compareSelect.value;
  const selection = new URLSearchParams();
  if (item) selection.set("recording", item.id);
  if (primary) selection.append("run", primary);
  const playerQuery = primary ? selection.toString() : null;
  const videoURL = playerQuery ? "/api/video?" + playerQuery : null;
  if (compare) selection.append("run", compare);
  const requestKey = selection.toString();
  if (!force && installedKey === requestKey && data) return;

  metricsRequest?.abort();
  const request = new AbortController();
  metricsRequest = request;
  try {
    localStorage.setItem("rally-review-selection", JSON.stringify({
      experiment: experimentSelect.value, recording: recordingSelect.value, primary, compare
    }));
  } catch { /* Loading an experiment does not require local storage. */ }

  if (video.getAttribute("src") !== videoURL || !primary) {
    configureOverlays(null);
    clearReview();
    loadSourceVideo(videoURL);
  }
  if (!item || !primary) {
    status.textContent = item ? "Source video unavailable. Register it in config/sources.local.json." : "";
    el("source").textContent = item?.source ?? "Local experiment viewer";
    el("experiment-picker").open = true;
    return;
  }
  status.textContent = "Reading " + item.source + " from disk…";
  try {
    const payload = await readJSON("/api/metrics?" + selection, request.signal);
    if (metricsRequest !== request) return;
    setup(validate(payload.data));
    configureOverlays(playerQuery, payload.data.overlayChunkSize);
    matchDuration();
    installedKey = requestKey;
    status.textContent = payload.warnings.join(" ");
  } catch (error) {
    if (metricsRequest !== request || error.name === "AbortError") return;
    configureOverlays(null);
    clearReview();
    el("experiment-picker").open = true;
    status.textContent = error.message;
  }
}

refresh.onclick = () => refreshCatalog();
experimentSelect.onchange = () => { updateRecordings(); loadRecording(); };
recordingSelect.onchange = () => { updateRuns(selectedRecording()); loadRecording(); };
primarySelect.onchange = () => { updateRuns(selectedRecording(), primarySelect.value, compareSelect.value); loadRecording(); };
compareSelect.onchange = () => loadRecording();
refreshCatalog(true);

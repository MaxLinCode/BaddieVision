"use strict";

const failureModes = {
  false: { name: "False starts", color: "#f7b76d", help: "Decoded play entirely outside labeled rallies in reviewed footage." },
  earlyStart: { name: "Early starts", color: "#f7b76d", help: "Decoded play before a matched rally starts." },
  lateStart: { name: "Late starts", color: "#f69caf", help: "Labeled play before the model starts its matched rally." },
  earlyEnd: { name: "Early ends", color: "#f69caf", help: "Labeled play after the model ends its matched rally." },
  lateEnd: { name: "Late ends", color: "#f7b76d", help: "Decoded play after a matched rally ends." },
  missed: { name: "Missed rallies", color: "#f69caf", help: "Fully scored labeled rallies with no decoded play." },
  split: { name: "Split rallies", color: "#bcb5fb", help: "Missed gaps between decoded fragments of one labeled rally." },
  merged: { name: "Merged rallies", color: "#bcb5fb", help: "Decoded play bridging a reviewed gap between labeled rallies." },
  lowOverlap: { name: "Low overlap", color: "#f69caf", help: "Single overlapping prediction and label with interval IoU below 0.50. Highlights their disagreement." },
  label: { name: "Labeled rallies", color: "#c6d0db", help: "Reviewed rally intervals from the labels." }
};
const boundaryModes = ["earlyStart", "lateStart", "earlyEnd", "lateEnd"];

function intervalOverlaps(predictions, labels) {
  const byPrediction = predictions.map(() => []);
  const byLabel = labels.map(() => []);
  let p = 0, l = 0;
  while (p < predictions.length && l < labels.length) {
    const [a,b] = predictions[p], [c,d] = labels[l];
    if (a <= d && c <= b) {
      byPrediction[p].push(l);
      byLabel[l].push(p);
    }
    if (b <= d) p++;
    if (d <= b) l++;
  }
  return { byPrediction, byLabel };
}

function collectFailures(data, decodedLayers) {
  const labels = data.labels ?? [];
  const coverage = data.coverage ?? [];
  const results = Object.fromEntries(Object.keys(failureModes).map(mode => [mode, []]));
  results.label = labels.map(span => ({ span, marks: [span], models: [] }));
  if (data.labels == null) return results;

  const reviewed = ([a,b]) => coverage.some(([lo,hi]) => lo <= a && b <= hi);
  const event = (mode, marks, model, references = {}) => {
    if (!marks.length) return;
    const span = [marks[0][0], marks.at(-1)[1]];
    results[mode].push({ span, marks, models: [model], ...references });
  };
  for (const { spans: predictions, model } of decodedLayers) {
    const { byPrediction, byLabel } = intervalOverlaps(predictions, labels);
    const decoded = data.models[model].decoded;
    const scored = ([start,end]) => !decoded.slice(start, end + 1).includes("?");
    predictions.forEach((prediction, i) => {
      const touched = byPrediction[i];
      if (!touched.length) {
        if (reviewed(prediction)) event("false", [prediction], model);
        return;
      }
      if (touched.length > 1) {
        const gaps = touched.slice(1).map((l,j) => [labels[touched[j]][1] + 1, labels[l][0] - 1]);
        event("merged", gaps.filter(span => span[0] <= span[1] && reviewed(span)), model, { prediction });
        return;
      }
      const labelIndex = touched[0];
      if (byLabel[labelIndex].length !== 1) return;
      const label = labels[labelIndex];
      const [a,b] = prediction, [c,d] = label;
      const overlap = Math.min(b,d) - Math.max(a,c) + 1;
      const iou = overlap / (Math.max(b,d) - Math.min(a,c) + 1);
      const mismatches = [
        ["earlyStart", [a, c-1], reviewed],
        ["lateStart", [c, a-1], scored],
        ["earlyEnd", [b+1, d], scored],
        ["lateEnd", [d+1, b], reviewed]
      ].filter(([, span, known]) => span[0] <= span[1] && known(span));
      const references = { prediction, label, iou };
      if (iou >= .50) {
        for (const [mode,span] of mismatches) event(mode, [span], model, references);
      } else if (scored(label) && (a >= c && b <= d || reviewed(prediction))) {
        event("lowOverlap", mismatches.map(([,span]) => span), model, references);
      }
    });
    labels.forEach((label, i) => {
      const touched = byLabel[i];
      if (!touched.length && scored(label)) event("missed", [label], model, { label });
      if (touched.length > 1) {
        const gaps = touched.slice(1).map((p,j) => [predictions[touched[j]][1] + 1, predictions[p][0] - 1]);
        event("split", gaps.filter(span => span[0] <= span[1] && scored(span)), model, { label });
      }
    });
  }

  for (const cases of Object.values(results)) cases.sort((a,b) => a.span[0]-b.span[0] || a.span[1]-b.span[1]);
  const groupedFalse = [];
  for (const current of results.false) {
    const previous = groupedFalse.at(-1);
    if (previous && current.span[0] <= previous.span[1]) {
      previous.span[1] = Math.max(previous.span[1], current.span[1]);
      previous.marks = [previous.span];
      previous.models = [...new Set([...previous.models, ...current.models])].sort((a,b)=>a-b);
    } else {
      groupedFalse.push({ ...current, span: [...current.span] });
    }
  }
  results.false = groupedFalse;
  return results;
}

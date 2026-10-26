'use strict';

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropEmpty = document.getElementById('drop-empty');
const preview = document.getElementById('preview');
const fileMeta = document.getElementById('file-meta');
const analyzeBtn = document.getElementById('analyze-btn');
const errorBox = document.getElementById('error-box');

const resultEmpty = document.getElementById('result-empty');
const resultBody = document.getElementById('result-body');
const resultLoading = document.getElementById('result-loading');
const verdictBadge = document.getElementById('verdict-badge');
const confidenceValue = document.getElementById('confidence-value');
const meterFill = document.getElementById('meter-fill');
const probFakeEl = document.getElementById('prob-fake');
const resultFile = document.getElementById('result-file');
const resultModel = document.getElementById('result-model');

const statusChip = document.getElementById('status-chip');
const modelChip = document.getElementById('model-chip');
const resetBtn = document.getElementById('reset-btn');
const verdictNote = document.getElementById('verdict-note');
const probRealValue = document.getElementById('prob-real-value');
const probFakeValue = document.getElementById('prob-fake-value');
const probRealFill = document.getElementById('prob-real-fill');
const probFakeFill = document.getElementById('prob-fake-fill');

let selectedFile = null;
const MAX_BYTES = 15 * 1024 * 1024;

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
}
function clearError() {
  errorBox.classList.add('hidden');
  errorBox.textContent = '';
}

function setFile(file) {
  clearError();
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showError('That file is not an image. Please choose a PNG, JPG or WebP image.');
    return;
  }
  if (file.size > MAX_BYTES) {
    showError('That image is larger than 15 MB. Please choose a smaller file.');
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove('hidden');
  dropEmpty.classList.add('hidden');
  fileMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  fileMeta.classList.remove('hidden');
  analyzeBtn.disabled = false;
  resetBtn.classList.remove('hidden');
}

function resetAll() {
  selectedFile = null;
  fileInput.value = '';
  preview.src = '';
  preview.classList.add('hidden');
  dropEmpty.classList.remove('hidden');
  fileMeta.classList.add('hidden');
  analyzeBtn.disabled = true;
  resetBtn.classList.add('hidden');
  resultBody.classList.add('hidden');
  resultLoading.classList.add('hidden');
  resultEmpty.classList.remove('hidden');
  clearError();
}
resetBtn.addEventListener('click', resetAll);

document.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (item) setFile(item.getAsFile());
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

['dragenter', 'dragover'].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('dragging'); })
);
['dragleave', 'drop'].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('dragging'); })
);
dropZone.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));

function pct(x) { return (x * 100).toFixed(1) + '%'; }

function renderResult(data) {
  const isFake = data.prediction === 'Fake';
  verdictBadge.textContent = isFake ? 'Fake' : 'Real';
  verdictBadge.className = 'verdict-badge ' + (isFake ? 'is-fake' : 'is-real');
  confidenceValue.textContent = pct(data.confidence);
  meterFill.style.width = pct(data.confidence);
  meterFill.className = 'meter-fill ' + (isFake ? 'is-fake' : 'is-real');
  probFakeEl.textContent = pct(data.probability_fake);
  const pFake = data.probability_fake;
  const pReal = 1 - pFake;
  probRealValue.textContent = pct(pReal);
  probFakeValue.textContent = pct(pFake);
  probRealFill.style.width = pct(pReal);
  probFakeFill.style.width = pct(pFake);
  const conf = data.confidence;
  if (conf >= 0.9) {
    verdictNote.textContent = isFake
      ? 'Strong signal: the CNN is highly confident this image carries manipulation artifacts.'
      : 'Strong signal: the CNN is highly confident this image looks like a genuine photograph.';
    verdictNote.className = 'verdict-note';
  } else if (conf >= 0.7) {
    verdictNote.textContent = isFake
      ? 'Moderate signal: the model leans fake, but treat this as a flag to inspect further — not proof.'
      : 'Moderate signal: the model leans real, with some uncertainty remaining.';
    verdictNote.className = 'verdict-note';
  } else {
    verdictNote.textContent = 'Borderline: this score is close to 50/50, so the honest verdict is inconclusive. Try a clearer, tighter face crop.';
    verdictNote.className = 'verdict-note is-warn';
  }
  verdictNote.classList.remove('hidden');
  resultFile.textContent = data.filename || '—';
  resultModel.textContent = data.model ? data.model.name : '—';
  resultLoading.classList.add('hidden');
  resultEmpty.classList.add('hidden');
  resultBody.classList.remove('hidden');
}

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  clearError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  resultBody.classList.add('hidden');
  resultEmpty.classList.add('hidden');
  resultLoading.classList.remove('hidden');
  try {
    const form = new FormData();
    form.append('file', selectedFile);
    const resp = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `Request failed (${resp.status})`);
    }
    renderResult(data);
  } catch (err) {
    resultLoading.classList.add('hidden');
    resultEmpty.classList.remove('hidden');
    showError(err.message || 'Something went wrong while analyzing the image.');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze image';
  }
});

// ---------- header status ----------
async function loadHealth() {
  try {
    const resp = await fetch('/api/health');
    const data = await resp.json();
    if (data.model_name) modelChip.textContent = data.model_name + ' · CNN';
    if (data.status === 'ok') {
      statusChip.innerHTML = '<span class="dot dot-ok"></span>model ready';
    } else {
      statusChip.innerHTML = '<span class="dot dot-bad"></span>model unavailable';
    }
  } catch {
    statusChip.innerHTML = '<span class="dot dot-bad"></span>server unreachable';
  }
}

// ---------- measured metrics ----------
async function loadMetrics() {
  const grid = document.getElementById('metrics-grid');
  const meta = document.getElementById('metrics-meta');
  const missing = document.getElementById('metrics-missing');
  try {
    const resp = await fetch('/api/metrics');
    if (!resp.ok) return;
    const m = await resp.json();
    document.getElementById('m-accuracy').textContent = pct(m.accuracy);
    document.getElementById('m-precision').textContent = pct(m.precision);
    document.getElementById('m-recall').textContent = pct(m.recall);
    document.getElementById('m-f1').textContent = pct(m.f1_score);
    document.getElementById('m-auc').textContent = m.roc_auc.toFixed(4);
    meta.textContent = `${m.dataset_name} · ${m.n_samples.toLocaleString()} images (${m.n_real.toLocaleString()} real / ${m.n_fake.toLocaleString()} fake) · threshold 0.5`;
    grid.classList.remove('hidden');
    meta.classList.remove('hidden');
    missing.classList.add('hidden');
    // Show the evaluation plots only if both plot files actually exist.
    const plots = document.getElementById('metrics-plots');
    const cmImg = document.getElementById('plot-cm');
    const rocImg = document.getElementById('plot-roc');
    let loaded = 0;
    const onPlot = (ok) => {
      if (ok && ++loaded === 2) plots.classList.remove('hidden');
    };
    [cmImg, rocImg].forEach((img) => {
      if (img.complete && img.naturalWidth > 0) onPlot(true);
      else {
        img.addEventListener('load', () => onPlot(true), { once: true });
        img.addEventListener('error', () => onPlot(false), { once: true });
      }
    });
  } catch { /* leave the "not available" note visible */ }
}

loadHealth();
loadMetrics();

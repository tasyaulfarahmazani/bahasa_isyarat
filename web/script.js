// ============================================================
// CONFIG
// ============================================================
const API_BASE   = "http://127.0.0.1:8000";
const SESSION_ID = "session_" + Math.random().toString(36).slice(2, 8);

const STATIC_INTERVAL_MS   = 1000;
const DYNAMIC_INTERVAL_MS  = 500;
const CONFIDENCE_THRESHOLD = 40;

// ============================================================
// STATE
// ============================================================
let currentMode   = "static";
let cameraRunning = false;
let intervalId    = null;
let stream        = null;

let sentence      = [];
let totalDetect   = 0;
let accepted      = 0;
let lastPred      = "";

let frameCount    = 0;
let fpsTimer      = null;

// ============================================================
// DOM
// ============================================================
const video          = document.getElementById("video");
const canvas         = document.getElementById("canvas");
const predValue      = document.getElementById("predValue");
const confFill       = document.getElementById("confFill");
const confPct        = document.getElementById("confPct");
const bufferWrap     = document.getElementById("bufferWrap");
const bufferFill     = document.getElementById("bufferFill");
const bufferPct      = document.getElementById("bufferPct");
const sentenceTokens = document.getElementById("sentenceTokens");
const btnStart       = document.getElementById("btnStart");
const apiDot         = document.getElementById("apiDot");
const apiStatus      = document.getElementById("apiStatus");
const scanLine       = document.getElementById("scanLine");
const idleOverlay    = document.getElementById("idleOverlay");
const statTotal      = document.getElementById("statTotal");
const statAccepted   = document.getElementById("statAccepted");
const statFps        = document.getElementById("statFps");
const modeLabel      = document.getElementById("modeLabel");

// ============================================================
// API HEALTH CHECK
// ============================================================
async function checkAPI() {
  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      apiDot.className      = "dot online";
      apiStatus.textContent = "API online";
    } else throw new Error();
  } catch {
    apiDot.className      = "dot offline";
    apiStatus.textContent = "API offline";
  }
}

checkAPI();
setInterval(checkAPI, 8000);

// ============================================================
// MODE SWITCH
// ============================================================
function switchMode(mode) {
  currentMode = mode;

  document.querySelectorAll(".tab").forEach(t => {
    t.classList.toggle("active", t.dataset.mode === mode);
  });

  bufferWrap.style.display = mode === "dynamic" ? "block" : "none";
  predValue.classList.toggle("word", mode === "dynamic");
  modeLabel.textContent = mode === "dynamic" ? "Kata (LSTM)" : "Huruf & Angka";

  setPrediction("—", 0);
  resetBufferBar();

  if (cameraRunning) {
    clearInterval(intervalId);
    const ms = mode === "dynamic" ? DYNAMIC_INTERVAL_MS : STATIC_INTERVAL_MS;
    intervalId = setInterval(sendFrame, ms);
  }
}

// ============================================================
// CAMERA
// ============================================================
async function toggleCamera() {
  if (cameraRunning) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
    });

    video.srcObject = stream;
    video.classList.add("active");
    idleOverlay.classList.add("hidden");
    scanLine.classList.add("active");

    btnStart.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
        <rect x="6" y="5" width="4" height="14" rx="1"/>
        <rect x="14" y="5" width="4" height="14" rx="1"/>
      </svg>
      Stop Kamera
    `;
    btnStart.classList.add("running");

    cameraRunning = true;

    const ms = currentMode === "dynamic" ? DYNAMIC_INTERVAL_MS : STATIC_INTERVAL_MS;
    intervalId = setInterval(sendFrame, ms);

    fpsTimer = setInterval(() => {
      statFps.textContent = frameCount;
      frameCount = 0;
    }, 1000);

    showToast("Kamera aktif");

  } catch (err) {
    console.error(err);
    showToast("Gagal membuka kamera");
  }
}

function stopCamera() {
  clearInterval(intervalId);
  clearInterval(fpsTimer);
  intervalId = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  video.srcObject = null;
  video.classList.remove("active");
  idleOverlay.classList.remove("hidden");
  scanLine.classList.remove("active");

  btnStart.innerHTML = `
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7z"/>
    </svg>
    Mulai Kamera
  `;
  btnStart.classList.remove("running");

  cameraRunning = false;
  statFps.textContent = "0";

  setPrediction("—", 0);
  showToast("Kamera dimatikan");
}

// ============================================================
// SEND FRAME
// Tampilan video di-mirror oleh CSS (scaleX(-1)).
// Canvas TIDAK di-flip — kirim frame normal ke API.
// API juga TIDAK flip — konsisten dengan cara training.
// ============================================================
async function sendFrame() {
  if (!cameraRunning || video.readyState < 2) return;

  const ctx = canvas.getContext("2d");
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  // Gambar langsung tanpa flip — biarkan CSS yang mirror tampilan
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!blob) return;
    frameCount++;
    totalDetect++;
    statTotal.textContent = totalDetect;

    const endpoint = currentMode === "dynamic" ? "/predict-dynamic" : "/predict";
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const url = currentMode === "dynamic"
        ? `${API_BASE}${endpoint}?session_id=${SESSION_ID}`
        : `${API_BASE}${endpoint}`;

      const res  = await fetch(url, { method: "POST", body: formData });
      const data = await res.json();

      if (currentMode === "dynamic") {
        handleDynamicResult(data);
      } else {
        handleStaticResult(data);
      }
    } catch (err) {
      console.warn("Fetch error:", err);
    }
  }, "image/jpeg", 0.85);
}

// ============================================================
// RESULT HANDLERS
// ============================================================
function handleStaticResult(data) {
  const pred = data.prediction || "—";
  const conf = data.confidence || 0;

  setPrediction(pred, conf);

  if (pred !== "Tidak ada tangan" && conf >= CONFIDENCE_THRESHOLD) {
    accepted++;
    statAccepted.textContent = accepted;
    addToSentence(pred);
  }
}

function handleDynamicResult(data) {
  const bufLen = data.buffer_len || 0;
  const pct    = Math.round((bufLen / 30) * 100);

  bufferFill.style.width = pct + "%";
  bufferPct.textContent  = `${bufLen} / 30`;

  if (data.prediction) {
    const conf = data.confidence || 0;
    setPrediction(data.prediction, conf);

    if (conf >= CONFIDENCE_THRESHOLD) {
      accepted++;
      statAccepted.textContent = accepted;
      addToSentence(data.prediction);
    }
  }
}

// ============================================================
// UI HELPERS
// ============================================================
function setPrediction(text, conf) {
  predValue.textContent = text;
  predValue.classList.toggle("word", text.length > 3);

  confFill.style.width = conf + "%";
  confPct.textContent  = conf.toFixed(1) + "%";

  confFill.style.background =
    conf >= 75 ? "#166534" :
    conf >= 45 ? "#1c1917" :
    "#9a3412";
}

function resetBufferBar() {
  bufferFill.style.width = "0%";
  bufferPct.textContent  = "0 / 30";
}

function addToSentence(word) {
  if (word === lastPred) return;
  lastPred = word;

  sentence.push(word);
  if (sentence.length > 20) sentence = sentence.slice(-20);
  renderSentence();
}

function renderSentence() {
  if (sentence.length === 0) {
    sentenceTokens.innerHTML =
      '<span class="token-placeholder">Mulai deteksi untuk melihat hasil...</span>';
    return;
  }

  sentenceTokens.innerHTML = sentence
    .map(w => `<span class="token">${w}</span>`)
    .join("");

  sentenceTokens.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function resetSentence() {
  sentence    = [];
  lastPred    = "";
  totalDetect = 0;
  accepted    = 0;

  statTotal.textContent    = "0";
  statAccepted.textContent = "0";

  renderSentence();
  setPrediction("—", 0);
  resetBufferBar();

  fetch(`${API_BASE}/session/${SESSION_ID}`, { method: "DELETE" }).catch(() => {});
  showToast("Reset");
}

async function copySentence() {
  const text = sentence.join(" ");
  if (!text) { showToast("Belum ada kalimat"); return; }
  try {
    await navigator.clipboard.writeText(text);
    showToast("Disalin ke clipboard");
  } catch {
    showToast("Gagal menyalin");
  }
}

function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2200);
}
/**
 * app/static/js/ui.js — UI Orchestrator, MediaPipe Pipeline, and Real-Time Dashboard.
 */

import { LandmarkNormalizer } from "./landmarks.js";
import { ASLWebSocketClient } from "./socket.js";

// DOM Elements
const videoEl = document.getElementById("webcam-video");
const ipcamImgEl = document.getElementById("ipcam-feed");
const canvasEl = document.getElementById("overlay-canvas");
const canvasCtx = canvasEl ? canvasEl.getContext("2d") : null;

const btnToggleCamera = document.getElementById("btn-toggle-camera");
const btnToggleIpcam = document.getElementById("btn-toggle-ipcam");
const inputIpcamUrl = document.getElementById("input-ipcam-url");
const ipcamStatusEl = document.getElementById("ipcam-status");
const btnMirror = document.getElementById("btn-mirror");
const btnForceCut = document.getElementById("btn-force-cut");
const btnClearTranscript = document.getElementById("btn-clear-transcript");
const btnSimulate = document.getElementById("btn-simulate");
const selectSampleClip = document.getElementById("select-sample-clip");

const connDot = document.getElementById("conn-dot");
const connText = document.getElementById("conn-text");
const hudStatus = document.getElementById("hud-status");
const hudFps = document.getElementById("hud-fps");
const bannerText = document.getElementById("banner-text");
const bannerMeta = document.getElementById("banner-meta");
const transcriptList = document.getElementById("transcript-list");

const valModelType = document.getElementById("val-model-type");
const valDevice = document.getElementById("val-device");
const valLatency = document.getElementById("val-latency");
const valSentences = document.getElementById("val-sentences");

// Offscreen canvas for clean snapshot capture of MJPEG IP Webcam frames
const offscreenCanvas = document.createElement("canvas");
const offscreenCtx = offscreenCanvas.getContext("2d");

// State
let normalizer = new LandmarkNormalizer();
let wsClient = null;
let camera = null;
let isCameraRunning = false;
let isIpcamRunning = false;
let isProcessingFrame = false;
let isMirrored = true;
let sentenceCount = 0;
let simulationInterval = null;

// Throttling to target 15 fps
const TARGET_FPS = 15;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;
let lastFrameTime = 0;
let fpsCounter = 0;
let fpsLastTime = performance.now();

// ---------------------------------------------------------------------------
// MediaPipe Holistic Setup
// ---------------------------------------------------------------------------

let holistic = null;

function initMediaPipe() {
  if (typeof window.Holistic === "undefined") {
    console.warn("MediaPipe Holistic script not loaded yet. Retrying in 500ms...");
    setTimeout(initMediaPipe, 500);
    return;
  }

  holistic = new window.Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });

  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  holistic.onResults(onHolisticResults);
  console.log("MediaPipe Holistic initialized successfully.");
}

function onHolisticResults(results) {
  const now = performance.now();

  // Draw landmarks on overlay canvas
  drawLandmarks(results);

  // Measure camera/detection FPS
  fpsCounter++;
  if (now - fpsLastTime >= 1000) {
    if (hudFps) hudFps.textContent = `${fpsCounter} FPS`;
    fpsCounter = 0;
    fpsLastTime = now;
  }

  // Throttle feature extraction & network streaming to ~15 fps
  if (now - lastFrameTime < FRAME_INTERVAL_MS) {
    return;
  }
  lastFrameTime = now;

  // Compute 1200-dim feature vector via LandmarkNormalizer
  const featureVector = normalizer.processFrame(results);

  // Stream binary vector across WebSocket
  if (wsClient && wsClient.isConnected) {
    wsClient.sendFrame(featureVector);
  }
}

// ---------------------------------------------------------------------------
// Canvas Drawing Utilities
// ---------------------------------------------------------------------------

function drawLandmarks(results) {
  if (!canvasCtx || !canvasEl) return;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  // If window.drawConnectors from MediaPipe Drawing Utils is available
  if (typeof window.drawConnectors !== "undefined" && typeof window.drawLandmarks !== "undefined") {
    // Upper body pose
    if (results.poseLandmarks) {
      window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, {
        color: "rgba(6, 182, 212, 0.6)",
        lineWidth: 3,
      });
      window.drawLandmarks(canvasCtx, results.poseLandmarks, {
        color: "#38bdf8",
        lineWidth: 1,
        radius: 3,
      });
    }

    // Left hand
    if (results.leftHandLandmarks) {
      window.drawConnectors(canvasCtx, results.leftHandLandmarks, window.HAND_CONNECTIONS, {
        color: "#10b981",
        lineWidth: 2,
      });
      window.drawLandmarks(canvasCtx, results.leftHandLandmarks, {
        color: "#6ee7b7",
        lineWidth: 1,
        radius: 2,
      });
    }

    // Right hand
    if (results.rightHandLandmarks) {
      window.drawConnectors(canvasCtx, results.rightHandLandmarks, window.HAND_CONNECTIONS, {
        color: "#f59e0b",
        lineWidth: 2,
      });
      window.drawLandmarks(canvasCtx, results.rightHandLandmarks, {
        color: "#fcd34d",
        lineWidth: 1,
        radius: 2,
      });
    }

    // Face mesh (subtle)
    if (results.faceLandmarks) {
      window.drawConnectors(canvasCtx, results.faceLandmarks, window.FACEMESH_LIPS, {
        color: "rgba(244, 63, 94, 0.4)",
        lineWidth: 1,
      });
    }
  }

  canvasCtx.restore();
}

// ---------------------------------------------------------------------------
// Camera & IP Webcam Control
// ---------------------------------------------------------------------------

function setIpcamStatus(message, state = "") {
  if (!ipcamStatusEl) return;
  ipcamStatusEl.textContent = message;
  ipcamStatusEl.className = "ipcam-status" + (state ? ` ${state}` : "");
}

async function toggleCamera() {
  if (isCameraRunning) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  if (!videoEl || !holistic) return;

  // Disconnect IP Webcam if active
  if (isIpcamRunning) {
    stopIpcam();
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 360, frameRate: { ideal: 30 } },
      audio: false,
    });

    videoEl.classList.remove("hidden");
    if (ipcamImgEl) ipcamImgEl.classList.add("hidden");

    videoEl.srcObject = stream;
    await videoEl.play();

    if (canvasEl) {
      canvasEl.width = videoEl.videoWidth || 640;
      canvasEl.height = videoEl.videoHeight || 360;
    }

    isCameraRunning = true;
    if (btnToggleCamera) {
      btnToggleCamera.textContent = "Stop Local Camera";
      btnToggleCamera.classList.add("btn-danger");
    }

    normalizer.reset();

    // Loop frames through Holistic
    async function frameLoop() {
      if (!isCameraRunning) return;
      if (videoEl.readyState >= 2 && !isProcessingFrame) {
        isProcessingFrame = true;
        try {
          await holistic.send({ image: videoEl });
        } finally {
          isProcessingFrame = false;
        }
      }
      requestAnimationFrame(frameLoop);
    }
    requestAnimationFrame(frameLoop);

    console.log("Local webcam started.");
  } catch (err) {
    console.error("Camera access failed:", err);
    alert(
      `Could not start local camera: ${err.message}. ` +
      `Use the IP Webcam option (172.15.45.93:8080) or Simulation Mode below!`
    );
  }
}

function stopCamera() {
  isCameraRunning = false;
  if (videoEl && videoEl.srcObject) {
    videoEl.srcObject.getTracks().forEach((t) => t.stop());
    videoEl.srcObject = null;
  }
  if (canvasCtx && canvasEl) {
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  }
  if (btnToggleCamera) {
    btnToggleCamera.textContent = "Start Local Camera";
    btnToggleCamera.classList.remove("btn-danger");
  }
  if (hudStatus) hudStatus.textContent = "Camera Off";
  normalizer.reset();
}

async function toggleIpcam() {
  if (isIpcamRunning) {
    stopIpcam();
  } else {
    await startIpcam();
  }
}

async function startIpcam() {
  if (!ipcamImgEl || !holistic) {
    setIpcamStatus("MediaPipe Holistic is still loading — please wait a second and retry.", "error");
    return;
  }

  const rawUrl = (inputIpcamUrl ? inputIpcamUrl.value : "http://172.15.45.93:8080/video").trim();
  if (!rawUrl) {
    setIpcamStatus("Please enter an IP Webcam address (e.g. 172.15.45.93:8080).", "error");
    return;
  }

  // Stop local camera if running
  if (isCameraRunning) {
    stopCamera();
  }

  if (btnToggleIpcam) {
    btnToggleIpcam.disabled = true;
    btnToggleIpcam.textContent = "Connecting...";
  }
  setIpcamStatus(`Testing connection to ${rawUrl}...`);

  try {
    const testRes = await fetch(`/api/ipcam/test?url=${encodeURIComponent(rawUrl)}`);
    const testData = await testRes.json();

    if (!testRes.ok || !testData.ok) {
      const errMsg = testData.error || testData.detail || "Unable to reach IP Webcam";
      setIpcamStatus(`Connection failed: ${errMsg}`, "error");
      if (btnToggleIpcam) {
        btnToggleIpcam.disabled = false;
        btnToggleIpcam.textContent = "Connect IP Webcam";
      }
      return;
    }

    const resolvedUrl = testData.resolved_url || rawUrl;
    if (inputIpcamUrl) {
      inputIpcamUrl.value = resolvedUrl;
    }

    // Show IP Webcam feed proxied via same-origin FastAPI server
    if (videoEl) videoEl.classList.add("hidden");
    ipcamImgEl.classList.remove("hidden");

    const proxyUrl = `/api/ipcam/stream?url=${encodeURIComponent(resolvedUrl)}&t=${Date.now()}`;
    ipcamImgEl.onerror = () => {
      if (isIpcamRunning) {
        setIpcamStatus(`IP Webcam stream interrupted (${resolvedUrl}). Check phone app & Wi-Fi.`, "error");
        stopIpcam(true);
      }
    };
    ipcamImgEl.src = proxyUrl;

    isIpcamRunning = true;
    normalizer.reset();
    if (wsClient) wsClient.sendReset();

    if (btnToggleIpcam) {
      btnToggleIpcam.disabled = false;
      btnToggleIpcam.textContent = "Disconnect IP Webcam";
      btnToggleIpcam.classList.remove("btn-cyan");
      btnToggleIpcam.classList.add("btn-danger");
    }
    setIpcamStatus(`Connected to ${resolvedUrl}`, "connected");
    if (hudStatus) hudStatus.textContent = "IP Webcam Active";

    // Frame extraction loop: draw current MJPEG frame to offscreen canvas and send to MediaPipe
    async function ipcamFrameLoop() {
      if (!isIpcamRunning) return;

      if (
        ipcamImgEl.complete &&
        ipcamImgEl.naturalWidth > 0 &&
        ipcamImgEl.naturalHeight > 0 &&
        !isProcessingFrame
      ) {
        const w = ipcamImgEl.naturalWidth;
        const h = ipcamImgEl.naturalHeight;

        if (offscreenCanvas.width !== w || offscreenCanvas.height !== h) {
          offscreenCanvas.width = w;
          offscreenCanvas.height = h;
        }
        if (canvasEl && (canvasEl.width !== w || canvasEl.height !== h)) {
          canvasEl.width = w;
          canvasEl.height = h;
        }

        isProcessingFrame = true;
        try {
          offscreenCtx.drawImage(ipcamImgEl, 0, 0, w, h);
          await holistic.send({ image: offscreenCanvas });
        } catch (err) {
          console.error("IP Webcam frame processing error:", err);
        } finally {
          isProcessingFrame = false;
        }
      }

      requestAnimationFrame(ipcamFrameLoop);
    }
    requestAnimationFrame(ipcamFrameLoop);

    console.log("IP Webcam connected:", resolvedUrl);
  } catch (err) {
    console.error("IP Webcam connection error:", err);
    setIpcamStatus(`Connection error: ${err.message}`, "error");
    if (btnToggleIpcam) {
      btnToggleIpcam.disabled = false;
      btnToggleIpcam.textContent = "Connect IP Webcam";
    }
  }
}

function stopIpcam(preserveStatus = false) {
  isIpcamRunning = false;
  if (ipcamImgEl) {
    ipcamImgEl.onerror = null;
    ipcamImgEl.src = "";
    ipcamImgEl.classList.add("hidden");
  }
  if (videoEl) {
    videoEl.classList.remove("hidden");
  }
  if (canvasCtx && canvasEl) {
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  }
  if (btnToggleIpcam) {
    btnToggleIpcam.disabled = false;
    btnToggleIpcam.textContent = "Connect IP Webcam";
    btnToggleIpcam.classList.remove("btn-danger");
    btnToggleIpcam.classList.add("btn-cyan");
  }
  if (!preserveStatus) {
    setIpcamStatus("IP Webcam disconnected.");
  }
  if (hudStatus) hudStatus.textContent = "Camera Off";
  if (hudFps) hudFps.textContent = "0 FPS";
  normalizer.reset();
  if (wsClient) wsClient.sendReset();
}

function toggleMirror() {
  isMirrored = !isMirrored;
  if (videoEl) videoEl.classList.toggle("unmirrored", !isMirrored);
  if (ipcamImgEl) ipcamImgEl.classList.toggle("unmirrored", !isMirrored);
  if (canvasEl) canvasEl.classList.toggle("unmirrored", !isMirrored);
}

// ---------------------------------------------------------------------------
// WebSocket Client & Handlers
// ---------------------------------------------------------------------------

function initWebSocket() {
  wsClient = new ASLWebSocketClient({
    binaryMode: true,
    onConnectionChange: (connected) => {
      if (connDot) {
        connDot.className = "status-dot " + (connected ? "connected" : "disconnected");
      }
      if (connText) {
        connText.textContent = connected ? "Connected" : "Disconnected";
      }
      if (hudStatus) {
        hudStatus.textContent = connected ? "Ready" : "Server Offline";
      }
    },
    onStatus: (data) => {
      if (hudStatus) {
        if (data.is_signing) {
          hudStatus.textContent = `Signing (${data.buffered_frames} frames)`;
          hudStatus.className = "hud-tag hud-signing";
        } else {
          hudStatus.textContent = data.pause_counter > 0 ? "Pause / Rest" : "Idle";
          hudStatus.className = "hud-tag";
        }
      }
    },
    onTranslating: (data) => {
      if (hudStatus) {
        hudStatus.textContent = "Translating...";
        hudStatus.className = "hud-tag";
      }
      if (connDot) {
        connDot.className = "status-dot translating";
      }
    },
    onTranslation: (data) => {
      handleTranslationResult(data);
    },
  });

  wsClient.connect();
}

function handleTranslationResult(data) {
  sentenceCount++;
  if (valSentences) valSentences.textContent = sentenceCount.toString();
  if (valLatency) valLatency.textContent = `${data.latency_ms} ms`;

  if (connDot) {
    connDot.className = "status-dot connected";
  }
  if (hudStatus) {
    hudStatus.textContent = "Translated";
  }

  const text = data.text || "(no speech detected)";

  // Update current banner
  if (bannerText) {
    bannerText.textContent = text;
    bannerText.classList.remove("empty");
  }
  if (bannerMeta) {
    bannerMeta.textContent = `${data.num_frames} frames | ${data.latency_ms} ms (${data.reason || "boundary"})`;
  }

  // Prepend to transcript history
  if (transcriptList) {
    const timeStr = new Date().toLocaleTimeString();
    const item = document.createElement("div");
    item.className = "transcript-item";
    item.innerHTML = `
      <div class="transcript-meta">
        <span>#${sentenceCount} • ${timeStr}</span>
        <div>
          <span class="badge badge-green">${data.latency_ms} ms</span>
          <span class="badge">${data.num_frames} frames</span>
        </div>
      </div>
      <div class="transcript-text">${escapeHtml(text)}</div>
    `;
    transcriptList.prepend(item);
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ---------------------------------------------------------------------------
// Simulation Mode (Simulate signing with How2Sign validation clips)
// ---------------------------------------------------------------------------

async function loadSampleClipsList() {
  if (!selectSampleClip) return;
  try {
    const res = await fetch("/api/sample-clips");
    const data = await res.json();
    selectSampleClip.innerHTML = "";

    const clips = data.val_clips || [];
    if (clips.length === 0) {
      selectSampleClip.innerHTML = "<option>No clips found</option>";
      return;
    }

    clips.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = `val/${name}`;
      opt.textContent = name;
      selectSampleClip.appendChild(opt);
    });
  } catch (e) {
    console.error("Could not fetch sample clips list:", e);
  }
}

async function runClipSimulation() {
  if (!selectSampleClip || !wsClient || !wsClient.isConnected) {
    alert("Please ensure the WebSocket server is connected.");
    return;
  }

  const selectedVal = selectSampleClip.value;
  if (!selectedVal) return;

  const [split, clipName] = selectedVal.split("/");
  if (btnSimulate) {
    btnSimulate.disabled = true;
    btnSimulate.textContent = "Simulating...";
  }

  try {
    const res = await fetch(`/api/clip/${split}/${clipName}`);
    if (!res.ok) throw new Error("Failed to load clip data");
    const clipData = await res.json();

    const frames = clipData.frames;
    console.log(`Simulating ${frames.length} frames for clip ${clipName}...`);

    let idx = 0;
    if (hudStatus) hudStatus.textContent = `Simulating: 0/${frames.length}`;

    simulationInterval = setInterval(() => {
      if (idx < frames.length) {
        const frameVec = new Float32Array(frames[idx]);
        wsClient.sendFrame(frameVec);
        idx++;
        if (hudStatus) hudStatus.textContent = `Simulating: ${idx}/${frames.length}`;
      } else {
        // Feed rest frames (15 frames) to trigger sentence boundary cut
        const restVec = new Float32Array(1200);
        for (let r = 0; r < 15; r++) {
          wsClient.sendFrame(restVec);
        }
        clearInterval(simulationInterval);
        simulationInterval = null;
        if (btnSimulate) {
          btnSimulate.disabled = false;
          btnSimulate.textContent = "Simulate Selected Clip";
        }
        console.log("Clip simulation stream finished.");
      }
    }, FRAME_INTERVAL_MS);
  } catch (err) {
    console.error("Simulation error:", err);
    alert(`Simulation failed: ${err.message}`);
    if (btnSimulate) {
      btnSimulate.disabled = false;
      btnSimulate.textContent = "Simulate Selected Clip";
    }
  }
}

// ---------------------------------------------------------------------------
// Server Status Polling
// ---------------------------------------------------------------------------

async function pollServerStatus() {
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      if (valModelType) valModelType.textContent = (data.model_type || "T5").toUpperCase();
      if (valDevice) valDevice.textContent = data.device || "CPU";
      if (inputIpcamUrl && data.default_ipcam_url && !inputIpcamUrl.dataset.userEdited) {
        inputIpcamUrl.value = data.default_ipcam_url;
      }
    }
  } catch (e) {
    // Ignore network glitch during initial load
  }
}

// ---------------------------------------------------------------------------
// Event Handlers Initialization
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initWebSocket();
  initMediaPipe();
  pollServerStatus();
  loadSampleClipsList();

  if (btnToggleCamera) btnToggleCamera.addEventListener("click", toggleCamera);
  if (btnToggleIpcam) btnToggleIpcam.addEventListener("click", toggleIpcam);
  if (inputIpcamUrl) {
    inputIpcamUrl.addEventListener("input", () => {
      inputIpcamUrl.dataset.userEdited = "true";
    });
    inputIpcamUrl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        toggleIpcam();
      }
    });
  }
  if (btnMirror) btnMirror.addEventListener("click", toggleMirror);
  if (btnSimulate) btnSimulate.addEventListener("click", runClipSimulation);

  if (btnForceCut) {
    btnForceCut.addEventListener("click", () => {
      if (wsClient) wsClient.sendForceCut();
    });
  }

  if (btnClearTranscript) {
    btnClearTranscript.addEventListener("click", () => {
      if (transcriptList) transcriptList.innerHTML = "";
      if (bannerText) {
        bannerText.textContent = "Transcript cleared. Start signing or simulate a clip.";
        bannerText.classList.add("empty");
      }
    });
  }
});

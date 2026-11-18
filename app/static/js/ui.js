/**
 * app/static/js/ui.js — UI Orchestrator, MediaPipe Pipeline, and Real-Time Dashboard.
 */

import { LandmarkNormalizer } from "./landmarks.js";
import { ASLWebSocketClient } from "./socket.js";

// DOM Elements
const videoEl = document.getElementById("webcam-video");
const ipcamCanvasEl = document.getElementById("ipcam-canvas");
const ipcamCtx = ipcamCanvasEl ? ipcamCanvasEl.getContext("2d") : null;
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
const hudHands = document.getElementById("hud-hands");
const hudFps = document.getElementById("hud-fps");
const bannerText = document.getElementById("banner-text");
const bannerMeta = document.getElementById("banner-meta");
const transcriptList = document.getElementById("transcript-list");

const valModelType = document.getElementById("val-model-type");
const valDevice = document.getElementById("val-device");
const valLatency = document.getElementById("val-latency");
const valSentences = document.getElementById("val-sentences");

// Offscreen canvas for clean snapshot capture of IP Webcam frames
const offscreenCanvas = document.createElement("canvas");
const offscreenCtx = offscreenCanvas.getContext("2d");

// State
let normalizer = new LandmarkNormalizer();
let wsClient = null;
let camera = null;
let isCameraRunning = false;
let isIpcamRunning = false;
let ipcamAbortController = null;
let isProcessingFrame = false;
let hasReceivedHolisticResult = false;
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
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });

  holistic.onResults(onHolisticResults);
  console.log("MediaPipe Holistic initialized successfully.");
}

function onHolisticResults(results) {
  const now = performance.now();
  hasReceivedHolisticResult = true;

  // Immediate local detection HUD feedback (no server round-trip delay)
  const hasPose = Boolean(results.poseLandmarks && results.poseLandmarks.length >= 13);
  const hasLh = Boolean(results.leftHandLandmarks && results.leftHandLandmarks.length === 21);
  const hasRh = Boolean(results.rightHandLandmarks && results.rightHandLandmarks.length === 21);

  if (hudHands) {
    hudHands.textContent = `Pose: ${hasPose ? "✓" : "–"} | L: ${hasLh ? "✓" : "–"} | R: ${hasRh ? "✓" : "–"}`;
    hudHands.className = (hasLh || hasRh) ? "hud-tag hud-detected" : "hud-tag";
  }

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
    if (ipcamCanvasEl) ipcamCanvasEl.classList.add("hidden");

    videoEl.srcObject = stream;
    await videoEl.play();

    if (canvasEl) {
      canvasEl.width = videoEl.videoWidth || 640;
      canvasEl.height = videoEl.videoHeight || 360;
    }

    isCameraRunning = true;
    hasReceivedHolisticResult = false;
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
  if (hudHands) {
    hudHands.textContent = "Pose: – | L: – | R: –";
    hudHands.className = "hud-tag";
  }
  normalizer.reset();
}

// ---------------------------------------------------------------------------
// IP Webcam MJPEG Stream Decoder (works identically in Firefox & Chrome)
// ---------------------------------------------------------------------------

function concatUint8(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function findBytes(buf, pattern, start = 0) {
  const limit = buf.length - pattern.length;
  for (let i = start; i <= limit; i++) {
    let match = true;
    for (let j = 0; j < pattern.length; j++) {
      if (buf[i + j] !== pattern[j]) {
        match = false;
        break;
      }
    }
    if (match) return i;
  }
  return -1;
}

function findLastJpegEoi(buf) {
  for (let i = buf.length - 2; i >= 0; i--) {
    if (buf[i] === 0xff && buf[i + 1] === 0xd9) {
      return i + 2;
    }
  }
  return -1;
}

async function renderAndProcessBitmap(bitmap) {
  const w = bitmap.width;
  const h = bitmap.height;

  if (ipcamCanvasEl && ipcamCtx) {
    if (ipcamCanvasEl.width !== w || ipcamCanvasEl.height !== h) {
      ipcamCanvasEl.width = w;
      ipcamCanvasEl.height = h;
    }
    ipcamCtx.drawImage(bitmap, 0, 0, w, h);
  }

  if (canvasEl && (canvasEl.width !== w || canvasEl.height !== h)) {
    canvasEl.width = w;
    canvasEl.height = h;
  }

  if (!isProcessingFrame && holistic) {
    if (offscreenCanvas.width !== w || offscreenCanvas.height !== h) {
      offscreenCanvas.width = w;
      offscreenCanvas.height = h;
    }
    offscreenCtx.drawImage(bitmap, 0, 0, w, h);

    if (!hasReceivedHolisticResult && hudStatus) {
      hudStatus.textContent = "Loading MediaPipe AI model...";
    }

    isProcessingFrame = true;
    holistic
      .send({ image: offscreenCanvas })
      .catch((err) => {
        console.error("MediaPipe Holistic frame error:", err);
        setIpcamStatus(`MediaPipe error: ${err.message || err}`, "error");
      })
      .finally(() => {
        isProcessingFrame = false;
      });
  }

  bitmap.close();
}

async function consumeMjpegStream(response, resolvedUrl) {
  const contentType = response.headers.get("content-type") || "";
  const boundaryMatch = contentType.match(/boundary=["']?([^;"'\s]+)["']?/i);
  const encoder = new TextEncoder();
  let boundaryBytes = null;
  if (boundaryMatch) {
    const rawBoundary = boundaryMatch[1].replace(/^--/, "");
    boundaryBytes = encoder.encode(`--${rawBoundary}`);
  }
  const soiMarker = new Uint8Array([0xff, 0xd8]);
  const eoiMarker = new Uint8Array([0xff, 0xd9]);

  const reader = response.body.getReader();
  let buffer = new Uint8Array(0);
  const MAX_BUFFER_BYTES = 4 * 1024 * 1024;

  try {
    while (isIpcamRunning) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value || value.length === 0) continue;

      buffer = concatUint8(buffer, value);
      if (buffer.length > MAX_BUFFER_BYTES) {
        // Drop stale backlog if consumer ever falls behind
        buffer = buffer.slice(buffer.length - 512 * 1024);
      }

      // Extract all complete JPEG frames currently in buffer; render only the latest to keep latency zero
      let latestJpeg = null;

      if (boundaryBytes) {
        while (true) {
          const b1 = findBytes(buffer, boundaryBytes, 0);
          if (b1 === -1) break;
          const b2 = findBytes(buffer, boundaryBytes, b1 + boundaryBytes.length);
          if (b2 === -1) {
            if (b1 > 0) buffer = buffer.slice(b1);
            break;
          }
          const part = buffer.slice(b1 + boundaryBytes.length, b2);
          buffer = buffer.slice(b2);

          const soi = findBytes(part, soiMarker, 0);
          const eoiEnd = findLastJpegEoi(part);
          if (soi !== -1 && eoiEnd > soi + 2) {
            latestJpeg = part.slice(soi, eoiEnd);
          }
        }
      } else {
        while (true) {
          const soi = findBytes(buffer, soiMarker, 0);
          if (soi === -1) {
            buffer = new Uint8Array(0);
            break;
          }
          const eoi = findBytes(buffer, eoiMarker, soi + 2);
          if (eoi === -1) {
            if (soi > 0) buffer = buffer.slice(soi);
            break;
          }
          latestJpeg = buffer.slice(soi, eoi + 2);
          buffer = buffer.slice(eoi + 2);
        }
      }

      if (latestJpeg) {
        try {
          const blob = new Blob([latestJpeg], { type: "image/jpeg" });
          const bitmap = await createImageBitmap(blob);
          await renderAndProcessBitmap(bitmap);
        } catch (decodeErr) {
          console.debug("Skipped partial JPEG frame:", decodeErr);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError" && isIpcamRunning) {
      console.error("IP Webcam stream error:", err);
      setIpcamStatus(`Stream interrupted (${resolvedUrl}): ${err.message}`, "error");
      stopIpcam(true);
    }
  } finally {
    try {
      await reader.cancel();
    } catch (_) {}
  }
}

async function toggleIpcam() {
  if (isIpcamRunning) {
    stopIpcam();
  } else {
    await startIpcam();
  }
}

async function startIpcam() {
  if (!ipcamCanvasEl || !holistic) {
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

    // Open streaming proxy connection via ReadableStream (bypasses img.complete & Firefox MJPEG bugs)
    ipcamAbortController = new AbortController();
    const proxyUrl = `/api/ipcam/stream?url=${encodeURIComponent(resolvedUrl)}&t=${Date.now()}`;
    const streamRes = await fetch(proxyUrl, { signal: ipcamAbortController.signal });

    if (!streamRes.ok || !streamRes.body) {
      throw new Error(`Proxy returned HTTP ${streamRes.status}`);
    }

    if (videoEl) videoEl.classList.add("hidden");
    ipcamCanvasEl.classList.remove("hidden");

    isIpcamRunning = true;
    hasReceivedHolisticResult = false;
    normalizer.reset();
    if (wsClient) wsClient.sendReset();

    if (btnToggleIpcam) {
      btnToggleIpcam.disabled = false;
      btnToggleIpcam.textContent = "Disconnect IP Webcam";
      btnToggleIpcam.classList.remove("btn-cyan");
      btnToggleIpcam.classList.add("btn-danger");
    }
    setIpcamStatus(`Connected to ${resolvedUrl} — streaming frames to MediaPipe`, "connected");
    if (hudStatus) hudStatus.textContent = "IP Webcam Active";

    // Start async MJPEG frame reader loop
    consumeMjpegStream(streamRes, resolvedUrl);
    console.log("IP Webcam connected:", resolvedUrl);
  } catch (err) {
    if (err.name === "AbortError") return;
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
  if (ipcamAbortController) {
    ipcamAbortController.abort();
    ipcamAbortController = null;
  }
  if (ipcamCanvasEl) {
    if (ipcamCtx) {
      ipcamCtx.clearRect(0, 0, ipcamCanvasEl.width, ipcamCanvasEl.height);
    }
    ipcamCanvasEl.classList.add("hidden");
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
  if (hudHands) {
    hudHands.textContent = "Pose: – | L: – | R: –";
    hudHands.className = "hud-tag";
  }
  if (hudFps) hudFps.textContent = "0 FPS";
  normalizer.reset();
  if (wsClient) wsClient.sendReset();
}

function toggleMirror() {
  isMirrored = !isMirrored;
  if (videoEl) videoEl.classList.toggle("unmirrored", !isMirrored);
  if (ipcamCanvasEl) ipcamCanvasEl.classList.toggle("unmirrored", !isMirrored);
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
        const handsTag = `L:${data.lh_detected ? "✓" : "–"} R:${data.rh_detected ? "✓" : "–"} | vel:${(data.vel_norm ?? 0).toFixed(2)}`;
        if (data.is_signing) {
          hudStatus.textContent = `Signing (${data.buffered_frames}f) • ${handsTag}`;
          hudStatus.className = "hud-tag hud-signing";
        } else {
          const stateLabel = data.pause_counter > 0 ? "Pause" : "Idle";
          hudStatus.textContent = `${stateLabel} • ${handsTag}`;
          hudStatus.className = "hud-tag";
        }
      }
    },
    onTranslating: (data) => {
      if (hudStatus) {
        hudStatus.textContent = `Translating (${data.num_frames} frames)...`;
        hudStatus.className = "hud-tag";
      }
      if (connDot) {
        connDot.className = "status-dot translating";
      }
    },
    onTranslation: (data) => {
      handleTranslationResult(data);
    },
    onNotice: (data) => {
      if (bannerMeta && data.message) {
        bannerMeta.textContent = data.message;
      }
      if (hudStatus && data.message) {
        hudStatus.textContent = data.message;
      }
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
    if (bannerMeta && clipData.reference_text) {
      bannerMeta.textContent = `Ground Truth: "${clipData.reference_text}"`;
    }

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

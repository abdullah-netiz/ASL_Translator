/**
 * app/static/js/socket.js — WebSocket Client for Real-Time ASL Translation.
 * Handles robust binary/JSON streaming, auto-reconnection, and server event dispatching.
 */

export class ASLWebSocketClient {
  constructor(options = {}) {
    this.url = options.url || this.getDefaultUrl();
    this.binaryMode = options.binaryMode !== undefined ? options.binaryMode : true;
    this.reconnectInterval = options.reconnectInterval || 2000;
    this.maxReconnectInterval = 10000;
    this.autoReconnect = true;

    this.ws = null;
    this.isConnected = false;
    this.currentBackoff = this.reconnectInterval;
    this.reconnectTimer = null;

    // Event listeners
    this.onConnectionChange = options.onConnectionChange || (() => {});
    this.onStatus = options.onStatus || (() => {});
    this.onTranslating = options.onTranslating || (() => {});
    this.onTranslation = options.onTranslation || (() => {});
    this.onError = options.onError || (() => {});
  }

  getDefaultUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    return `${proto}//${host}/ws/translate`;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    try {
      this.ws = new WebSocket(this.url);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => {
        this.isConnected = true;
        this.currentBackoff = this.reconnectInterval;
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
        this.onConnectionChange(true);
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        this.onConnectionChange(false);
        if (this.autoReconnect) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (err) => {
        this.onError(err);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleServerMessage(data);
        } catch (e) {
          console.error("Error parsing server message:", e);
        }
      };
    } catch (err) {
      console.error("Failed to initialize WebSocket:", err);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.currentBackoff = Math.min(this.currentBackoff * 1.5, this.maxReconnectInterval);
      this.connect();
    }, this.currentBackoff);
  }

  handleServerMessage(data) {
    switch (data.type) {
      case "status":
        this.onStatus(data);
        break;
      case "translating":
        this.onTranslating(data);
        break;
      case "translation":
        this.onTranslation(data);
        break;
      case "connected":
      case "pong":
      case "reset_ack":
        break;
      default:
        console.debug("Received message:", data);
    }
  }

  /**
   * Send a 1200-dim feature vector to the server.
   * @param {Float32Array|Array<number>} vector
   */
  sendFrame(vector) {
    if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    if (this.binaryMode && vector instanceof Float32Array) {
      // Send raw binary buffer (4800 bytes)
      this.ws.send(vector.buffer);
    } else {
      // Fallback JSON
      const arr = Array.isArray(vector) ? vector : Array.from(vector);
      this.ws.send(JSON.stringify({ type: "frame", landmarks: arr }));
    }
  }

  sendReset() {
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "reset" }));
    }
  }

  sendForceCut() {
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "force_cut" }));
    }
  }

  disconnect() {
    this.autoReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    this.onConnectionChange(false);
  }
}

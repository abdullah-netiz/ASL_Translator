# Web dashboard

The browser front-end of the ASL translator. It is served by the FastAPI server at `/`. There is no build step.

| File | Role |
|---|---|
| `index.html` | Layout: camera view with skeleton overlay, current sentence, transcript, recognised words, vocabulary panel, HUD |
| `static/js/landmarks.js` | Runs MediaPipe Holistic and converts each frame into the 1,200-number feature vector (bit-for-bit match with `src/extract/normalize.py`) |
| `static/js/socket.js` | WebSocket client for `/ws/translate`: streams frames, handles `status`, `live_words`, `translation` |
| `static/js/ui.js` | Camera / IP-webcam input, skeleton drawing, simulation of test clips, rendering of results |
| `static/css/style.css` | Dark theme, responsive layout |

What the user sees:

- **Hearing signs: …** words recognised so far while signing (`live_words`).
- **Translation** the English sentence when the sentence ends (~1.4 s pause or hands down), with each recognised
  word and its confidence.
- **Vocabulary panel** the words the server currently recognises (`/api/vocab`).
- **HUD** hand detection, motion level and buffered frames, for checking the camera set-up.

Camera access needs `http://localhost` or HTTPS. For other devices use a Cloudflare tunnel; see
[docs/DEMO_GUIDE.md](../docs/DEMO_GUIDE.md). Message formats are in [docs/API.md](../docs/API.md).

If you change the normalisation in `landmarks.js`, change `src/extract/normalize.py` the same way and run
`pytest tests/test_js_python_parity.py`. The models were trained on the Python features, so any mismatch silently
breaks recognition.

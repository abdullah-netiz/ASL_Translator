# Demo guide

How to start the translator, open it on any device, and get good results while signing.

## 1. Start the server

```bash
cd ~/Downloads/ASL/ASL_Translator
source .venv/bin/activate
python -m src.serve.server --port 8000
```

Wait for `Uvicorn running on http://0.0.0.0:8000` (about 30 s; it loads the sign models and the Qwen composer).
Then open **http://localhost:8000**.

Useful options:

| Option | Effect |
|---|---|
| `--vocab all` | Recognise all 2,304 words instead of the 366-word everyday list (more words, lower accuracy: 76% vs 89%) |
| `--vocab path/to/list.txt` | Recognise only the words in your own list (one word per line); fewer words = higher accuracy |
| `--pause-frames 30` | Wait longer (2 s instead of 1.4 s) before ending a sentence |
| `--min-raw 0.3` | Stricter: fewer false words when you sign something outside the list, but misses more real signs. Default 0.2; `0` = old, most permissive behaviour |
| `--port 8080` | Use a different port |

The words currently active are shown in the dashboard's vocabulary panel and at `http://localhost:8000/api/vocab`.

## 2. Open it on a phone or another laptop

Browsers only allow camera access on `localhost` or on HTTPS, so other devices need an HTTPS link. A Cloudflare quick
tunnel gives one for free, with no account:

```bash
~/.local/bin/cloudflared tunnel --edge-ip-version 4 --url http://localhost:8000
```

It prints a link like `https://some-random-words.trycloudflare.com`. Open it on the other device and allow the camera.

- The link changes every time the tunnel restarts (for example after a reboot). Start it shortly before the demo.
- The first page load can take up to ~30 s while Cloudflare connects; after that it is fast.
- A new link can take a minute or two to resolve on the PC that started it; other devices are usually fine at once.
- Anyone with the link can use the app. Stop sharing with `pkill -x cloudflared`.
- To run it in the background: `setsid nohup ~/.local/bin/cloudflared tunnel --edge-ip-version 4 --url http://localhost:8000 > data/cloudflared.log 2>&1 &`
  and read the link from `data/cloudflared.log`.

Recognition runs on the PC, so the device only needs a camera and a browser.

## 3. Using the dashboard

1. Click **Start Camera** (or pick an IP-webcam stream). The skeleton overlay should follow your shoulders and hands.
   Choose the **Mode**:
   - **Word**: sign one word, pause briefly (~0.6 s). You get exactly that word, plus "Other guesses".
     Best for showing individual signs.
   - **Sentence**: sign several words, then lower your hands (~1.4 s). They are combined into an English sentence.

   The choice is remembered in the browser.
2. Sign. While you sign, "Hearing signs: …" shows the words recognised so far.
3. Lower your hands or pause for ~1.4 s. The sentence is finished and the English translation appears, with the
   recognised words and their confidence underneath.
4. With no camera, **Simulate** plays recorded test clips through the same pipeline.

## 4. Signing tips for best accuracy

- **Framing:** head, shoulders and both hands fully in view, upper body roughly centred, about 1–1.5 m from the camera.
- **Light:** light on your face and hands, nothing bright behind you. Plain background helps.
- **Pace:** sign at a normal, clear pace. Keep hands up between signs of the same sentence; drop them to end it.
- **Words:** use words from the active list. Words outside it cannot be recognised.
- **Sentence order:** ASL order works well: time first (`TOMORROW ME GO SCHOOL`), question word last
  (`YOU NAME WHAT`), `NOT` before the verb (`ME NOT UNDERSTAND`).
- **Yes/no questions:** in ASL these are marked by raised eyebrows. The system cannot see that, so it treats
  "YOU …" without a question word as a question ("YOU DEAF" → "Are you deaf?").

### Sentences that demo well

| Sign | English |
|---|---|
| HELLO | Hello. |
| NICE MEET YOU | Nice to meet you. |
| YOU NAME WHAT | What is your name? |
| ME DEAF / ME HEARING | I am deaf. / I am hearing. |
| ME LEARN SIGN_LANGUAGE | I am learning sign language. |
| YOU UNDERSTAND | Do you understand? |
| ME NOT UNDERSTAND | I don't understand. |
| TOMORROW ME GO SCHOOL | I will go to school tomorrow. |
| ME WANT DRINK WATER | I want to drink water. |
| BATHROOM WHERE | Where is the bathroom? |
| THANK_YOU HELP ME | Thank you for helping me. |

If the recognised words do not form a sensible sentence, the app shows the words themselves ("Dog, cat, bird.")
instead of inventing a sentence. That is the safety check working, not a crash.

## 5. Make it better for your own signing

Accuracy improves a lot when the model has seen *your* signing:

```bash
python scripts/record_signs.py hello thank_you help water --n 8     # SPACE start/stop a take, S skip, Q quit
bash scripts/finetune_custom.sh                                     # ~3 h; writes checkpoints/sign/final_custom
# restart the server: it uses final_custom automatically (delete that folder to go back)
```

Record 6–10 takes per word, varying position and speed a little.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` on start | The GPU is shared. The composer falls back to a smaller model or the CPU automatically; if the sign models still fail, wait until other jobs finish, or stop other processes on the GPU (`nvidia-smi`). |
| Camera does not start on phone/laptop | Use the `https://` tunnel link, not `http://<ip>:8000`, and allow camera permission. |
| Skeleton visible but nothing is recognised | Check both hands are in view and the word is in the active list. Look at the HUD: "LH/RH detected" should light up. |
| Sentence cut in the middle | You paused too long between signs; keep hands up, or start with `--pause-frames 30`. |
| Wrong word | Try again more clearly, check lighting, or record your own samples (section 5). |
| Tunnel link does not open | Wait 1–2 minutes (new link), or restart the tunnel; check the server is still running at `http://localhost:8000`. |
| Port 8000 already in use | A previous server is still running: `pgrep -af src.serve.server`, then stop it, or use `--port 8001`. |

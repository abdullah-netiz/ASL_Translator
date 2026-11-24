# Testing bigger models in the cloud

Why this exists: Qwen2.5-3B needs about 8 GB of free GPU memory. On the shared lab PC other users' jobs usually leave
only 2–6 GB, so the server falls back to the 1.5B model. This page explains where a 3B can be tested properly, and how.

## What the student offers actually give (checked November 2026)

| Option | GPU? | Live app hosting allowed? | Verdict |
|---|---|---|---|
| Azure for Students ($100 credit) | **No**: GPU VM quota is 0; staff say upgrading to pay-as-you-go is required | Yes (own VM) | Only small CPU VMs: no better than the lab PC |
| AWS Educate / AWS Academy | No evidence of GPU instances on student credits | Yes (own VM) | Same as Azure |
| Kaggle notebooks | **Yes**: T4 ×2 or P100, ~30 h/week (community-reported) | Not addressed in official terms; community calls tunnels a grey area | **Use for testing** (notebook below) |
| Google Colab (free) | Yes: T4, no guaranteed hours | **No**: free tier restricts web UIs and remote proxies | **Use for testing** only |
| Hugging Face Spaces | No (free) | Code-running Spaces now need a paid plan | Not free any more |
| Paid GPU VM (AWS g4dn, Azure NCasT4_v3, …) | Yes | Yes | Works; costs money per hour; new accounts must request GPU quota first |

Sources:
- Azure: [NC family docs](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family), [Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/5857982/can-i-get-the-gpu-nc-series).
- AWS: [AWS Educate 2026 overview](https://builder.aws.com/content/3J0F6sCj6NWuOC3jqP6ZYMRyxrc/aws-educate-and-free-credit-programs-for-students-whats-actually-available-in-2026).
- Colab: [Colab FAQ](https://research.google.com/colaboratory/faq.html).
- Hugging Face: [Spaces docs](https://huggingface.co/docs/hub/en/spaces-overview), [HF forum](https://discuss.huggingface.co/t/free-cpu-basic-quota-is-0-on-new-account/180764).
- Kaggle quota: [free GPU overview](https://www.apartsin.com/teaching/guides/free-gpu-resources.html).

Offers change: check your own Azure portal (*Usage + quotas*) and each platform's current terms.

**Bottom line:** there is no free, rule-compliant way to *host* the live app on a cloud GPU with student benefits.
There is a free way to *test* the models properly: run the evaluation inside a Kaggle or Colab notebook.

## Measured so far

| Sentence model (115 held-out sentences) | Signed words kept | Invented content | Question form | chrF |
|---|---|---|---|---|
| Qwen2.5-1.5B + our fine-tune (deployed) | 99.8% | **0.0%** | 99.1% | **87.7** |
| Qwen2.5-3B, stock | 98.3% | 4.3% | 97.4% | 81.0 |
| Qwen2.5-3B + our fine-tune | not trained yet; notebook section C | | | |

The fine-tuned 1.5B already beats the stock 3B on every measure, so "switching to 3B" only makes sense if the 3B is
fine-tuned the same way. The notebook can do that.

## Run the tests on Kaggle or Colab

1. On the lab PC, build the bundle (code + models + held-out test clips, ~380 MB; no raw datasets):
   ```bash
   python scripts/package_for_cloud.py --with-tests     # -> dist/asl_cloud_bundle_with_tests.tar.gz
   ```
2. **Kaggle:**
   - Create a **private** dataset from the bundle (the sign datasets have research-use licences; do not publish it).
   - New notebook → *File → Import notebook* → `deploy/cloud_eval.ipynb`.
   - Settings → Accelerator **GPU T4 x2**, **Internet on**.
   - Add the dataset, then *Run all*.

   **Colab:**
   - Upload `deploy/cloud_eval.ipynb`.
   - Runtime → **T4 GPU**.
   - Put the bundle in Google Drive (or upload it when asked), then *Run all*.
3. The notebook reports:
   - (A) the 1.5B fine-tune vs the stock 3B;
   - (B) word-recognition accuracy in Word and Sentence mode through the real server code;
   - (C, optional) it fine-tunes the 3B (several hours on a T4), evaluates it, and lets you download `lora_3b.tar.gz`.

The code adapts to these GPUs automatically: T4 and P100 lack bf16, so inference and training switch to fp16, with
loss scaling for training. On the lab GPU, fp16 gave identical scores to bf16 (99.8% / 0% / 99.1% / 87.7).

## Using a 3B model on the lab PC

```bash
python -m src.serve.server --port 8000 --composer qwen-3b          # stock 3B
python -m src.serve.server --port 8000 --composer qwen-3b-lora     # your fine-tuned 3B (after section C)
python -m src.serve.server --port 8000 --composer none             # words only, no sentence model
```

The 3B needs ~8 GB of free GPU memory (`nvidia-smi`). If that is not available, the server logs a warning and loads
the fine-tuned 1.5B instead, so it never fails to start. Check which model loaded in the server log line
`Composer loaded: …`.

## If you need the 3B live for the presentation

Rent a GPU VM for a few hours on pay-as-you-go:
- AWS `g4dn.xlarge` (T4 16 GB), or
- Azure `Standard_NC4as_T4_v3`.

Request GPU quota a few days ahead. Then:

```bash
# on the VM (Ubuntu + NVIDIA driver image), after copying and unpacking dist/asl_cloud_bundle.tar.gz
cd ASL_Translator && python3 -m venv .venv && . .venv/bin/activate
pip install torch && pip install -r requirements-serve.txt && python -m nltk.downloader wordnet
python -m src.serve.server --port 8000 --composer qwen-3b-lora &
cloudflared tunnel --url http://localhost:8000          # HTTPS link for laptops/phones
```

Stop the VM afterwards; GPU VMs bill by the hour even when idle.

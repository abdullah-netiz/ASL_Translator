"""LoRA fine-tune of the Qwen gloss->English composer on data/sentences/{train,val}.tsv.
Loss only on the English answer tokens. Keeps the adapter with the LOWEST validation loss and stops early when
validation loss stops improving (patience). Waits for enough free VRAM before starting (shared GPU).
python scripts/train_composer.py [--model Qwen/Qwen2.5-1.5B-Instruct] [--out checkpoints/composer/lora_1.5b]"""
import sys, csv, json, math, random, time, argparse
from pathlib import Path
sys.path.insert(0, '.')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.sign.composer import SYSTEM_FT, GlossComposer, _gpu_dtype
from src.sign.resources import free_vram_gb

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--out", default="checkpoints/composer/lora_1.5b")
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--r", type=int, default=16); ap.add_argument("--eval-every", type=int, default=150); ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--min-vram", type=float, default=5.0); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
a = ap.parse_args()
torch.manual_seed(0); random.seed(0)

while free_vram_gb() < a.min_vram:
    print(f"waiting for VRAM ({free_vram_gb():.1f} GB free < {a.min_vram})", flush=True); time.sleep(120)

tok = AutoTokenizer.from_pretrained(a.model)
def encode(gloss, english):
    msgs = [{"role": "system", "content": SYSTEM_FT}, {"role": "user", "content": " ".join(GlossComposer.clean(g) for g in gloss.split())}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    ans = tok(english + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + ans, [-100] * len(p) + ans
def load(f): return [encode(r["gloss"], r["english"]) for r in csv.DictReader(open(f), delimiter="\t")]
tr, va = load("data/sentences/train.tsv"), load("data/sentences/val.tsv")
print(f"train {len(tr)} val {len(va)} | max len {max(len(x) for x, _ in tr)}", flush=True)
def batches(data, bs, shuffle):
    idx = list(range(len(data)));
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [data[j] for j in idx[i:i + bs]]; L = max(len(x) for x, _ in b)
        ids = torch.full((len(b), L), tok.pad_token_id); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
        for k, (x, y) in enumerate(b):
            ids[k, :len(x)] = torch.tensor(x); lab[k, :len(y)] = torch.tensor(y); att[k, :len(x)] = 1
        yield ids.cuda(), lab.cuda(), att.cuda()

DT = _gpu_dtype()          # bf16 on Ampere+; fp16 (+ loss scaling) on T4/P100/V100; override ASL_COMPOSER_DTYPE
if DT == torch.float32: DT = torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=DT == torch.float16)
print("training dtype", DT, flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=DT).cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
for n, p in model.named_parameters():
    if p.requires_grad: p.data = p.data.float()        # adapters in fp32, base in bf16
model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
total = a.epochs * math.ceil(len(tr) / (a.bs * a.accum)); warm = 50
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

@torch.no_grad()
def val_loss():
    model.eval(); tot = n = 0
    for ids, lab, att in batches(va, a.bs, False):
        with torch.autocast("cuda", dtype=DT): out = model(input_ids=ids, attention_mask=att, labels=lab)
        k = (lab[:, 1:] != -100).sum().item(); tot += out.loss.item() * k; n += k
    model.train(); return tot / n

best, bad, step, t0 = float("inf"), 0, 0, time.time(); hist = []
model.train(); stop = False
for ep in range(a.epochs):
    for ids, lab, att in batches(tr, a.bs, True):
        with torch.autocast("cuda", dtype=DT): loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        scaler.scale(loss / a.accum).backward(); micro = locals().get("micro", 0) + 1
        if micro % a.accum: continue
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if a.max_steps and step >= a.max_steps: stop = True; break
        if step % a.eval_every == 0:
            vl = val_loss(); hist.append((step, round(loss.item(), 4), round(vl, 4)))
            mark = ""
            if vl < best - 1e-4: best, bad = vl, 0; model.save_pretrained(a.out); mark = " *saved*"
            else: bad += 1
            print(f"ep{ep+1} step {step}/{total} train {loss.item():.4f} val {vl:.4f} best {best:.4f}{mark} [{time.time()-t0:.0f}s]", flush=True)
            if bad >= a.patience: print("early stop: val loss stopped improving"); stop = True; break
    if stop: break
Path(a.out).mkdir(parents=True, exist_ok=True)
json.dump({"args": vars(a), "best_val_loss": best, "history": hist}, open(f"{a.out}/results.json", "w"), indent=1)
print("DONE best val", best)

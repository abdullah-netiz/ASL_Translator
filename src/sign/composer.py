"""Gloss sequence -> fluent English using an instruction-tuned Qwen model.

This is the *right* job for an LLM in this system: the sign classifier supplies
(few, reliable) content words, Qwen only adds grammar/function words. It never
sees raw video/landmarks, so it cannot hallucinate content from a language prior
the way the old end-to-end T5 did.
"""
from __future__ import annotations
import os, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You convert American Sign Language gloss sequences into one short, natural English sentence. "
    "ASL gloss uses content words only, in ASL order (e.g. 'ME GO STORE TOMORROW'). "
    "Add only the grammar words needed (articles, auxiliaries, prepositions, pronoun forms). "
    "Never add new facts, names or extra clauses. If a gloss is a question word, make it a question. "
    "Reply with the sentence only."
)
# shorter prompt used by (and trained into) the LoRA fine-tuned composer: no few-shot examples needed
SYSTEM_FT = "Translate the ASL gloss into one natural English sentence. Keep every signed word; add only grammar words."
ADAPTER_DIR = "checkpoints/composer/lora_1.5b"

SHOTS = [
    ("ME GO STORE TOMORROW", "I am going to the store tomorrow."),
    ("YOU NAME WHAT", "What is your name?"),
    ("HELP ME PLEASE", "Please help me."),
    ("THANK-YOU", "Thank you."),
    ("MOTHER COOK DINNER", "Mother is cooking dinner."),
    ("YOU WANT DRINK WATER", "Do you want to drink some water?"),
]


_ROOT = Path(__file__).resolve().parents[2]
QWEN_15B, QWEN_3B = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"
# --composer presets: name -> (base model, adapter dir or None)
PRESETS = {
    "qwen-1.5b-lora": (QWEN_15B, "checkpoints/composer/lora_1.5b"),   # default: best measured
    "qwen-3b-lora": (QWEN_3B, "checkpoints/composer/lora_3b"),        # if you fine-tune the 3B (e.g. on a cloud GPU)
    "qwen-3b": (QWEN_3B, None),                                       # stock 3B, few-shot prompt
    "qwen-1.5b": (QWEN_15B, None),                                    # stock 1.5B, few-shot prompt
}


def _adapter_for(model_name: str) -> str | None:
    """Fine-tuned adapter that matches this base model, if it exists."""
    d = {QWEN_15B: "checkpoints/composer/lora_1.5b", QWEN_3B: "checkpoints/composer/lora_3b"}.get(model_name)
    return str(_ROOT / d) if d and (_ROOT / d / "adapter_config.json").exists() else None


def _gpu_dtype() -> torch.dtype:
    """bf16 on Ampere+ (RTX 30xx/40xx, A100, L4); fp16 on older GPUs without native bf16 (T4, P100, V100).
    Override with ASL_COMPOSER_DTYPE=bf16|fp16|fp32."""
    o = os.environ.get("ASL_COMPOSER_DTYPE")
    if o: return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[o]
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


class GlossComposer:
    FALLBACK = QWEN_15B

    def __init__(self, model_name: str = QWEN_3B, device: str = "cuda", adapter: str | None = "auto"):
        """Shared-GPU friendly: use the requested model only if it fits in free VRAM, else the 1.5B, else the 1.5B on CPU.
        model_name may be a preset name from PRESETS (e.g. "qwen-3b") or a Hugging Face model id.
        adapter="auto": use Qwen2.5-1.5B + its fine-tuned LoRA when that exists (it beats the stock 3B on the held-out
        set); adapter=None forces the stock few-shot model; a path uses that adapter."""
        import logging
        log = logging.getLogger("composer")
        if model_name in PRESETS:
            model_name, preset_adapter = PRESETS[model_name]
            adapter = str(_ROOT / preset_adapter) if preset_adapter else None
            if adapter and not (Path(adapter) / "adapter_config.json").exists():
                raise FileNotFoundError(f"adapter {adapter} not found - train it first (scripts/train_composer.py)")
        elif adapter == "auto":
            adapter = _adapter_for(QWEN_15B)
            if adapter:
                model_name = QWEN_15B                     # the default adapter was trained on the 1.5B
        # (model, device, adapter) attempts; a fallback model only uses an adapter trained for it
        fb_adapter = _adapter_for(QWEN_15B) if adapter else None
        tries = [(model_name, device, adapter)]
        if model_name != self.FALLBACK:
            tries.append((self.FALLBACK, device, fb_adapter))
        tries.append((self.FALLBACK, "cpu", fb_adapter if model_name != self.FALLBACK else adapter))
        params_b = {"3B": 3.1, "1.5B": 1.55}
        last = None
        for name, dev, ad in tries:
            try:
                if dev.startswith("cuda") and torch.cuda.is_available():
                    dtype = _gpu_dtype()
                    free = torch.cuda.mem_get_info()[0] / 1e9
                    bytes_per = 4 if dtype == torch.float32 else 2
                    need = next((v for k, v in params_b.items() if k in name), 1.55) * bytes_per + 1.5   # weights + working memory
                    if free < need:
                        log.warning("Only %.1f GB VRAM free (< %.1f needed for %s); trying a smaller option", free, need, name)
                        continue
                elif dev.startswith("cuda"):
                    continue
                else:
                    dtype = torch.float32
                adapter = ad
                self.tok = AutoTokenizer.from_pretrained(name)
                model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
                if adapter:                                   # merge the LoRA so inference costs the same as the base
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
                self.model = model.to(dev).eval()
                self.device, self.model_name = dev, name + (f" + LoRA({adapter})" if adapter else "")
                log.warning("Composer loaded: %s%s on %s (%s)", name, " + LoRA" if adapter else "", dev, str(dtype).replace("torch.", ""))
                break
            except torch.cuda.OutOfMemoryError as e:
                last = e; torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"Could not load a Qwen composer: {last}")
        if adapter:
            self._prefix = [{"role": "system", "content": SYSTEM_FT}]
        else:
            msgs = [{"role": "system", "content": SYSTEM}]
            for g, e in SHOTS:
                msgs += [{"role": "user", "content": g}, {"role": "assistant", "content": e}]
            self._prefix = msgs

    @staticmethod
    def clean(g: str) -> str:
        return re.sub(r"[^A-Za-z\-' ]", "", g.replace("_", "-")).strip().upper().replace(" ", "-")

    @torch.no_grad()
    def compose(self, glosses: list[str], guard: bool = True) -> dict:
        t0 = time.perf_counter()
        gl = [self.clean(g) for g in glosses if g]
        if not gl:
            return {"text": "", "latency_ms": 0.0}
        if len(gl) == 1:  # nothing to compose; avoid any LLM invention
            w = gl[0].replace("-", " ").lower()
            return {"text": w.capitalize() + ".", "latency_ms": 0.0}
        msgs = self._prefix + [{"role": "user", "content": " ".join(gl)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        out = self.model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
        # safety net: never show content that was not signed. If the sentence drops a signed word or invents a new
        # content word, show the recognised words themselves instead.
        from src.sign.faithful import score
        words = [g.replace("-", "_").lower() for g in gl]
        cov, added = score(" ".join(words), text)
        guarded = guard and (cov < 1.0 or bool(added))
        if guarded:
            text = ", ".join(w.replace("_", " ") for w in words).capitalize() + "."
        return {"text": text, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "guarded": guarded}

"""
Evaluation metrics for ASL-to-English translation.
Computes sacreBLEU (BLEU-1..4) and ROUGE-L, and formats prediction samples.
"""
from __future__ import annotations

from typing import Sequence
import sacrebleu
from rouge_score import rouge_scorer


def compute_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """
    Compute translation metrics:
      - sacreBLEU: BLEU (standard BLEU-4), BLEU-1, BLEU-2, BLEU-3, BLEU-4
      - ROUGE: ROUGE-1, ROUGE-2, ROUGE-L (F1 scores in percentage)
    """
    if len(hypotheses) == 0 or len(references) == 0:
        return {
            "bleu": 0.0,
            "bleu_1": 0.0,
            "bleu_2": 0.0,
            "bleu_3": 0.0,
            "bleu_4": 0.0,
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
        }

    # Ensure all strings are clean
    clean_hyps = [h.strip() if h.strip() else "." for h in hypotheses]
    clean_refs = [r.strip() if r.strip() else "." for r in references]

    # sacreBLEU expects list of hypotheses and list of reference lists: [[ref1, ref2, ...]]
    ref_list = [clean_refs]

    # BLEU-4 (standard corpus BLEU)
    bleu_res = sacrebleu.corpus_bleu(clean_hyps, ref_list)
    bleu_score = float(bleu_res.score)

    # Individual n-gram precisions
    precisions = bleu_res.precisions
    bleu_1 = float(precisions[0]) if len(precisions) > 0 else 0.0
    bleu_2 = float(precisions[1]) if len(precisions) > 1 else 0.0
    bleu_3 = float(precisions[2]) if len(precisions) > 2 else 0.0
    bleu_4 = float(precisions[3]) if len(precisions) > 3 else 0.0

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_list, r2_list, rl_list = [], [], []

    for hyp, ref in zip(clean_hyps, clean_refs):
        scores = scorer.score(ref, hyp)
        r1_list.append(scores["rouge1"].fmeasure * 100.0)
        r2_list.append(scores["rouge2"].fmeasure * 100.0)
        rl_list.append(scores["rougeL"].fmeasure * 100.0)

    avg_r1 = sum(r1_list) / len(r1_list) if r1_list else 0.0
    avg_r2 = sum(r2_list) / len(r2_list) if r2_list else 0.0
    avg_rl = sum(rl_list) / len(rl_list) if rl_list else 0.0

    return {
        "bleu": round(bleu_score, 2),
        "bleu_1": round(bleu_1, 2),
        "bleu_2": round(bleu_2, 2),
        "bleu_3": round(bleu_3, 2),
        "bleu_4": round(bleu_4, 2),
        "rouge_1": round(avg_r1, 2),
        "rouge_2": round(avg_r2, 2),
        "rouge_l": round(avg_rl, 2),
    }


def format_sample_table(
    clip_ids: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str],
    num_samples: int = 10,
) -> str:
    """Format markdown table comparing predictions against references."""
    lines = [
        "| Clip ID | Ground Truth Reference | Model Prediction |",
        "|---|---|---|",
    ]
    count = min(num_samples, len(hypotheses))
    for i in range(count):
        cid = clip_ids[i] if i < len(clip_ids) else f"sample_{i}"
        ref = references[i].replace("|", "\\|")
        hyp = hypotheses[i].replace("|", "\\|")
        lines.append(f"| `{cid}` | {ref} | {hyp} |")
    return "\n".join(lines)

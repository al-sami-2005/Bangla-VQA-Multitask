"""
Evaluation Metrics & Ablation Table Generator
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Responsibilities:
1. Calculate Exact Match (EM) and token-level F1 scores for Bengali text answers.
2. Compute ROUGE-L and BERTScore semantic alignment.
3. Compute Hallucination Metric: False Positive Rate (FPR) on counterfactual/negative samples.
4. Export comprehensive ablation tables to CSV (tracked in Git under `outputs/results/eval_results.csv`).
"""

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, load_config


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Computes exact string match accuracy between normalized Bengali predictions and references.
    """
    if not predictions or not references:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return (matches / len(predictions)) * 100.0


def compute_token_f1_bangla(predictions: List[str], references: List[str]) -> float:
    """
    Computes macro token-level F1 score across Bengali predictions and references.
    Tokenizes text by Unicode whitespace and Bengali punctuation.
    """
    if not predictions or not references:
        return 0.0
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.strip().split())
        ref_tokens = set(ref.strip().split())
        common = pred_tokens.intersection(ref_tokens)
        if not common:
            f1_scores.append(0.0)
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return (sum(f1_scores) / len(f1_scores)) * 100.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """
    Calculates ROUGE-L score using Longest Common Subsequence (LCS).
    """
    # Step 2 Implementation: evaluate.load('rouge') or rouge-score
    return 0.0


def compute_hallucination_rate(
    predictions: List[str],
    is_counterfactual: List[bool],
    refusal_keywords: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Measures the Hallucination / False Positive Rate on negative counterfactual samples.
    If an object is absent, the model should refuse or explicitly note absence (e.g. 'নেই').
    If the model generates an assertive description instead, it is counted as a hallucination.

    Returns:
        Dict with 'hallucination_rate_percent' and 'counterfactual_accuracy_percent'.
    """
    if refusal_keywords is None:
        refusal_keywords = ["নেই", "উপস্থিত নেই", "দেখা যাচ্ছে না", "না", "ভুল"]

    total_cf = sum(1 for cf in is_counterfactual if cf)
    if total_cf == 0:
        return {"hallucination_rate": 0.0, "counterfactual_accuracy": 100.0}

    hallucinations = 0
    for pred, cf in zip(predictions, is_counterfactual):
        if cf:
            # Check if any refusal keyword appears in prediction
            has_refusal = any(kw in pred for kw in refusal_keywords)
            if not has_refusal:
                hallucinations += 1

    hallucination_rate = (hallucinations / total_cf) * 100.0
    counterfactual_acc = 100.0 - hallucination_rate
    return {
        "hallucination_rate": hallucination_rate,
        "counterfactual_accuracy": counterfactual_acc,
    }


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    is_counterfactual: List[bool],
) -> Dict[str, float]:
    """
    Aggregates all evaluation metrics into a unified dictionary.
    """
    em = compute_exact_match(predictions, references)
    f1 = compute_token_f1_bangla(predictions, references)
    rouge_l = compute_rouge_l(predictions, references)
    hallucination_metrics = compute_hallucination_rate(predictions, is_counterfactual)

    return {
        "exact_match": em,
        "token_f1": f1,
        "rouge_l": rouge_l,
        "hallucination_rate": hallucination_metrics["hallucination_rate"],
        "counterfactual_accuracy": hallucination_metrics["counterfactual_accuracy"],
    }


def export_ablation_table(
    rows: List[Dict[str, Any]],
    output_path: str = "./outputs/results/eval_results.csv",
) -> None:
    """
    Exports ablation comparisons across configurations to a tracked CSV file.
    Columns: Model, Method, LoRA_r, Use_VCD, EM, F1, ROUGE-L, Hallucination_Rate
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Model",
        "Method",
        "LoRA_r",
        "Use_VCD",
        "EM",
        "Token_F1",
        "ROUGE_L",
        "Hallucination_Rate",
        "CF_Accuracy",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[evaluate_metrics] Exported ablation table with {len(rows)} rows to {path}")


def run_evaluation_pipeline(
    config: ExperimentConfig,
    output_csv: str = "./outputs/results/eval_results.csv",
) -> None:
    """
    Runs the complete evaluation and produces initial benchmark rows.
    """
    print("=" * 65)
    print(" [evaluate_metrics] Starting Evaluation & Ablation Logging")
    print(f"  Target Model: {config.model.base_model_id}")
    print(f"  Output CSV:   {output_csv}")
    print("=" * 65)

    # Initial placeholder baseline row for verification
    sample_rows = [
        {
            "Model": config.model.base_model_id,
            "Method": "Counterfactual Multi-Task + VCD" if config.decoding.use_vcd else "Counterfactual Multi-Task",
            "LoRA_r": config.lora.r,
            "Use_VCD": config.decoding.use_vcd,
            "EM": 0.0,
            "Token_F1": 0.0,
            "ROUGE_L": 0.0,
            "Hallucination_Rate": 0.0,
            "CF_Accuracy": 0.0,
        }
    ]
    export_ablation_table(sample_rows, output_path=output_csv)
    print("[evaluate_metrics] Ready for Step 2 benchmark evaluation.")


def main():
    config = load_config()
    run_evaluation_pipeline(config)


if __name__ == "__main__":
    main()

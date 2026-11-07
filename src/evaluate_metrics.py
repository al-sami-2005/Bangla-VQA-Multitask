"""
Evaluation Metrics & Publication Ablation Table Generator
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Core Responsibilities:
1. Parses multi-task Bangla output:
   - প্রাসঙ্গিকতা: {হ্যাঁ/না}
   - ভিজ্যুয়াল প্রমাণ: {grounding}
   - চূড়ান্ত উত্তর: {answer}
2. Computes Key Research Metrics:
   - Answerability Accuracy & Macro-F1 (identifying answerable vs unanswerable questions).
   - Hallucination Rate (% of unanswerable counterfactual samples where model failed to abstain).
   - Exact Match (EM) on Bengali final answers.
   - Token-level F1 for Bengali text.
   - ROUGE-L (Longest Common Subsequence).
3. Exports and appends results to `outputs/results/ablation_results_table.csv`.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, load_config


# ---------------------------------------------------------------------------
# Text Normalization & Output Parsing
# ---------------------------------------------------------------------------

PUNCTUATION_REGEX = re.compile(r"[\.,?!:;।\-—\"'“”‘’\(\)\[\]\{\}]")


def normalize_bangla_text(text: str) -> str:
    """
    Normalizes Bengali text by removing punctuation, collapsing whitespace,
    and trimming surrounding spaces.
    """
    if not text:
        return ""
    cleaned = PUNCTUATION_REGEX.sub(" ", str(text))
    return " ".join(cleaned.strip().split())


def parse_bangla_output(prediction_text: str) -> Dict[str, Any]:
    """
    Parses generated response into relevance, grounding, answer, and answerability prediction.
    Supports both 3-part multi-task outputs and 1-part baseline outputs.
    """
    relevance = ""
    grounding = ""
    answer = ""

    lines = [line.strip() for line in prediction_text.strip().split("\n") if line.strip()]
    for line in lines:
        if "প্রাসঙ্গিকতা:" in line or "প্রাসঙ্গিকতা :" in line:
            relevance = line.split(":", 1)[1].strip()
        elif "ভিজ্যুয়াল প্রমাণ:" in line or "ভিজ্যুয়াল প্রমাণ :" in line:
            grounding = line.split(":", 1)[1].strip()
        elif "চূড়ান্ত উত্তর:" in line or "চূড়ান্ত উত্তর :" in line:
            answer = line.split(":", 1)[1].strip()

    # Fallback if no explicit prefixes were identified
    if not answer:
        answer = prediction_text.strip()

    # Determine predicted answerability
    refusal_keywords = [
        "অনুপস্থিত",
        "উপস্থিত নেই",
        "দেখা যাচ্ছে না",
        "তথ্যটি নেই",
        "ছবিতে নেই",
        "নেই",
    ]

    is_pred_answerable = True
    rel_clean = normalize_bangla_text(relevance).lower()
    ans_clean = normalize_bangla_text(answer).lower()

    if rel_clean in ["না", "অপ্রাসঙ্গিক", "ভুল"]:
        is_pred_answerable = False
    elif any(kw in ans_clean for kw in refusal_keywords):
        is_pred_answerable = False

    return {
        "relevance": relevance,
        "grounding": grounding,
        "answer": answer,
        "is_answerable": is_pred_answerable,
    }


# ---------------------------------------------------------------------------
# Metric Calculation Engines
# ---------------------------------------------------------------------------

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Computes exact match percentage after Bengali normalization."""
    if not predictions:
        return 0.0
    matches = 0
    for p, r in zip(predictions, references):
        if normalize_bangla_text(p) == normalize_bangla_text(r):
            matches += 1
    return (matches / len(predictions)) * 100.0


def compute_token_f1(pred: str, ref: str) -> float:
    """Computes token-level F1 score for a single pair of Bengali texts."""
    pred_tokens = normalize_bangla_text(pred).split()
    ref_tokens = normalize_bangla_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_macro_token_f1(predictions: List[str], references: List[str]) -> float:
    """Computes mean token-level F1 percentage across all predictions."""
    if not predictions:
        return 0.0
    scores = [compute_token_f1(p, r) for p, r in zip(predictions, references)]
    return (sum(scores) / len(scores)) * 100.0


def compute_lcs(s1: List[str], s2: List[str]) -> int:
    """Computes Longest Common Subsequence length between token lists."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """Computes ROUGE-L F1 percentage across Bengali predictions and references."""
    if not predictions:
        return 0.0
    scores = []
    for p, r in zip(predictions, references):
        p_tokens = normalize_bangla_text(p).split()
        r_tokens = normalize_bangla_text(r).split()
        if not p_tokens or not r_tokens:
            scores.append(1.0 if p_tokens == r_tokens else 0.0)
            continue
        lcs_len = compute_lcs(p_tokens, r_tokens)
        if lcs_len == 0:
            scores.append(0.0)
            continue
        prec = lcs_len / len(p_tokens)
        rec = lcs_len / len(r_tokens)
        scores.append((2 * prec * rec) / (prec + rec))
    return (sum(scores) / len(scores)) * 100.0


def compute_answerability_metrics(
    pred_answerable: List[bool],
    true_answerable: List[bool],
) -> Dict[str, float]:
    """
    Computes Accuracy and Macro-F1 for binary Answerability detection
    (Answerable [Positive] vs Unanswerable [Negative Counterfactual]).
    """
    total = len(true_answerable)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    correct = sum(1 for p, t in zip(pred_answerable, true_answerable) if p == t)
    acc = (correct / total) * 100.0

    # True positives, False positives, False negatives for Class 1 (Answerable)
    tp1 = sum(1 for p, t in zip(pred_answerable, true_answerable) if p and t)
    fp1 = sum(1 for p, t in zip(pred_answerable, true_answerable) if p and not t)
    fn1 = sum(1 for p, t in zip(pred_answerable, true_answerable) if not p and t)
    f1_class1 = (2 * tp1 / (2 * tp1 + fp1 + fn1)) if (2 * tp1 + fp1 + fn1) > 0 else 0.0

    # Class 0 (Unanswerable / Counterfactual)
    tp0 = sum(1 for p, t in zip(pred_answerable, true_answerable) if not p and not t)
    fp0 = sum(1 for p, t in zip(pred_answerable, true_answerable) if not p and t)
    fn0 = sum(1 for p, t in zip(pred_answerable, true_answerable) if p and not t)
    f1_class0 = (2 * tp0 / (2 * tp0 + fp0 + fn0)) if (2 * tp0 + fp0 + fn0) > 0 else 0.0

    macro_f1 = ((f1_class1 + f1_class0) / 2.0) * 100.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def compute_hallucination_rate(
    pred_answerable: List[bool],
    true_answerable: List[bool],
) -> float:
    """
    Calculates Hallucination Rate on Counterfactual Samples:
    The percentage of unanswerable questions where the model failed to abstain
    (i.e. model mistakenly asserted that the absent object/attribute was present).
    """
    cf_indices = [i for i, is_ans in enumerate(true_answerable) if not is_ans]
    total_cf = len(cf_indices)
    if total_cf == 0:
        return 0.0

    # Failed abstentions: unanswerable samples where model predicted answerable=True
    failed_abstentions = sum(1 for i in cf_indices if pred_answerable[i])
    return (failed_abstentions / total_cf) * 100.0


# ---------------------------------------------------------------------------
# Evaluation Pipeline & Ablation Table Exporter
# ---------------------------------------------------------------------------

def evaluate_predictions_file(
    predictions_file: str,
    run_name: str = "eval_run",
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    output_csv_path: str = "outputs/results/ablation_results_table.csv",
) -> Dict[str, Any]:
    """
    Parses predictions JSONL file, computes all research metrics, and logs to CSV.
    """
    pred_path = Path(predictions_file)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    records = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    total_samples = len(records)
    if total_samples == 0:
        raise ValueError(f"No records found in {pred_path}")

    print("==================================================================")
    print(" Evaluating Predictions: Hallucination & Accuracy Metrics")
    print(f"  Input File:      {pred_path.resolve()}")
    print(f"  Total Samples:   {total_samples}")
    print(f"  Run Name:        {run_name}")
    print("==================================================================")

    true_answerable = []
    pred_answerable = []
    pred_answers = []
    gt_answers = []
    use_vcd = False

    for r in records:
        use_vcd = r.get("use_vcd", False)
        is_ans = bool(r.get("is_answerable", True))
        gt_ans = str(r.get("ground_truth_answer", ""))
        pred_text = str(r.get("prediction", ""))

        parsed = parse_bangla_output(pred_text)

        true_answerable.append(is_ans)
        pred_answerable.append(parsed["is_answerable"])
        pred_answers.append(parsed["answer"])
        gt_answers.append(gt_ans)

    # Calculate metrics
    ans_metrics = compute_answerability_metrics(pred_answerable, true_answerable)
    hallucination_rate = compute_hallucination_rate(pred_answerable, true_answerable)
    em = compute_exact_match(pred_answers, gt_answers)
    token_f1 = compute_macro_token_f1(pred_answers, gt_answers)
    rouge_l = compute_rouge_l(pred_answers, gt_answers)

    method_name = "Counterfactual Multi-Task"
    if use_vcd:
        method_name += " + VCD"

    results_row = {
        "Run_Name": run_name,
        "Model": model_name,
        "Method": method_name,
        "Use_VCD": use_vcd,
        "Total_Samples": total_samples,
        "Answerability_Acc": round(ans_metrics["accuracy"], 2),
        "Answerability_F1": round(ans_metrics["macro_f1"], 2),
        "Hallucination_Rate": round(hallucination_rate, 2),
        "EM": round(em, 2),
        "Token_F1": round(token_f1, 2),
        "ROUGE_L": round(rouge_l, 2),
    }

    # Export / Append to publication ablation CSV
    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results_row.keys())

    existing_rows = []
    if csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Deduplicate if run_name already exists
                if row.get("Run_Name") != run_name:
                    existing_rows.append(row)

    existing_rows.append(results_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print("\n==================================================================")
    print(" PUBLICATION METRICS REPORT")
    print("==================================================================")
    print(f"  Method:               {method_name}")
    print(f"  Answerability Acc:    {results_row['Answerability_Acc']}%")
    print(f"  Answerability F1:     {results_row['Answerability_F1']}%")
    print(f"  Hallucination Rate:   {results_row['Hallucination_Rate']}% (lower is better)")
    print(f"  Exact Match (EM):     {results_row['EM']}%")
    print(f"  Bengali Token F1:     {results_row['Token_F1']}%")
    print(f"  ROUGE-L Score:        {results_row['ROUGE_L']}%")
    print(f"  Ablation Table CSV:   {csv_path.resolve()}")
    print("==================================================================\n")

    return results_row


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bangla VQA Metrics & Generate Ablation Table")
    parser.add_argument(
        "--predictions_file",
        type=str,
        default="outputs/results/predictions_test_demo.jsonl",
        help="Input JSONL predictions file",
    )
    parser.add_argument("--run_name", type=str, default="eval_run", help="Run identifier")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/results/ablation_results_table.csv",
        help="Output ablation CSV path",
    )

    args, unknown = parser.parse_known_args()

    evaluate_predictions_file(
        predictions_file=args.predictions_file,
        run_name=args.run_name,
        model_name=args.model_name,
        output_csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()

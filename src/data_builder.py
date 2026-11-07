"""
Counterfactual Dataset Curation & Multi-Task Instruction Formatting
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Dataset Source: Remian9080/Bangla-Bayanno (Hugging Face Hub)
Repo Structure:
  - polar_dataset/qa.json (question_type='Polar', images in polar_dataset/images/{image_file})
  - numeric_dataset/qa.json (question_type='Quantitative', images in numeric_dataset/images/{image_file})
  - descriptive_dataset/qa.json (question_type='Descriptive', images in descriptive_dataset/images/{image_file})
Each qa.json record format:
  {'qa_id': int, 'image_id': int, 'image_file': str, 'question_bn': str, 'answer_bn': str}

Key Pipeline Steps:
1. Download 3 category qa.json files from Hugging Face Hub (hf_hub_download with urllib fallback).
2. Group QA records by unique `image_id` and select a balanced subset up to `--subset_size` (default 10,000).
3. Perform strict 80/10/10 image-level split by `image_id` (seed=42) to guarantee zero visual leakage.
4. Download only the unique images needed for the selected split using ThreadPoolExecutor(max_workers=16).
5. Apply 70% positive / 30% counterfactual negative curation and format multi-task instructions.
6. Export data/{train,val,test}.jsonl and save outputs/results/dataset_statistics.csv.
7. Support fast local verification via `--smoke_test`.
"""

import argparse
import concurrent.futures
import csv
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, load_config

# Optional Hugging Face Hub library import
try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    hf_hub_download = None
    HAS_HF_HUB = False


DEFAULT_HF_REPO = "Remian9080/Bangla-Bayanno"

DATASET_CATEGORIES = [
    {
        "folder": "polar_dataset",
        "qa_file": "polar_dataset/qa.json",
        "question_type": "Polar",
    },
    {
        "folder": "numeric_dataset",
        "qa_file": "numeric_dataset/qa.json",
        "question_type": "Quantitative",
    },
    {
        "folder": "descriptive_dataset",
        "qa_file": "descriptive_dataset/qa.json",
        "question_type": "Descriptive",
    },
]


# ---------------------------------------------------------------------------
# Bengali Visual Grounding Synthesis Templates
# ---------------------------------------------------------------------------

def synthesize_positive_grounding(question_type: str, answer: str) -> str:
    """
    Generates rule-based Bengali visual grounding rationale for factual positive samples.
    """
    clean_ans = str(answer).strip()
    if question_type == "Polar":
        return f"ছবিটি সতর্কতার সাথে পর্যবেক্ষণ করে প্রশ্নের প্রেক্ষাপট অনুযায়ী '{clean_ans}' সংক্রান্ত সুনির্দিষ্ট ভিজ্যুয়াল লক্ষণ শনাক্ত করা হয়েছে।"
    elif question_type == "Quantitative":
        return f"ছবিতে দৃশ্যমান উপাদানসমূহ পুঙ্খানুপুঙ্খভাবে গণনা করে প্রশ্নের আলোকে মোট '{clean_ans}' পরিমাণ সুস্পষ্টভাবে প্রত্যক্ষ করা গেছে।"
    else:  # Descriptive
        return f"ছবিটি পর্যবেক্ষণ করে প্রশ্নের আলোকে '{clean_ans}' সম্পর্কিত দৃশ্যমান প্রমাণ ও প্রাসঙ্গিক তথ্য নিশ্চিত করা গেছে।"


COUNTERFACTUAL_GROUNDING = "ছবিতে প্রশ্নে উল্লেখিত বস্তু বা প্রাসঙ্গিক প্রেক্ষাপট অনুপস্থিত।"
COUNTERFACTUAL_ANSWER = "ছবিতে এই তথ্যটি অনুপস্থিত।"
COUNTERFACTUAL_RELEVANCE = "না"
POSITIVE_RELEVANCE = "হ্যাঁ"

USER_PROMPT_TEMPLATE = (
    "<image>\n"
    "প্রশ্ন: {question}\n"
    "উপরের ছবিটি বিশ্লেষণ করে প্রশ্নটির প্রাসঙ্গিকতা, ভিজ্যুয়াল প্রমাণ এবং চূড়ান্ত উত্তর বাংলায় প্রদান করুন।"
)

ASSISTANT_RESPONSE_TEMPLATE = (
    "প্রাসঙ্গিকতা: {relevance}\n"
    "ভিজ্যুয়াল প্রমাণ: {grounding}\n"
    "চূড়ান্ত উত্তর: {answer}"
)


# ---------------------------------------------------------------------------
# Hugging Face Download Helper (hf_hub_download with robust urllib fallback)
# ---------------------------------------------------------------------------

def download_hf_file(
    repo_id: str,
    filename: str,
    destination: Path,
    max_retries: int = 3,
) -> Path:
    """
    Downloads a single file from Hugging Face Hub dataset repo.
    Uses hf_hub_download when available; otherwise falls back to direct HTTP streaming.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    # 1. Attempt using hf_hub_download
    if HAS_HF_HUB:
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
            )
            if Path(cached_path).resolve() != destination.resolve():
                shutil.copyfile(cached_path, destination)
            return destination
        except Exception:
            pass  # Fall through to direct HTTP

    # 2. HTTP download fallback
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    temp_dest = destination.with_suffix(destination.suffix + ".tmp")
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Bangla-VQA-DataBuilder"},
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_dest, "wb") as f_out:
                shutil.copyfileobj(response, f_out)
            temp_dest.replace(destination)
            return destination
        except Exception as e:
            if temp_dest.is_file():
                temp_dest.unlink()
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download '{url}' after {max_retries} attempts: {e}")
            time.sleep(1.0 * (attempt + 1))
    return destination


# ---------------------------------------------------------------------------
# Real Data Loader for Remian9080/Bangla-Bayanno
# ---------------------------------------------------------------------------

def load_bangla_bayanno_records(
    repo_id: str = DEFAULT_HF_REPO,
    raw_cache_dir: str = "data/raw",
) -> List[Dict[str, Any]]:
    """
    Downloads and aggregates all QA records from the three dataset folders:
    polar_dataset/qa.json, numeric_dataset/qa.json, and descriptive_dataset/qa.json.
    """
    cache_path = Path(raw_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []

    print(f"[data_builder] Fetching metadata from '{repo_id}' on Hugging Face Hub...")

    for cat in DATASET_CATEGORIES:
        folder = cat["folder"]
        qa_file = cat["qa_file"]
        q_type = cat["question_type"]

        dest_qa_path = cache_path / qa_file
        print(f"[data_builder] Downloading '{qa_file}'...")
        download_hf_file(repo_id=repo_id, filename=qa_file, destination=dest_qa_path)

        with open(dest_qa_path, "r", encoding="utf-8") as f:
            cat_data = json.load(f)

        print(f"[data_builder] Loaded {len(cat_data)} {q_type} items from '{qa_file}'.")

        for item in cat_data:
            img_file = item.get("image_file")
            if not img_file and "image_id" in item:
                img_file = f"COCO_train2014_{int(item['image_id']):012d}.jpg"
            q_bn = item.get("question_bn") or item.get("image_bn")
            a_bn = item.get("answer_bn")
            if not img_file or not q_bn or a_bn is None:
                continue

            repo_img_path = f"{folder}/images/{img_file}"
            all_records.append(
                {
                    "qa_id": item.get("qa_id", len(all_records)),
                    "image_id": item["image_id"],
                    "image_file": img_file,
                    "repo_image_path": repo_img_path,
                    "question_bn": q_bn,
                    "answer_bn": a_bn,
                    "question_type": q_type,
                }
            )

    print(f"[data_builder] Aggregated {len(all_records)} total raw QA pairs across all categories.")
    return all_records


def select_balanced_image_subset(
    records: List[Dict[str, Any]],
    target_subset_size: int = 10000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Groups QA records by unique image_id and selects a balanced subset of images
    until total QA pairs reach approximately `target_subset_size`.
    Ensures balanced representation across Polar, Quantitative, and Descriptive questions.
    """
    if target_subset_size <= 0 or target_subset_size >= len(records):
        print(f"[data_builder] Using full dataset ({len(records)} QA pairs).")
        return records

    rng = random.Random(seed)

    # Group records by image_id
    by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_image[r["image_id"]].append(r)

    # Bucket unique images by question type present
    quant_images = []
    polar_images = []
    desc_images = []

    for img_id, items in by_image.items():
        types = {it["question_type"] for it in items}
        if "Quantitative" in types:
            quant_images.append(img_id)
        elif "Polar" in types:
            polar_images.append(img_id)
        else:
            desc_images.append(img_id)

    rng.shuffle(quant_images)
    rng.shuffle(polar_images)
    rng.shuffle(desc_images)

    selected_image_ids: Set[int] = set()
    selected_qa_count = 0

    # Interleave image clusters from each question type bucket
    i_q, i_p, i_d = 0, 0, 0
    max_len = max(len(quant_images), len(polar_images), len(desc_images))

    for step in range(max_len):
        for bucket, idx in [
            (quant_images, i_q),
            (polar_images, i_p),
            (desc_images, i_d),
        ]:
            if idx < len(bucket):
                chosen_id = bucket[idx]
                if chosen_id not in selected_image_ids:
                    selected_image_ids.add(chosen_id)
                    selected_qa_count += len(by_image[chosen_id])
                if selected_qa_count >= target_subset_size:
                    break
        i_q += 1
        i_p += 1
        i_d += 1
        if selected_qa_count >= target_subset_size:
            break

    selected_records = [r for r in records if r["image_id"] in selected_image_ids]
    print(
        f"[data_builder] Selected {len(selected_image_ids)} unique images "
        f"({len(selected_records)} QA pairs, target ~{target_subset_size})."
    )
    return selected_records


# ---------------------------------------------------------------------------
# Strict Image-Level Splitting
# ---------------------------------------------------------------------------

def split_records_by_image(
    records: List[Dict[str, Any]],
    split_ratios: List[float] = [0.8, 0.1, 0.1],
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Partitions QA records by unique image_id with strict zero visual leakage.
    Train: 80%, Val: 10%, Test: 10%.
    """
    rng = random.Random(seed)

    by_image: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_image[r["image_id"]].append(r)

    unique_images = sorted(list(by_image.keys()), key=lambda x: str(x))
    rng.shuffle(unique_images)

    total_imgs = len(unique_images)
    n_train = max(1, int(total_imgs * split_ratios[0]))
    n_val = max(1, int(total_imgs * split_ratios[1]))
    if n_train + n_val >= total_imgs:
        n_val = max(1, (total_imgs - n_train) // 2)
    n_test = total_imgs - (n_train + n_val)
    if n_test <= 0:
        n_test = 1
        n_train = max(1, total_imgs - n_val - n_test)

    train_imgs = set(unique_images[:n_train])
    val_imgs = set(unique_images[n_train : n_train + n_val])
    test_imgs = set(unique_images[n_train + n_val :])

    # Enforce disjoint partitions
    assert train_imgs.isdisjoint(val_imgs), "Image leakage detected: Train and Val overlap!"
    assert train_imgs.isdisjoint(test_imgs), "Image leakage detected: Train and Test overlap!"
    assert val_imgs.isdisjoint(test_imgs), "Image leakage detected: Val and Test overlap!"

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for img_id, items in by_image.items():
        if img_id in train_imgs:
            splits["train"].extend(items)
        elif img_id in val_imgs:
            splits["val"].extend(items)
        else:
            splits["test"].extend(items)

    print(
        f"[data_builder] Image-level split completed across {total_imgs} unique images:\n"
        f"  - Train: {len(train_imgs)} images, {len(splits['train'])} QA pairs\n"
        f"  - Val:   {len(val_imgs)} images, {len(splits['val'])} QA pairs\n"
        f"  - Test:  {len(test_imgs)} images, {len(splits['test'])} QA pairs"
    )
    return splits


# ---------------------------------------------------------------------------
# Concurrent Image Downloader for Real Dataset
# ---------------------------------------------------------------------------

def download_images_for_split(
    split_name: str,
    split_records: List[Dict[str, Any]],
    output_dir: Path,
    repo_id: str,
    max_workers: int = 16,
) -> Dict[str, str]:
    """
    Downloads only the unique images required for a split concurrently.
    Saves them directly into data/images/{split_name}/{image_file}.
    Returns a mapping of image_file -> relative disk path.
    """
    images_dir = output_dir / "images" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate unique images needed for this split
    unique_images_map: Dict[str, str] = {}
    for r in split_records:
        img_file = r["image_file"]
        if img_file not in unique_images_map:
            unique_images_map[img_file] = r["repo_image_path"]

    total_images = len(unique_images_map)
    print(f"[data_builder] Concurrently downloading {total_images} unique images for '{split_name}' (workers={max_workers})...")

    def _download_task(img_file: str, repo_img_path: str) -> Tuple[str, str]:
        dest_path = images_dir / img_file
        # Check if already downloaded
        if not (dest_path.is_file() and dest_path.stat().st_size > 0):
            try:
                download_hf_file(repo_id=repo_id, filename=repo_img_path, destination=dest_path)
            except Exception:
                # If path failed in current folder, fallback to alternative folder paths
                alt_folders = ["polar_dataset", "numeric_dataset", "descriptive_dataset"]
                downloaded = False
                for f_alt in alt_folders:
                    alt_repo_path = f"{f_alt}/images/{img_file}"
                    try:
                        download_hf_file(repo_id=repo_id, filename=alt_repo_path, destination=dest_path)
                        downloaded = True
                        break
                    except Exception:
                        continue
                if not downloaded:
                    print(f"[data_builder] WARNING: Could not download image '{img_file}' from HF.")
        rel_path = f"{output_dir.name}/images/{split_name}/{img_file}"
        return img_file, rel_path

    path_mapping: Dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(_download_task, img_file, repo_path): img_file
            for img_file, repo_path in unique_images_map.items()
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_img):
            img_file, rel_path = future.result()
            path_mapping[img_file] = rel_path
            completed += 1
            if completed % 500 == 0 or completed == total_images:
                print(f"[data_builder] [{split_name}] Downloaded {completed}/{total_images} images...")

    return path_mapping


# ---------------------------------------------------------------------------
# 70% Positive / 30% Counterfactual Negative Curation
# ---------------------------------------------------------------------------

def apply_counterfactual_curation(
    split_records: List[Dict[str, Any]],
    image_paths_map: Dict[str, str],
    positive_ratio: float = 0.70,
    seed: int = 42,
    all_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Curates 70% Positive and 30% Counterfactual Negative samples.
    Borrow questions from different image clusters whose ground-truth answers do not match.
    """
    if not split_records:
        return []

    rng = random.Random(seed)

    by_image: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in split_records:
        by_image[r["image_id"]].append(r)

    all_image_ids = list(by_image.keys())
    curated_samples: List[Dict[str, Any]] = []

    for img_id, items in by_image.items():
        current_answers = {str(item["answer_bn"]).strip().lower() for item in items}

        # Build negative candidate pool from other image clusters in split
        other_ids = [o_id for o_id in all_image_ids if o_id != img_id]
        candidate_pool: List[Dict[str, Any]] = []
        for o_id in other_ids:
            for candidate in by_image[o_id]:
                cand_ans = str(candidate["answer_bn"]).strip().lower()
                if cand_ans not in current_answers:
                    candidate_pool.append(candidate)

        # Fallback to candidates from full dataset if split has only 1 image cluster
        if not candidate_pool and all_records:
            for candidate in all_records:
                if candidate["image_id"] != img_id:
                    cand_ans = str(candidate["answer_bn"]).strip().lower()
                    if cand_ans not in current_answers:
                        candidate_pool.append(candidate)

        items_copy = list(items)
        rng.shuffle(items_copy)

        n_total = len(items_copy)
        n_pos = max(1, round(n_total * positive_ratio))
        if n_pos >= n_total and n_total > 1:
            n_pos = n_total - 1

        for i, record in enumerate(items_copy):
            is_pos = i < n_pos or not candidate_pool
            img_file = record["image_file"]
            resolved_img_path = image_paths_map.get(img_file, f"data/images/{img_file}")

            if is_pos:
                q_type = record["question_type"]
                grounding = synthesize_positive_grounding(q_type, record["answer_bn"])
                curated_samples.append(
                    {
                        "image_id": record["image_id"],
                        "image_path": resolved_img_path,
                        "question": record["question_bn"],
                        "question_type": q_type,
                        "is_answerable": True,
                        "relevance": POSITIVE_RELEVANCE,
                        "grounding": grounding,
                        "answer": record["answer_bn"],
                    }
                )
            else:
                borrowed = rng.choice(candidate_pool)
                curated_samples.append(
                    {
                        "image_id": record["image_id"],
                        "image_path": resolved_img_path,
                        "question": borrowed["question_bn"],
                        "question_type": borrowed["question_type"],
                        "is_answerable": False,
                        "relevance": COUNTERFACTUAL_RELEVANCE,
                        "grounding": COUNTERFACTUAL_GROUNDING,
                        "answer": COUNTERFACTUAL_ANSWER,
                    }
                )

    rng.shuffle(curated_samples)
    return curated_samples


# ---------------------------------------------------------------------------
# Multi-Task JSONL Formatting & Export
# ---------------------------------------------------------------------------

def export_split_jsonl(
    samples: List[Dict[str, Any]],
    split_name: str,
    output_dir: Path,
) -> Path:
    """
    Exports samples to data/{split_name}.jsonl matching the multi-task conversation format.
    """
    jsonl_path = output_dir / f"{split_name}.jsonl"
    records = []

    for idx, s in enumerate(samples, start=1):
        sample_id = f"{split_name}_{idx:04d}"
        user_val = USER_PROMPT_TEMPLATE.format(question=s["question"])
        assistant_val = ASSISTANT_RESPONSE_TEMPLATE.format(
            relevance=s["relevance"],
            grounding=s["grounding"],
            answer=s["answer"],
        )

        record = {
            "sample_id": sample_id,
            "image_path": s["image_path"],
            "question": s["question"],
            "question_type": s["question_type"],
            "is_answerable": s["is_answerable"],
            "grounding": s["grounding"],
            "answer": s["answer"],
            "conversations": [
                {"from": "user", "value": user_val},
                {"from": "assistant", "value": assistant_val},
            ],
        }
        records.append(record)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[data_builder] Successfully exported {len(records)} samples to '{jsonl_path}'")
    return jsonl_path


def save_dataset_statistics_csv(
    curated_splits: Dict[str, List[Dict[str, Any]]],
    output_csv_path: str = "outputs/results/dataset_statistics.csv",
) -> None:
    """
    Saves a comprehensive breakdown table to outputs/results/dataset_statistics.csv
    """
    csv_file = Path(output_csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Split",
        "Unique_Images",
        "Total_Samples",
        "Positive_Samples",
        "Counterfactual_Samples",
        "Polar",
        "Quantitative",
        "Descriptive",
    ]

    rows = []
    tot_imgs, tot_samples, tot_pos, tot_cf, tot_polar, tot_quant, tot_desc = 0, 0, 0, 0, 0, 0, 0

    for split_name in ["train", "val", "test"]:
        items = curated_splits.get(split_name, [])
        u_imgs = len({it["image_id"] for it in items})
        t_samples = len(items)
        pos = sum(1 for it in items if it["is_answerable"])
        cf = sum(1 for it in items if not it["is_answerable"])
        polar = sum(1 for it in items if it["question_type"] == "Polar")
        quant = sum(1 for it in items if it["question_type"] == "Quantitative")
        desc = sum(1 for it in items if it["question_type"] == "Descriptive")

        tot_imgs += u_imgs
        tot_samples += t_samples
        tot_pos += pos
        tot_cf += cf
        tot_polar += polar
        tot_quant += quant
        tot_desc += desc

        rows.append(
            {
                "Split": split_name,
                "Unique_Images": u_imgs,
                "Total_Samples": t_samples,
                "Positive_Samples": pos,
                "Counterfactual_Samples": cf,
                "Polar": polar,
                "Quantitative": quant,
                "Descriptive": desc,
            }
        )

    # Total row
    rows.append(
        {
            "Split": "Total",
            "Unique_Images": tot_imgs,
            "Total_Samples": tot_samples,
            "Positive_Samples": tot_pos,
            "Counterfactual_Samples": tot_cf,
            "Polar": tot_polar,
            "Quantitative": tot_quant,
            "Descriptive": tot_desc,
        }
    )

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[data_builder] Dataset statistics written to: {csv_file}")
    print("=" * 86)
    print(f"{'Split':<8} | {'Images':<7} | {'Total':<7} | {'Positive':<9} | {'Counterfactual':<15} | {'Polar':<6} | {'Quant':<6} | {'Desc':<6}")
    print("-" * 86)
    for r in rows:
        print(
            f"{r['Split']:<8} | {r['Unique_Images']:<7} | {r['Total_Samples']:<7} | "
            f"{r['Positive_Samples']:<9} | {r['Counterfactual_Samples']:<15} | "
            f"{r['Polar']:<6} | {r['Quantitative']:<6} | {r['Descriptive']:<6}"
        )
    print("=" * 86 + "\n")


# ---------------------------------------------------------------------------
# Synthetic Smoke-Test (Fast Offline Verification)
# ---------------------------------------------------------------------------

SMOKE_TEST_SCENES = [
    {
        "bg_color": (220, 240, 255),
        "shape": ("rectangle", (30, 80, 190, 160), (220, 50, 50)),
        "qa_pairs": [
            ("ছবিতে কয়টি গাড়ি দেখা যাচ্ছে?", "১টি", "Quantitative"),
            ("গাড়ির প্রধান রং কি লাল?", "হ্যাঁ", "Polar"),
            ("গাড়িটি কোথায় অবস্থান করছে?", "রাস্তার পাশে", "Descriptive"),
            ("ছবিতে কি কোনো বাস উপস্থিত আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (245, 235, 220),
        "shape": ("circle", (50, 50, 170, 170), (50, 120, 220)),
        "qa_pairs": [
            ("ছবিতে কয়টি ফুটবল রাখা আছে?", "১টি", "Quantitative"),
            ("বলটির আকার কেমন?", "গোলাকার", "Descriptive"),
            ("বলটি কি খেলার মাঠে আছে?", "হ্যাঁ", "Polar"),
            ("মাঠে কি একাধিক খেলোয়াড় দেখা যাচ্ছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (230, 255, 230),
        "shape": ("rectangle", (60, 40, 160, 180), (34, 139, 34)),
        "qa_pairs": [
            ("ছবিতে কয়টি গাছ দৃশ্যমান?", "২টি", "Quantitative"),
            ("গাছের পাতার রং কি সবুজ?", "হ্যাঁ", "Polar"),
            ("গাছটি কোন পরিবেশে অবস্থিত?", "উন্মুক্ত বাগানে", "Descriptive"),
            ("বাগানে কি কোনো নদী প্রবাহিত হচ্ছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (255, 245, 230),
        "shape": ("circle", (60, 60, 160, 160), (255, 165, 0)),
        "qa_pairs": [
            ("টেবিলে কয়টি পাকা আম রাখা আছে?", "৩টি", "Quantitative"),
            ("আমগুলোর রং কি হলুদ ও কমলা?", "হ্যাঁ", "Polar"),
            ("ফলগুলো কোথায় রাখা হয়েছে?", "খাবারের টেবিলের ওপর", "Descriptive"),
            ("টেবিলে কি কোনো আপেল আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (210, 230, 250),
        "shape": ("polygon", [(40, 140), (180, 140), (150, 180), (70, 180)], (139, 69, 19)),
        "qa_pairs": [
            ("নদীতে কয়টি নৌকা ভাসছে?", "১টি", "Quantitative"),
            ("নৌকাটি কি পানিতে ভাসমান অবস্থায় আছে?", "হ্যাঁ", "Polar"),
            ("নৌকাটি কি কাঠের তৈরি?", "হ্যাঁ", "Polar"),
            ("নদীর পানির রং কেমন?", "নীলাভ ও স্বচ্ছ", "Descriptive"),
        ],
    },
]


def run_smoke_test_pipeline(
    output_dir: Path,
    split_ratios: List[float] = [0.8, 0.1, 0.1],
    positive_ratio: float = 0.70,
    seed: int = 42,
) -> Dict[str, Path]:
    """Fast local synthetic pipeline producing 100 samples across 25 scenes."""
    print("[data_builder] Running Fast Local Synthetic Smoke Test...")
    rng = random.Random(seed)

    raw_samples = []
    sample_counter = 0
    images_by_id: Dict[str, Image.Image] = {}

    for scene_idx in range(25):
        scene = SMOKE_TEST_SCENES[scene_idx % len(SMOKE_TEST_SCENES)]
        img_id = f"smoke_img_{scene_idx:03d}"
        img = Image.new("RGB", (224, 224), color=scene["bg_color"])
        draw = ImageDraw.Draw(img)
        shape_type, coords, fill_color = scene["shape"]
        if shape_type == "rectangle":
            draw.rectangle(coords, fill=fill_color, outline=(20, 20, 20), width=2)
        elif shape_type == "circle":
            draw.ellipse(coords, fill=fill_color, outline=(20, 20, 20), width=2)
        elif shape_type == "polygon":
            draw.polygon(coords, fill=fill_color, outline=(20, 20, 20))
        images_by_id[img_id] = img

        for q, a, q_type in scene["qa_pairs"]:
            sample_counter += 1
            raw_samples.append(
                {
                    "qa_id": sample_counter,
                    "image_id": img_id,
                    "image_file": f"{img_id}.jpg",
                    "question_bn": q,
                    "answer_bn": a,
                    "question_type": q_type,
                }
            )

    splits = split_records_by_image(raw_samples, split_ratios=split_ratios, seed=seed)
    curated_splits: Dict[str, List[Dict[str, Any]]] = {}
    exported_paths: Dict[str, Path] = {}

    for split_name, records in splits.items():
        img_dir = output_dir / "images" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path_map: Dict[str, str] = {}

        for r in records:
            img_file = r["image_file"]
            img_id = r["image_id"]
            disk_path = img_dir / img_file
            if not disk_path.is_file():
                images_by_id[img_id].save(disk_path, format="JPEG", quality=95)
            img_path_map[img_file] = f"{output_dir.name}/images/{split_name}/{img_file}"

        curated = apply_counterfactual_curation(
            split_records=records,
            image_paths_map=img_path_map,
            positive_ratio=positive_ratio,
            seed=seed,
        )
        curated_splits[split_name] = curated
        exported_paths[split_name] = export_split_jsonl(curated, split_name, output_dir)

    save_dataset_statistics_csv(curated_splits)
    return exported_paths


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_data_pipeline(
    config: ExperimentConfig,
    smoke_test: bool = False,
    hf_dataset_id: Optional[str] = None,
    subset_size: Optional[int] = None,
    output_dir: str = "data",
    max_workers: int = 16,
) -> Dict[str, Path]:
    """
    Orchestrates dataset curation, strict image-level splitting, concurrent image download,
    counterfactual curation, and multi-task JSONL export.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_repo = hf_dataset_id or config.dataset.name
    if target_repo == "Bangla-Bayanno":
        target_repo = DEFAULT_HF_REPO

    target_subset = subset_size if subset_size is not None else config.dataset.subset_size
    seed = config.dataset.seed
    split_ratios = config.dataset.split_ratios
    positive_ratio = config.dataset.positive_ratio

    print("==================================================================")
    print(" Bangla-VQA Data Pipeline: Remian9080/Bangla-Bayanno")
    print(f"  Target HF Repo:      {target_repo}")
    print(f"  Smoke Test Mode:     {smoke_test}")
    print(f"  Target Subset Size:  {target_subset}")
    print(f"  Split Ratios:        {split_ratios}")
    print(f"  Positive Ratio:      {positive_ratio * 100:.1f}%")
    print(f"  Counterfactual %:    {config.dataset.negative_counterfactual_ratio * 100:.1f}%")
    print(f"  Random Seed:         {seed}")
    print(f"  Max Workers:         {max_workers}")
    print(f"  Output Directory:    {out_path.resolve()}")
    print("==================================================================")

    # 1. Fast offline smoke-test branch
    if smoke_test:
        return run_smoke_test_pipeline(
            output_dir=out_path,
            split_ratios=split_ratios,
            positive_ratio=positive_ratio,
            seed=seed,
        )

    # 2. Real data mode: download metadata JSONs
    raw_records = load_bangla_bayanno_records(repo_id=target_repo)

    # 3. Balanced subset selection by image_id
    selected_records = select_balanced_image_subset(
        records=raw_records,
        target_subset_size=target_subset,
        seed=seed,
    )

    # 4. Strict 80/10/10 image-level split
    split_dict = split_records_by_image(
        records=selected_records,
        split_ratios=split_ratios,
        seed=seed,
    )

    # 5. Concurrent image downloading and counterfactual curation per split
    curated_splits: Dict[str, List[Dict[str, Any]]] = {}
    exported_paths: Dict[str, Path] = {}

    for split_name in ["train", "val", "test"]:
        records = split_dict[split_name]
        image_path_map = download_images_for_split(
            split_name=split_name,
            split_records=records,
            output_dir=out_path,
            repo_id=target_repo,
            max_workers=max_workers,
        )
        curated = apply_counterfactual_curation(
            split_records=records,
            image_paths_map=image_path_map,
            positive_ratio=positive_ratio,
            seed=seed,
            all_records=selected_records,
        )
        curated_splits[split_name] = curated
        exported_paths[split_name] = export_split_jsonl(curated, split_name, out_path)

    # 6. Save comprehensive dataset statistics CSV
    save_dataset_statistics_csv(curated_splits)
    return exported_paths


def main():
    parser = argparse.ArgumentParser(description="Bangla VQA Counterfactual Data Builder (Native HF Hub)")
    parser.add_argument("--smoke_test", action="store_true", help="Generate fast local synthetic dataset (no downloads)")
    parser.add_argument("--hf_dataset_id", type=str, default=DEFAULT_HF_REPO, help="HF Hub dataset repository")
    parser.add_argument("--subset_size", type=int, default=None, help="Target QA subset size to curate (default: from config)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max_workers", type=int, default=16, help="Concurrent image download worker threads")

    args, unknown = parser.parse_known_args()
    config = load_config(cli_args=unknown)

    run_data_pipeline(
        config=config,
        smoke_test=args.smoke_test,
        hf_dataset_id=args.hf_dataset_id,
        subset_size=args.subset_size,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()

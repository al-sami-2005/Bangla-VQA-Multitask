"""
Counterfactual Dataset Curation & Multi-Task Instruction Formatting
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Core Responsibilities:
1. Dataset Loading: Load Bangla VQA data from Hugging Face or generate synthetic smoke-test data.
2. Strict Image-Level Splitting: Partition unique images (80% Train, 10% Val, 10% Test) to prevent visual leakage.
3. 70% Positive / 30% Counterfactual Negative Generation:
   - Positives (70%): Original question & answer, with synthesized Bangla visual grounding statement.
   - Negatives (30%): Borrow questions from distinct image clusters whose answers do not match the current image,
     setting relevance='না', grounding='ছবিতে প্রশ্নে উল্লেখিত বস্তু বা প্রাসঙ্গিক প্রেক্ষাপট অনুপস্থিত।', and answer='ছবিতে এই তথ্যটি অনুপস্থিত।'
4. Multi-Task Output Formatting: Export image files to data/images/{split}/ and JSONL files with multi-task conversation templates.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, load_config


# ---------------------------------------------------------------------------
# Bengali Question Classification & Grounding Synthesis
# ---------------------------------------------------------------------------

POLAR_KEYWORDS = ["কি ", " কি", "আছে কি", "হচ্ছে কি", "দেখা যাচ্ছে কি", "উপস্থিত কি", "নাকি", "সম্ভব কি"]
POLAR_ANSWERS = ["হ্যাঁ", "না", "হাঁ", "জি", "জি না", "না, নেই", "হ্যাঁ, আছে"]

QUANTITATIVE_KEYWORDS = ["কত", "কয়টি", "কয়জন", "কতগুলো", "সংখ্যা", "কয়টা", "কতজন", "কতখানি", "কতটুকু"]
QUANTITATIVE_NUMERALS = ["০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def classify_bangla_question(question: str, answer: str = "") -> str:
    """
    Classifies a Bengali question into Quantitative, Polar, or Descriptive.
    """
    q_lower = question.strip()
    a_lower = answer.strip()

    # 1. Quantitative check
    if any(kw in q_lower for kw in QUANTITATIVE_KEYWORDS):
        return "Quantitative"
    if any(char in a_lower for char in QUANTITATIVE_NUMERALS):
        return "Quantitative"

    # 2. Polar check
    if a_lower in POLAR_ANSWERS:
        return "Polar"
    if any(kw in q_lower for kw in POLAR_KEYWORDS):
        return "Polar"

    # 3. Default to Descriptive
    return "Descriptive"


def synthesize_positive_grounding(question_type: str, answer: str) -> str:
    """
    Generates a rule-based Bengali visual grounding rationale for factual samples.
    """
    clean_ans = answer.strip()
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


# ---------------------------------------------------------------------------
# Synthetic Smoke-Test Sample Generator
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
    {
        "bg_color": (255, 250, 240),
        "shape": ("rectangle", (50, 50, 170, 170), (70, 70, 70)),
        "qa_pairs": [
            ("টেবিলে কয়টি খোলা ল্যাপটপ আছে?", "১টি", "Quantitative"),
            ("ল্যাপটপের পর্দা কি চালু আছে?", "হ্যাঁ", "Polar"),
            ("ল্যাপটপটি কোথায় স্থাপন করা হয়েছে?", "অফিসের ডেস্কে", "Descriptive"),
            ("ডেস্কে কি কোনো কফি মগ আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (255, 235, 235),
        "shape": ("circle", (70, 70, 150, 150), (200, 30, 30)),
        "qa_pairs": [
            ("প্লেটে কয়টি লাল আপেল রয়েছে?", "২টি", "Quantitative"),
            ("আপেলটি কি তাজা দেখা যাচ্ছে?", "হ্যাঁ", "Polar"),
            ("ফলটি কোন পাত্রে রাখা হয়েছে?", "সাদা প্লেটে", "Descriptive"),
            ("প্লেটে কি কোনো কলা রয়েছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (230, 240, 230),
        "shape": ("rectangle", (40, 60, 180, 160), (100, 149, 237)),
        "qa_pairs": [
            ("শ্রেণীকক্ষে কয়টি ব্ল্যাকবোর্ড আছে?", "১টি", "Quantitative"),
            ("বোর্ডে কি কিছু লেখা আছে?", "হ্যাঁ", "Polar"),
            ("বোর্ডটি কোথায় টাঙানো হয়েছে?", "দেয়ালের ওপর", "Descriptive"),
            ("কক্ষে কি প্রজেক্টর চালু আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (240, 255, 255),
        "shape": ("circle", (50, 50, 170, 170), (255, 215, 0)),
        "qa_pairs": [
            ("আকাশে কি সূর্য দেখা যাচ্ছে?", "হ্যাঁ", "Polar"),
            ("সূর্যের আভা কেমন?", "উজ্জ্বল সোনালী", "Descriptive"),
            ("আকাশে কয়টি সূর্য দৃশ্যমান?", "১টি", "Quantitative"),
            ("আকাশে কি ঘন কালো মেঘ আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (245, 245, 245),
        "shape": ("rectangle", (60, 50, 160, 170), (160, 82, 45)),
        "qa_pairs": [
            ("ছবিতে কয়টি কাঠের চেয়ার দৃশ্যমান?", "১টি", "Quantitative"),
            ("চেয়ারটি কি কাঠের তৈরি?", "হ্যাঁ", "Polar"),
            ("চেয়ারের ওপর কি কেউ বসে আছে?", "না", "Polar"),
            ("চেয়ারটির অবস্থান কোথায়?", "ঘরের কোণে", "Descriptive"),
        ],
    },
    {
        "bg_color": (235, 245, 235),
        "shape": ("polygon", [(110, 40), (170, 160), (50, 160)], (46, 139, 87)),
        "qa_pairs": [
            ("ছবিতে কয়টি পর্বতশৃঙ্গ দৃশ্যমান?", "১টি", "Quantitative"),
            ("পাহাড়ের গায়ে কি গাছপালা আছে?", "হ্যাঁ", "Polar"),
            ("পাহাড়ের চূড়া দেখতে কেমন?", "খাড়া ও সূক্ষ্ম", "Descriptive"),
            ("পাহাড়ে কি বরফ জমে আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (255, 240, 245),
        "shape": ("circle", (60, 60, 160, 160), (219, 112, 147)),
        "qa_pairs": [
            ("ফুলদানিতে কয়টি গোলাপ ফুল আছে?", "৪টি", "Quantitative"),
            ("গোলাপের পাপড়ির রং কি গোলাপী?", "হ্যাঁ", "Polar"),
            ("ফুলগুলো কোথায় সাজানো?", "কাঁচের ফুলদানিতে", "Descriptive"),
            ("ফুলদানিতে কি কোনো পদ্ম ফুল আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (240, 240, 250),
        "shape": ("rectangle", (50, 70, 170, 150), (112, 128, 144)),
        "qa_pairs": [
            ("রাস্তায় কয়টি বাস চলাচল করছে?", "১টি", "Quantitative"),
            ("বাসটি কি গতিশীল অবস্থায় রয়েছে?", "হ্যাঁ", "Polar"),
            ("বাসটির রং কেমন?", "ধূসর ও নীল", "Descriptive"),
            ("বাসের ছাদে কি কোনো যাত্রী আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (250, 250, 230),
        "shape": ("circle", (70, 70, 150, 150), (255, 140, 0)),
        "qa_pairs": [
            ("গাছের ডালে কয়টি পাখি বসে আছে?", "২টি", "Quantitative"),
            ("পাখিগুলোর পালকের রং কি কমলা?", "হ্যাঁ", "Polar"),
            ("পাখিগুলো কি উড়ন্ত অবস্থায় আছে?", "না", "Polar"),
            ("পাখির ঠোঁট দেখতে কেমন?", "ধারালো ও ছোট", "Descriptive"),
        ],
    },
    {
        "bg_color": (230, 250, 250),
        "shape": ("rectangle", (60, 40, 160, 180), (0, 128, 128)),
        "qa_pairs": [
            ("ঘরে কয়টি বুকশেলফ রয়েছে?", "১টি", "Quantitative"),
            ("শেলফে কি অনেকগুলো বই সাজানো আছে?", "হ্যাঁ", "Polar"),
            ("শেলফটি কোথায় অবস্থিত?", "লাইব্রেরির দেয়ালে", "Descriptive"),
            ("শেলফের ওপর কি টেলিভিশন আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (255, 248, 220),
        "shape": ("circle", (80, 80, 140, 140), (218, 165, 32)),
        "qa_pairs": [
            ("থালায় কয়টি মিষ্টি সাজানো আছে?", "৫টি", "Quantitative"),
            ("মিষ্টিগুলোর রং কি সোনালী?", "হ্যাঁ", "Polar"),
            ("মিষ্টিগুলো কোন পাত্রে রাখা?", "পিতলের থালায়", "Descriptive"),
            ("থালায় কি কোনো নোনতা খাবার আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (240, 255, 240),
        "shape": ("rectangle", (70, 60, 150, 160), (85, 107, 47)),
        "qa_pairs": [
            ("ছবিতে কয়টি সাইকেল দেখা যাচ্ছে?", "১টি", "Quantitative"),
            ("সাইকেলে কি দুটি চাকা আছে?", "হ্যাঁ", "Polar"),
            ("সাইকেলটি কোথায় রাখা হয়েছে?", "ফুটপাতের পাশে", "Descriptive"),
            ("সাইকেলের সাথে কি কোনো মোটর ইঞ্জিন আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (245, 240, 255),
        "shape": ("polygon", [(110, 50), (170, 170), (50, 170)], (147, 112, 219)),
        "qa_pairs": [
            ("ছবিতে কয়টি তাবু দৃশ্যমান?", "১টি", "Quantitative"),
            ("তাবুটি কি পাহাড়ি সমতলে খাটানো হয়েছে?", "হ্যাঁ", "Polar"),
            ("তাবুর রং কি বেগুনি?", "হ্যাঁ", "Polar"),
            ("তাবুর ভেতরে কি আলো জ্বলছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (255, 240, 240),
        "shape": ("rectangle", (50, 50, 170, 170), (178, 34, 34)),
        "qa_pairs": [
            ("ছবিতে কয়টি লাল ইট দেখা যাচ্ছে?", "৬টি", "Quantitative"),
            ("ইটগুলো কি একটি আরেকটির ওপর স্তূপ করা?", "হ্যাঁ", "Polar"),
            ("ইটের উপরিভাগ কেমন?", "শক্ত ও রুক্ষ", "Descriptive"),
            ("ইটের গায়ে কি কোনো জলছাপ আছে?", "না", "Polar"),
        ],
    },
    {
        "bg_color": (230, 245, 255),
        "shape": ("circle", (70, 70, 150, 150), (30, 144, 255)),
        "qa_pairs": [
            ("আকাশে কয়টি ঘুড়ি উড়ছে?", "৩টি", "Quantitative"),
            ("ঘুড়িগুলো কি রঙিন?", "হ্যাঁ", "Polar"),
            ("ঘুড়িগুলো কোথায় উড়ছে?", "উন্মুক্ত খোলা আকাশে", "Descriptive"),
            ("আকাশে কি হেলিকপ্টার উড়ছে?", "না", "Polar"),
        ],
    },
]


def create_synthetic_image(scene_def: Dict[str, Any], size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Creates a distinct synthetic RGB PIL image for smoke testing."""
    img = Image.new("RGB", size, color=scene_def["bg_color"])
    draw = ImageDraw.Draw(img)
    shape_type, coords, fill_color = scene_def["shape"]
    if shape_type == "rectangle":
        draw.rectangle(coords, fill=fill_color, outline=(20, 20, 20), width=2)
    elif shape_type == "circle":
        draw.ellipse(coords, fill=fill_color, outline=(20, 20, 20), width=2)
    elif shape_type == "polygon":
        draw.polygon(coords, fill=fill_color, outline=(20, 20, 20))
    return img


def generate_smoke_test_dataset(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Generates synthetic Bangla VQA data structured by image clusters.
    """
    print(f"[data_builder] Generating {num_samples} synthetic smoke-test samples across 20 scenes...")
    samples = []
    sample_counter = 0

    # Repeat or cycle through scenes until we hit desired sample count
    scene_idx = 0
    while len(samples) < num_samples:
        scene = SMOKE_TEST_SCENES[scene_idx % len(SMOKE_TEST_SCENES)]
        image_id = f"smoke_img_{scene_idx:03d}"
        pil_image = create_synthetic_image(scene)

        for q, a, q_type in scene["qa_pairs"]:
            if len(samples) >= num_samples:
                break
            sample_counter += 1
            samples.append(
                {
                    "sample_uid": f"raw_{sample_counter:04d}",
                    "image_id": image_id,
                    "image": pil_image,
                    "question": q,
                    "answer": a,
                    "question_type": q_type,
                }
            )
        scene_idx += 1

    print(f"[data_builder] Generated {len(samples)} raw samples across {scene_idx} unique images.")
    return samples


# ---------------------------------------------------------------------------
# Hugging Face Dataset Loader
# ---------------------------------------------------------------------------

def load_huggingface_dataset(
    hf_dataset_id: str,
    subset_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Attempts to load a dataset from Hugging Face datasets library with automatic fallback.
    """
    print(f"[data_builder] Attempting to load '{hf_dataset_id}' from Hugging Face Hub...")
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_dataset_id, split="train")
        if subset_size and len(ds) > subset_size:
            ds = ds.select(range(subset_size))

        samples = []
        for idx, item in enumerate(ds):
            # Extract image
            img = item.get("image")
            if img is None:
                continue

            # Determine unique image identifier
            if hasattr(img, "filename") and img.filename:
                image_id = Path(img.filename).stem
            else:
                image_id = f"hf_img_{idx:05d}"

            q = item.get("question") or item.get("question_bn") or item.get("query") or ""
            a = item.get("answer") or item.get("answer_bn") or item.get("label") or ""
            q_type = item.get("question_type") or classify_bangla_question(q, a)

            samples.append(
                {
                    "sample_uid": f"hf_{idx:05d}",
                    "image_id": image_id,
                    "image": img,
                    "question": q,
                    "answer": a,
                    "question_type": q_type,
                }
            )

        print(f"[data_builder] Successfully loaded {len(samples)} samples from Hugging Face.")
        return samples

    except Exception as e:
        print(f"[data_builder] WARNING: Could not load '{hf_dataset_id}' from Hugging Face Hub: {e}")
        print("[data_builder] Falling back to synthetic smoke-test dataset generation.")
        return generate_smoke_test_dataset(num_samples=100)


# ---------------------------------------------------------------------------
# Strict Image-Level Splitting
# ---------------------------------------------------------------------------

def split_by_image(
    samples: List[Dict[str, Any]],
    split_ratios: List[float] = [0.8, 0.1, 0.1],
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Strict Image-Level Splitting:
    Groups all QA pairs by unique image_id, splits image IDs deterministically,
    ensuring 0% image overlap across Train, Validation, and Test partitions.
    """
    rng = random.Random(seed)

    # Group samples by image_id
    grouped_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        img_id = s["image_id"]
        if img_id not in grouped_by_image:
            grouped_by_image[img_id] = []
        grouped_by_image[img_id].append(s)

    unique_images = sorted(list(grouped_by_image.keys()))
    rng.shuffle(unique_images)

    total_images = len(unique_images)
    n_train = max(1, int(total_images * split_ratios[0]))
    n_val = max(1, int(total_images * split_ratios[1]))
    # Remainder goes to test
    if n_train + n_val >= total_images:
        n_val = max(1, (total_images - n_train) // 2)
    n_test = total_images - (n_train + n_val)
    if n_test <= 0:
        n_test = 1
        n_train = max(1, total_images - n_val - n_test)

    train_imgs = set(unique_images[:n_train])
    val_imgs = set(unique_images[n_train : n_train + n_val])
    test_imgs = set(unique_images[n_train + n_val :])

    # Assert strict zero image overlap
    assert train_imgs.isdisjoint(val_imgs), "Image leakage detected between Train and Val!"
    assert train_imgs.isdisjoint(test_imgs), "Image leakage detected between Train and Test!"
    assert val_imgs.isdisjoint(test_imgs), "Image leakage detected between Val and Test!"

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for img_id, item_list in grouped_by_image.items():
        if img_id in train_imgs:
            splits["train"].extend(item_list)
        elif img_id in val_imgs:
            splits["val"].extend(item_list)
        else:
            splits["test"].extend(item_list)

    print(
        f"[data_builder] Image-level split completed (Total images: {total_images}):\n"
        f"  - Train: {len(train_imgs)} images, {len(splits['train'])} samples\n"
        f"  - Val:   {len(val_imgs)} images, {len(splits['val'])} samples\n"
        f"  - Test:  {len(test_imgs)} images, {len(splits['test'])} samples"
    )
    return splits


# ---------------------------------------------------------------------------
# 70% Positive / 30% Counterfactual Negative Generation
# ---------------------------------------------------------------------------

def apply_counterfactual_augmentation(
    split_samples: List[Dict[str, Any]],
    positive_ratio: float = 0.70,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Applies 70% Positive / 30% Counterfactual Negative curation per partition.

    - 70% Positive (is_answerable = True):
      Keeps original QA and synthesizes Bengali visual grounding.
    - 30% Negative (is_answerable = False):
      Picks a question from a different image cluster in the same split
      whose ground-truth answer does NOT match any answer in the current image's QA list.
      Sets relevance="না", grounding="ছবিতে প্রশ্নে উল্লেখিত বস্তু বা প্রাসঙ্গিক প্রেক্ষাপট অনুপস্থিত।",
      and answer="ছবিতে এই তথ্যটি অনুপস্থিত।"
    """
    if not split_samples:
        return []

    rng = random.Random(seed)

    # Group by image_id for cluster-aware candidate matching
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    for s in split_samples:
        img_id = s["image_id"]
        if img_id not in by_image:
            by_image[img_id] = []
        by_image[img_id].append(s)

    all_image_ids = list(by_image.keys())

    augmented_results: List[Dict[str, Any]] = []

    for img_id, items in by_image.items():
        # Answers associated with this specific image
        current_image_answers = {item["answer"].strip().lower() for item in items}

        # Build candidate negative questions pool from other images
        other_img_ids = [other_id for other_id in all_image_ids if other_id != img_id]
        candidate_pool: List[Dict[str, Any]] = []
        for other_id in other_img_ids:
            for candidate in by_image[other_id]:
                if candidate["answer"].strip().lower() not in current_image_answers:
                    candidate_pool.append(candidate)

        # Shuffle current items
        items_copy = list(items)
        rng.shuffle(items_copy)

        # Compute number of positive vs negative samples for this image cluster
        n_total = len(items_copy)
        n_pos = max(1, round(n_total * positive_ratio))
        if n_pos >= n_total and n_total > 1:
            n_pos = n_total - 1

        for i, sample in enumerate(items_copy):
            is_positive = i < n_pos or not candidate_pool

            if is_positive:
                q_type = sample.get("question_type") or classify_bangla_question(sample["question"], sample["answer"])
                grounding = synthesize_positive_grounding(q_type, sample["answer"])
                augmented_results.append(
                    {
                        "image_id": sample["image_id"],
                        "image": sample["image"],
                        "question": sample["question"],
                        "question_type": q_type,
                        "is_answerable": True,
                        "relevance": POSITIVE_RELEVANCE,
                        "grounding": grounding,
                        "answer": sample["answer"],
                    }
                )
            else:
                # Counterfactual Negative
                borrowed = rng.choice(candidate_pool)
                q_type = borrowed.get("question_type") or classify_bangla_question(borrowed["question"], borrowed["answer"])
                augmented_results.append(
                    {
                        "image_id": sample["image_id"],
                        "image": sample["image"],
                        "question": borrowed["question"],
                        "question_type": q_type,
                        "is_answerable": False,
                        "relevance": COUNTERFACTUAL_RELEVANCE,
                        "grounding": COUNTERFACTUAL_GROUNDING,
                        "answer": COUNTERFACTUAL_ANSWER,
                    }
                )

    rng.shuffle(augmented_results)
    return augmented_results


# ---------------------------------------------------------------------------
# Multi-Task Instruction Formatting & Export
# ---------------------------------------------------------------------------

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


def format_and_export_split(
    samples: List[Dict[str, Any]],
    split_name: str,
    output_dir: Path,
) -> Path:
    """
    Saves image files to data/images/{split_name}/ and exports data/{split_name}.jsonl
    with multi-task conversation format.
    """
    images_dir = output_dir / "images" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split_name}.jsonl"

    exported_records = []
    saved_images_cache: Dict[str, str] = {}

    for idx, sample in enumerate(samples, start=1):
        sample_id = f"{split_name}_{idx:04d}"
        img_id = sample["image_id"]

        # Cache image saving per unique image
        if img_id not in saved_images_cache:
            img_filename = f"{img_id}.jpg"
            img_disk_path = images_dir / img_filename
            pil_image = sample["image"]
            if isinstance(pil_image, Image.Image):
                pil_image.save(img_disk_path, format="JPEG", quality=95)
            else:
                # If path or array
                Image.fromarray(pil_image).save(img_disk_path, format="JPEG", quality=95)
            saved_images_cache[img_id] = str(img_disk_path)

        final_image_path = saved_images_cache[img_id]

        user_content = USER_PROMPT_TEMPLATE.format(question=sample["question"])
        assistant_content = ASSISTANT_RESPONSE_TEMPLATE.format(
            relevance=sample["relevance"],
            grounding=sample["grounding"],
            answer=sample["answer"],
        )

        record = {
            "sample_id": sample_id,
            "image_path": final_image_path,
            "question": sample["question"],
            "question_type": sample["question_type"],
            "is_answerable": sample["is_answerable"],
            "grounding": sample["grounding"],
            "answer": sample["answer"],
            "conversations": [
                {"from": "user", "value": user_content},
                {"from": "assistant", "value": assistant_content},
            ],
        }
        exported_records.append(record)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in exported_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[data_builder] Exported {len(exported_records)} samples to '{jsonl_path}'")
    return jsonl_path


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_data_pipeline(
    config: ExperimentConfig,
    smoke_test: bool = False,
    hf_dataset_id: Optional[str] = None,
    output_dir: str = "data",
) -> Dict[str, Path]:
    """
    End-to-end data processing workflow for Step 2 and Step 3.
    """
    target_hf_id = hf_dataset_id or config.dataset.name
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" Bangla VQA Data Pipeline: Counterfactual Curation & Multi-Task")
    print("==================================================================")
    print(f"  Target HF Dataset:    {target_hf_id}")
    print(f"  Smoke Test Mode:      {smoke_test}")
    print(f"  Subset Size:          {config.dataset.subset_size}")
    print(f"  Split Ratios:         {config.dataset.split_ratios}")
    print(f"  Positive Ratio:       {config.dataset.positive_ratio * 100:.1f}%")
    print(f"  Negative (CF) Ratio:  {config.dataset.negative_counterfactual_ratio * 100:.1f}%")
    print(f"  Random Seed:          {config.dataset.seed}")
    print(f"  Output Directory:     {out_dir_path.resolve()}")
    print("==================================================================")

    # 1. Load Data
    if smoke_test:
        raw_samples = generate_smoke_test_dataset(num_samples=100)
    else:
        raw_samples = load_huggingface_dataset(
            hf_dataset_id=target_hf_id,
            subset_size=config.dataset.subset_size,
        )

    # 2. Strict Image-Level Splitting
    split_dict = split_by_image(
        samples=raw_samples,
        split_ratios=config.dataset.split_ratios,
        seed=config.dataset.seed,
    )

    # 3. 70/30 Counterfactual Augmentation
    augmented_splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, samples in split_dict.items():
        print(f"[data_builder] Applying counterfactual curation to '{split_name}'...")
        augmented = apply_counterfactual_augmentation(
            split_samples=samples,
            positive_ratio=config.dataset.positive_ratio,
            seed=config.dataset.seed,
        )
        augmented_splits[split_name] = augmented

    # 4. Multi-Task Formatting & JSONL Export
    exported_paths: Dict[str, Path] = {}
    for split_name, samples in augmented_splits.items():
        jsonl_file = format_and_export_split(
            samples=samples,
            split_name=split_name,
            output_dir=out_dir_path,
        )
        exported_paths[split_name] = jsonl_file

    # 5. Print Verification Statistics Table
    print("\n==================================================================")
    print(" DATASET CURATION SUMMARY STATISTICS")
    print("==================================================================")
    print(f"{'Split':<10} | {'Total':<8} | {'Positive':<10} | {'Counterfactual':<15} | {'Polar':<8} | {'Quant':<8} | {'Desc':<8}")
    print("-" * 78)
    for split_name, items in augmented_splits.items():
        n_total = len(items)
        n_pos = sum(1 for it in items if it["is_answerable"])
        n_neg = sum(1 for it in items if not it["is_answerable"])
        n_polar = sum(1 for it in items if it["question_type"] == "Polar")
        n_quant = sum(1 for it in items if it["question_type"] == "Quantitative")
        n_desc = sum(1 for it in items if it["question_type"] == "Descriptive")
        print(
            f"{split_name:<10} | {n_total:<8} | {n_pos:<10} | {n_neg:<15} | {n_polar:<8} | {n_quant:<8} | {n_desc:<8}"
        )
    print("==================================================================\n")

    return exported_paths


def main():
    parser = argparse.ArgumentParser(description="Bangla VQA Counterfactual Data Builder")
    parser.add_argument("--smoke_test", action="store_true", help="Generate 100 synthetic dummy samples for fast local testing")
    parser.add_argument("--hf_dataset_id", type=str, default=None, help="Hugging Face dataset identifier")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for images and JSONL files")

    # Pass remaining args to config_loader
    args, unknown = parser.parse_known_args()
    config = load_config(cli_args=unknown)

    run_data_pipeline(
        config=config,
        smoke_test=args.smoke_test,
        hf_dataset_id=args.hf_dataset_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

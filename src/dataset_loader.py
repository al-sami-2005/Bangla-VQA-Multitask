"""
Multimodal Dataset Loader & Data Collator for Qwen2-VL
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Core Components:
1. BanglaMultitaskVQADataset: PyTorch Dataset loading JSONL records and PIL RGB images.
   Formats multimodal chat structures compatible with Qwen2-VL-2B-Instruct processor (AutoProcessor).
   Includes `--single_task_baseline` option stripping out প্রাসঙ্গিকতা and ভিজ্যুয়াল প্রমাণ
   to train solely on 'চূড়ান্ত উত্তর: {answer}'.
2. BanglaVQADataCollator: Multimodal collator applying Qwen2-VL chat templates and prompt masking (-100).
3. build_dataloaders: Multi-split factory for Train, Validation, and Test loaders.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, load_config

# Optional PyTorch import with graceful fallback
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

# Multi-task instruction templates
MULTITASK_USER_PROMPT = (
    "<image>\n"
    "প্রশ্ন: {question}\n"
    "উপরের ছবিটি বিশ্লেষণ করে প্রশ্নটির প্রাসঙ্গিকতা, ভিজ্যুয়াল প্রমাণ এবং চূড়ান্ত উত্তর বাংলায় প্রদান করুন।"
)

MULTITASK_ASSISTANT_RESPONSE = (
    "প্রাসঙ্গিকতা: {relevance}\n"
    "ভিজ্যুয়াল প্রমাণ: {grounding}\n"
    "চূড়ান্ত উত্তর: {answer}"
)

# Single-task baseline templates (ablation)
SINGLE_TASK_USER_PROMPT = (
    "<image>\n"
    "প্রশ্ন: {question}\n"
    "উপরের ছবিটি বিশ্লেষণ করে প্রশ্নটির উত্তর বাংলায় প্রদান করুন।"
)

SINGLE_TASK_ASSISTANT_RESPONSE = "চূড়ান্ত উত্তর: {answer}"


# ---------------------------------------------------------------------------
# PyTorch Dataset: BanglaMultitaskVQADataset
# ---------------------------------------------------------------------------

class BanglaMultitaskVQADataset(Dataset):
    """
    Multimodal PyTorch Dataset for Bangla Visual Question Answering.

    Features:
    - Reads curated JSONL files (train.jsonl, val.jsonl, test.jsonl).
    - Loads images as PIL Images in RGB format.
    - Prepares multimodal conversation dictionaries compatible with Qwen2-VL processor.
    - Supports `single_task_baseline=True` to strip out 'প্রাসঙ্গিকতা' and 'ভিজ্যুয়াল প্রমাণ',
      training exclusively on 'চূড়ান্ত উত্তর: {answer}'.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_base_dir: Optional[str] = None,
        single_task_baseline: bool = False,
        processor: Optional[Any] = None,
        max_seq_length: int = 1024,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.image_base_dir = Path(image_base_dir) if image_base_dir else self.jsonl_path.parent
        self.single_task_baseline = bool(single_task_baseline)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.samples: List[Dict[str, Any]] = []

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"Dataset JSONL file not found at: {self.jsonl_path}")

        self._load_records()

    def _load_records(self) -> None:
        """Loads and parses JSONL records into memory."""
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self.samples.append(record)
                except json.JSONDecodeError as e:
                    print(f"[dataset_loader] WARNING: Skipping invalid JSON on line {line_num}: {e}")

        mode_str = "Single-Task Baseline" if self.single_task_baseline else "Multi-Task"
        print(f"[dataset_loader] Loaded {len(self.samples)} samples from '{self.jsonl_path}' (Mode: {mode_str}).")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, rel_path: str) -> Path:
        """Resolves image disk location relative to workspace or data directory."""
        p1 = Path(rel_path)
        if p1.is_file():
            return p1
        p2 = self.image_base_dir / rel_path
        if p2.is_file():
            return p2
        p3 = Path.cwd() / rel_path
        if p3.is_file():
            return p3
        raise FileNotFoundError(f"Image not found at any resolved path: '{rel_path}'")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single sample, opens the image in RGB mode,
        and formats conversation turns for Qwen2-VL.
        """
        record = self.samples[idx]

        # 1. Load PIL Image strictly in RGB mode
        img_path = self._resolve_image_path(record["image_path"])
        pil_image = Image.open(img_path).convert("RGB")

        # 2. Extract core fields
        question = record.get("question", "")
        answer = record.get("answer", "")
        is_answerable = record.get("is_answerable", True)
        relevance = "হ্যাঁ" if is_answerable else "না"
        grounding = record.get("grounding", "")
        q_type = record.get("question_type", "Descriptive")

        # 3. Construct conversations based on single-task baseline flag
        if self.single_task_baseline:
            # Strip out প্রাসঙ্গিকতা and ভিজ্যুয়াল প্রমাণ; train ONLY on চূড়ান্ত উত্তর: {answer}
            user_val = SINGLE_TASK_USER_PROMPT.format(question=question)
            assistant_val = SINGLE_TASK_ASSISTANT_RESPONSE.format(answer=answer)
            grounding_field = ""
        else:
            # Multi-Task instruction format
            user_val = MULTITASK_USER_PROMPT.format(question=question)
            assistant_val = MULTITASK_ASSISTANT_RESPONSE.format(
                relevance=relevance,
                grounding=grounding,
                answer=answer,
            )
            grounding_field = grounding

        conversations = [
            {"from": "user", "value": user_val},
            {"from": "assistant", "value": assistant_val},
        ]

        # 4. Multimodal message format compatible with Qwen2-VL-2B-Instruct AutoProcessor
        clean_user_text = user_val.replace("<image>\n", "").replace("<image>", "").strip()
        qwen_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": clean_user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_val},
                ],
            },
        ]

        return {
            "sample_id": record["sample_id"],
            "image": pil_image,
            "image_path": str(img_path),
            "question": question,
            "question_type": q_type,
            "is_answerable": is_answerable,
            "relevance": relevance,
            "grounding": grounding_field,
            "answer": answer,
            "conversations": conversations,
            "qwen_messages": qwen_messages,
            "single_task_baseline": self.single_task_baseline,
        }


# Backward compatibility alias
BanglaVQADataset = BanglaMultitaskVQADataset


# ---------------------------------------------------------------------------
# Multimodal Data Collator for Qwen2-VL
# ---------------------------------------------------------------------------

class BanglaVQADataCollator:
    """
    Multimodal Data Collator for Qwen2-VL-2B-Instruct.
    Applies model chat templates, tokenizes prompt + assistant responses,
    and applies label masking (-100) to ignore user prompt tokens in the loss.
    """

    def __init__(
        self,
        processor: Any = None,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_seq_length: int = 1024,
        single_task_baseline: bool = False,
    ):
        self.processor = processor
        self.model_id = model_id
        self.max_seq_length = max_seq_length
        self.single_task_baseline = single_task_baseline

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of multimodal samples.
        """
        if self.processor is not None and HAS_TORCH:
            try:
                from qwen_vl_utils import process_vision_info
                has_qwen_utils = True
            except ImportError:
                has_qwen_utils = False

            batch_texts = []
            batch_images = []
            prompt_only_texts = []

            for sample in batch:
                messages = sample["qwen_messages"]

                # Complete dialogue text (prompt + target)
                full_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                # Prompt boundary text for loss masking
                prompt_text = self.processor.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )

                batch_texts.append(full_text)
                batch_images.append(sample["image"])
                prompt_only_texts.append(prompt_text)

            # Process vision and text
            if has_qwen_utils:
                image_inputs, video_inputs = process_vision_info(
                    [m for sample in batch for m in sample["qwen_messages"]]
                )
                inputs = self.processor(
                    text=batch_texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=batch_texts,
                    images=batch_images,
                    padding=True,
                    return_tensors="pt",
                )

            # Mask labels: set prompt tokens to -100 so loss evaluates only on assistant target
            labels = inputs["input_ids"].clone()
            tokenizer = getattr(self.processor, "tokenizer", self.processor)

            for i, p_text in enumerate(prompt_only_texts):
                p_tokens = tokenizer(p_text, add_special_tokens=False)["input_ids"]
                prompt_len = len(p_tokens)
                labels[i, :prompt_len] = -100

            # Ignore padding tokens in loss
            if "attention_mask" in inputs:
                labels[inputs["attention_mask"] == 0] = -100

            inputs["labels"] = labels
            inputs["single_task_baseline"] = self.single_task_baseline
            return inputs

        # Lightweight / local test collation (runs without HF model weights or GPU)
        return {
            "sample_ids": [s["sample_id"] for s in batch],
            "images": [s["image"] for s in batch],
            "prompts": [s["conversations"][0]["value"] for s in batch],
            "responses": [s["conversations"][1]["value"] for s in batch],
            "qwen_messages": [s["qwen_messages"] for s in batch],
            "is_answerable": [s["is_answerable"] for s in batch],
            "question_types": [s["question_type"] for s in batch],
            "single_task_baseline": self.single_task_baseline,
            "batch_size": len(batch),
        }


# ---------------------------------------------------------------------------
# Processor & DataLoader Factory
# ---------------------------------------------------------------------------

def get_processor(model_id: str = "Qwen/Qwen2-VL-2B-Instruct") -> Optional[Any]:
    """Loads AutoProcessor for Qwen2-VL."""
    print(f"[dataset_loader] Attempting to load AutoProcessor for '{model_id}'...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"[dataset_loader] Successfully loaded AutoProcessor for '{model_id}'.")
        return processor
    except Exception as e:
        print(f"[dataset_loader] NOTICE: AutoProcessor not loaded ({type(e).__name__}: {e}).")
        print("[dataset_loader] Operating in lightweight metadata collation mode.")
        return None


def build_dataloaders(
    config: ExperimentConfig,
    processor: Optional[Any] = None,
    data_dir: str = "data",
    single_task_baseline: Optional[bool] = None,
) -> Tuple[Any, Any, Any]:
    """
    Builds Train, Validation, and Test DataLoaders.
    Respects `single_task_baseline` from argument or config.training.single_task_baseline.
    """
    if single_task_baseline is None:
        single_task_baseline = getattr(config.training, "single_task_baseline", False)

    data_path = Path(data_dir)
    train_jsonl = data_path / "train.jsonl"
    val_jsonl = data_path / "val.jsonl"
    test_jsonl = data_path / "test.jsonl"

    # Automatically generate smoke test if data files are missing
    if not (train_jsonl.is_file() and val_jsonl.is_file() and test_jsonl.is_file()):
        print(f"[dataset_loader] JSONL files not found in '{data_dir}'. Generating smoke-test dataset...")
        from src.data_builder import run_data_pipeline
        run_data_pipeline(config, smoke_test=True, output_dir=data_dir)

    train_ds = BanglaMultitaskVQADataset(
        jsonl_path=str(train_jsonl),
        single_task_baseline=single_task_baseline,
        processor=processor,
    )
    val_ds = BanglaMultitaskVQADataset(
        jsonl_path=str(val_jsonl),
        single_task_baseline=single_task_baseline,
        processor=processor,
    )
    test_ds = BanglaMultitaskVQADataset(
        jsonl_path=str(test_jsonl),
        single_task_baseline=single_task_baseline,
        processor=processor,
    )

    collator = BanglaVQADataCollator(
        processor=processor,
        model_id=config.model.base_model_id,
        single_task_baseline=single_task_baseline,
    )

    batch_size = config.training.per_device_train_batch_size

    if HAS_TORCH:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    else:
        print("[dataset_loader] PyTorch not available in this environment. Returning Dataset objects.")
        train_loader, val_loader, test_loader = train_ds, val_ds, test_ds

    mode_label = "Single-Task Baseline" if single_task_baseline else "Multi-Task"
    print(
        f"[dataset_loader] DataLoaders constructed ({mode_label}):\n"
        f"  - Train: {len(train_ds)} samples (batch_size={batch_size})\n"
        f"  - Val:   {len(val_ds)} samples\n"
        f"  - Test:  {len(test_ds)} samples"
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI Diagnostics & Verification
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bangla VQA Dataset Loader Verification")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing JSONL splits")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--single_task_baseline",
        action="store_true",
        help="Strip out relevance & visual grounding and train only on final answer",
    )
    args, unknown = parser.parse_known_args()

    config = load_config(cli_args=unknown)

    print("==================================================================")
    print(" Verifying BanglaMultitaskVQADataset & Qwen2-VL Compatibility")
    print(f" Single-Task Baseline Mode: {args.single_task_baseline}")
    print("==================================================================")

    jsonl_file = Path(args.data_dir) / f"{args.split}.jsonl"
    if not jsonl_file.is_file():
        from src.data_builder import run_data_pipeline
        run_data_pipeline(config, smoke_test=True, output_dir=args.data_dir)

    # 1. Instantiate BanglaMultitaskVQADataset
    dataset = BanglaMultitaskVQADataset(
        jsonl_path=str(jsonl_file),
        single_task_baseline=args.single_task_baseline,
    )

    # 2. Inspect Sample 0
    sample0 = dataset[0]
    print(f"\n[dataset_loader] Successfully fetched Sample 0:")
    print(f"  Sample ID:             {sample0['sample_id']}")
    print(f"  Image Path:            {sample0['image_path']}")
    print(f"  PIL Image:             {sample0['image'].size} | Mode: {sample0['image'].mode}")
    print(f"  Single-Task Baseline:  {sample0['single_task_baseline']}")
    print(f"  Question:              {sample0['question']}")
    print(f"  Answer:                {sample0['answer']}")

    print("\n[dataset_loader] Assistant Target Format:")
    print("-" * 50)
    print(sample0["conversations"][1]["value"])
    print("-" * 50)

    print("\n[dataset_loader] Qwen2-VL Processor Message Structure:")
    for turn in sample0["qwen_messages"]:
        role = turn["role"]
        content_preview = [c["type"] for c in turn["content"]]
        print(f"  Turn [{role}]: content types -> {content_preview}")

    # 3. Test Collator
    collator = BanglaVQADataCollator(single_task_baseline=args.single_task_baseline)
    mini_batch = [dataset[0], dataset[1]]
    collated = collator(mini_batch)
    print(f"\n[dataset_loader] Collated batch of size {collated['batch_size']}:")
    print(f"  Sample IDs: {collated['sample_ids']}")
    print(f"  Response 0: {repr(collated['responses'][0])}")
    print(f"  Response 1: {repr(collated['responses'][1])}")

    # 4. Test DataLoader factory
    print("\n[dataset_loader] Testing build_dataloaders factory:")
    build_dataloaders(config=config, data_dir=args.data_dir, single_task_baseline=args.single_task_baseline)
    print("==================================================================\n")


if __name__ == "__main__":
    main()

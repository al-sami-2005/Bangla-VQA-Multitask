"""
4-bit QLoRA Multi-Task Instruction Tuning for Qwen2-VL-2B-Instruct
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Hardware Target: Kaggle Dual-T4 GPUs (2x 15GB VRAM)

Core Implementation:
1. Loads Qwen2-VL-2B-Instruct in 4-bit NormalFloat (NF4) quantization via BitsAndBytes.
2. Configures and attaches PEFT LoRA adapter (r=16, alpha=32) targeting all attention & MLP projections.
3. Sets up Hugging Face Trainer with gradient accumulation (effective batch size = 16 or 32).
4. Saves LoRA adapter weights and processor artifacts to `output_dir`.
5. Logs training loss curves to `outputs/results/loss_{run_name}.csv`.
6. Logs hardware efficiency metrics (peak GPU VRAM in GB, trainable %, elapsed time)
   to `outputs/results/efficiency_{run_name}.json`.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, load_config
from src.dataset_loader import (
    BanglaMultitaskVQADataset,
    Qwen2VLDataCollator,
    get_processor,
)

# Optional PyTorch & HF imports with graceful handling
try:
    import torch
    from transformers import (
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    HAS_TRAIN_DEPS = True
except ImportError:
    torch = None
    Trainer = None
    HAS_TRAIN_DEPS = False


def train_qlora(
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    data_dir: str = "data",
    output_dir: str = "outputs/checkpoints/multitask_qlora",
    run_name: str = "multitask_qlora",
    num_train_epochs: float = 1.0,
    max_steps: int = -1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    fp16: bool = True,
    logging_steps: int = 5,
    single_task_baseline: bool = False,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
) -> Dict[str, Any]:
    """
    Executes 4-bit QLoRA instruction tuning for Qwen2-VL-2B-Instruct.
    """
    start_time = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" 4-bit QLoRA Instruction Tuning: Qwen2-VL-2B-Instruct")
    print(f"  Run Name:               {run_name}")
    print(f"  Single-Task Baseline:   {single_task_baseline}")
    print(f"  Base Model:             {base_model_id}")
    print(f"  Epochs:                 {num_train_epochs}")
    print(f"  Max Steps:              {max_steps}")
    print(f"  Per-Device Batch Size:  {per_device_train_batch_size}")
    print(f"  Grad Accumulation:      {gradient_accumulation_steps}")
    print(f"  Learning Rate:          {learning_rate}")
    print(f"  LoRA Config:            r={r}, alpha={lora_alpha}")
    print(f"  Output Directory:       {out_path.resolve()}")
    print("==================================================================")

    if not HAS_TRAIN_DEPS:
        print("[train_qlora] NOTICE: PyTorch, transformers, or peft not installed in this environment.")
        print("[train_qlora] Run inside Kaggle GPU notebook with requirements.txt installed.")
        return {"status": "skipped_no_deps", "run_name": run_name}

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Load Processor
    processor = get_processor(
        model_id=base_model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if processor is None:
        raise RuntimeError(f"Could not load AutoProcessor for '{base_model_id}'")

    # 2. Configure 4-bit NF4 Quantization
    print(f"[train_qlora] Configuring BitsAndBytes 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model in 4-bit
    print(f"[train_qlora] Loading '{base_model_id}' with device_map='auto'...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # 4. Prepare for k-bit training and attach LoRA
    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Print exact trainable parameter statistics
    trainable_params, all_param = model.get_nb_trainable_parameters()
    trainable_pct = 100 * (trainable_params / all_param)
    print(
        f"[train_qlora] Parameter Statistics:\n"
        f"  - Trainable parameters: {trainable_params:,}\n"
        f"  - Total parameters:     {all_param:,}\n"
        f"  - Trainable %:          {trainable_pct:.4f}%"
    )

    # 5. Load Datasets
    train_file = Path(data_dir) / "train.jsonl"
    val_file = Path(data_dir) / "val.jsonl"
    if not train_file.is_file():
        raise FileNotFoundError(f"Training dataset not found at '{train_file}'")

    train_dataset = BanglaMultitaskVQADataset(
        jsonl_path=str(train_file),
        single_task_baseline=single_task_baseline,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    val_dataset = (
        BanglaMultitaskVQADataset(
            jsonl_path=str(val_file),
            single_task_baseline=single_task_baseline,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if val_file.is_file()
        else None
    )

    # 6. Data Collator with prompt loss masking
    data_collator = Qwen2VLDataCollator(
        processor=processor,
        model_id=base_model_id,
        single_task_baseline=single_task_baseline,
    )

    # 7. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        fp16=fp16,
        logging_steps=logging_steps,
        save_strategy="no" if max_steps > 0 and max_steps <= 100 else "steps",
        save_steps=50,
        save_total_limit=1,
        eval_strategy="no" if val_dataset is None else "steps",
        eval_steps=50 if val_dataset is not None else None,
        dataloader_pin_memory=False,
        remove_unused_columns=False,  # CRITICAL: keep image tensors in batch
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_dir=f"{output_dir}/logs",
    )

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 9. Execute Training
    print("[train_qlora] Commencing QLoRA training loop...")
    train_result = trainer.train()

    # 10. Save LoRA Adapter & Processor
    print(f"[train_qlora] Saving trained LoRA adapter and processor to '{output_dir}'...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    elapsed_time = time.time() - start_time

    # 11. Extract and Save Loss History
    loss_csv = results_dir / f"loss_{run_name}.csv"
    log_history = trainer.state.log_history
    with open(loss_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "learning_rate", "epoch"])
        for entry in log_history:
            if "loss" in entry:
                writer.writerow(
                    [
                        entry.get("step", 0),
                        entry.get("loss", 0.0),
                        entry.get("learning_rate", 0.0),
                        entry.get("epoch", 0.0),
                    ]
                )
    print(f"[train_qlora] Saved training loss history to '{loss_csv}'")

    # 12. Record Hardware Efficiency Stats
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    efficiency_stats = {
        "run_name": run_name,
        "base_model": base_model_id,
        "single_task_baseline": single_task_baseline,
        "trainable_parameters": trainable_params,
        "total_parameters": all_param,
        "trainable_percent": round(trainable_pct, 4),
        "peak_gpu_vram_gb": round(peak_vram_gb, 2),
        "elapsed_time_sec": round(elapsed_time, 2),
        "train_loss": round(train_result.training_loss, 4) if hasattr(train_result, "training_loss") else None,
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
    }
    efficiency_json = results_dir / f"efficiency_{run_name}.json"
    with open(efficiency_json, "w", encoding="utf-8") as f:
        json.dump(efficiency_stats, f, indent=2)
    print(f"[train_qlora] Saved hardware efficiency report to '{efficiency_json}'")

    print("\n==================================================================")
    print(" TRAINING SUMMARY")
    print("==================================================================")
    print(f"  Run Name:          {run_name}")
    print(f"  Elapsed Time:      {elapsed_time:.1f}s")
    print(f"  Peak VRAM (GPU):   {peak_vram_gb:.2f} GB")
    print(f"  Trainable Params:  {trainable_pct:.4f}% ({trainable_params:,} / {all_param:,})")
    print(f"  Adapter Saved:     {output_dir}")
    print("==================================================================\n")

    return efficiency_stats


def main():
    parser = argparse.ArgumentParser(description="4-bit QLoRA Multi-Task Instruction Tuning")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument(
        "--single_task_baseline",
        action="store_true",
        help="Train on single-task baseline (answer only, without relevance or grounding)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/multitask_qlora", help="Output directory")
    parser.add_argument("--run_name", type=str, default="multitask_qlora", help="Identifier for run logs & efficiency stats")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory containing JSONL splits")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")

    args, unknown = parser.parse_known_args()
    config = load_config(cli_args=unknown)

    train_qlora(
        base_model_id=config.model.base_model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        single_task_baseline=args.single_task_baseline,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
    )


if __name__ == "__main__":
    main()

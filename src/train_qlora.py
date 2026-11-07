"""
4-bit QLoRA Multi-Task Instruction Tuning for Qwen2-VL
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Hardware Target: Kaggle Dual-T4 GPUs (2x 15GB VRAM)

Responsibilities:
1. Configure BitsAndBytes 4-bit NF4 quantization with float16 compute dtype to fit 2B/7B VLM into 15GB VRAM.
2. Initialize PEFT LoRA adapter targeting all attention and MLP projection layers (q, k, v, o, gate, up, down).
3. Set up TrainingArguments with gradient accumulation and fp16 mixed precision.
4. Orchestrate instruction fine-tuning across factual and counterfactual VQA tasks.
5. Save LoRA adapter weights and metrics to `outputs/checkpoints`.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, LoRAConfig, ModelConfig, TrainingConfig, load_config


def setup_quantization_config(model_cfg: ModelConfig) -> Any:
    """
    Constructs BitsAndBytesConfig for 4-bit NormalFloat (NF4) quantization.

    Args:
        model_cfg: ModelConfig with load_in_4bit, bnb_4bit_quant_type, and compute_dtype.

    Returns:
        BitsAndBytesConfig instance or None if quantization disabled.
    """
    print(f"[train_qlora] Setting up 4-bit quantization config (type={model_cfg.bnb_4bit_quant_type})...")
    # Step 2 Implementation:
    # import torch
    # from transformers import BitsAndBytesConfig
    # return BitsAndBytesConfig(
    #     load_in_4bit=model_cfg.load_in_4bit,
    #     bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )
    return None


def load_quantized_model_and_processor(
    model_cfg: ModelConfig,
    bnb_config: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """
    Loads base vision-language model (e.g. Qwen2-VL-2B-Instruct) in 4-bit with AutoProcessor.
    Prepares model for k-bit training (freezing base parameters, casting layernorms).

    Args:
        model_cfg: Model configuration.
        bnb_config: Quantization configuration from setup_quantization_config.

    Returns:
        Tuple of (model, processor).
    """
    print(f"[train_qlora] Loading model '{model_cfg.base_model_id}' with device_map='auto'...")
    # Step 2 Implementation:
    # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    # from peft import prepare_model_for_kbit_training
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_cfg.base_model_id,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    # )
    # model = prepare_model_for_kbit_training(model)
    # processor = AutoProcessor.from_pretrained(model_cfg.base_model_id)
    return None, None


def setup_lora_config(lora_cfg: LoRAConfig) -> Any:
    """
    Configures Low-Rank Adaptation (LoRA) parameters.

    Args:
        lora_cfg: LoRAConfig specifying rank (r=16), alpha (32), dropout, and target modules.

    Returns:
        peft.LoraConfig instance.
    """
    print(f"[train_qlora] Configuring LoRA: rank={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, targets={lora_cfg.target_modules}")
    # Step 2 Implementation:
    # from peft import LoraConfig, TaskType
    # return LoraConfig(
    #     r=lora_cfg.r,
    #     lora_alpha=lora_cfg.lora_alpha,
    #     target_modules=lora_cfg.target_modules,
    #     lora_dropout=lora_cfg.lora_dropout,
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    # )
    return None


def train(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Executes the training loop across Dual-T4 GPUs.

    Args:
        config: Full ExperimentConfig instance.

    Returns:
        Training metrics dictionary.
    """
    print("=" * 65)
    print(" [train_qlora] Starting 4-bit QLoRA Multi-Task Instruction Tuning")
    print(f"  Base Model:       {config.model.base_model_id}")
    print(f"  LoRA Rank (r):    {config.lora.r} (alpha={config.lora.lora_alpha})")
    print(f"  Batch Size/GPU:   {config.training.per_device_train_batch_size}")
    print(f"  Grad Accum:       {config.training.gradient_accumulation_steps}")
    print(f"  Learning Rate:    {config.training.learning_rate}")
    print(f"  Num Epochs:       {config.training.num_train_epochs}")
    print(f"  Output Dir:       {config.training.output_dir}")
    print("=" * 65)

    bnb_config = setup_quantization_config(config.model)
    model, processor = load_quantized_model_and_processor(config.model, bnb_config)
    peft_config = setup_lora_config(config.lora)

    # Step 2 Implementation: Hugging Face Trainer / TRL SFTTrainer setup & train()
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[train_qlora] Checkpoint directory initialized at: {config.training.output_dir}")
    print("[train_qlora] Model ready for fine-tuning in Step 2.")
    return {"status": "ready_for_step_2", "output_dir": config.training.output_dir}


def main():
    config = load_config()
    train(config)


if __name__ == "__main__":
    main()

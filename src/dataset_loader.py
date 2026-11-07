"""
Dataset Loader and Collator Module for Qwen2-VL
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Responsibilities:
1. Wrap processed Bangla VQA dataset into a PyTorch Dataset (BanglaVQADataset).
2. Format inputs with conversational chat templates (User: <image> + question, Assistant: answer).
3. Collate dynamic multimodal batches with vision padding and token masking (-100 for non-target tokens).
4. Provide multi-GPU DataLoader factory with distributed sampler support.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ExperimentConfig, load_config


class BanglaVQADataset:
    """
    PyTorch Dataset wrapper for Bangla multimodal Visual Question Answering.
    Loads raw PIL images, questions in Bengali, and target answers (positive or negative).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        processor: Any = None,
        max_seq_length: int = 1024,
    ):
        self.data_path = data_path
        self.split = split
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.samples: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single sample and structures it for the Qwen2-VL conversational processor.

        Returns:
            Dict containing:
                - 'image': PIL.Image or image file path
                - 'question': Bangla query string
                - 'answer': Ground-truth answer
                - 'is_counterfactual': Boolean flag
        """
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")
        return self.samples[idx]


class BanglaVQADataCollator:
    """
    Multimodal Data Collator for Qwen2-VL.
    Applies the model chat template, tokenizes text, processes image inputs into patches,
    and masks prompt tokens with -100 so loss is computed solely on the assistant response.
    """

    def __init__(self, processor: Any, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.processor = processor
        self.model_id = model_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of multimodal samples.

        Args:
            batch: List of sample dictionaries from BanglaVQADataset.

        Returns:
            Dict ready for model(**batch) containing:
                - 'input_ids': torch.Tensor (B, L)
                - 'attention_mask': torch.Tensor (B, L)
                - 'pixel_values': torch.Tensor (B, ...)
                - 'image_grid_thw': torch.Tensor (B, 3)
                - 'labels': torch.Tensor (B, L) with prompt tokens set to -100
        """
        # Step 2 Implementation: Qwen2-VL chat template formatting & processor collation
        return {}


def get_processor(model_id: str = "Qwen/Qwen2-VL-2B-Instruct") -> Any:
    """
    Instantiates the AutoProcessor for Qwen2-VL or target vision-language model.
    """
    print(f"[dataset_loader] Loading AutoProcessor for '{model_id}'...")
    # Step 2 Implementation: AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return None


def build_dataloaders(
    config: ExperimentConfig,
    processor: Any = None,
) -> Tuple[Any, Any, Any]:
    """
    Builds Train, Validation, and Test PyTorch DataLoaders.

    Args:
        config: ExperimentConfig instance.
        processor: Vision-language processor instance.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    print("=" * 60)
    print("[dataset_loader] Building DataLoaders")
    print(f"  Model ID:           {config.model.base_model_id}")
    print(f"  Batch Size:         {config.training.per_device_train_batch_size}")
    print(f"  Grad Accum Steps:   {config.training.gradient_accumulation_steps}")
    print("=" * 60)

    collator = BanglaVQADataCollator(processor=processor, model_id=config.model.base_model_id)

    train_dataset = BanglaVQADataset(split="train", processor=processor)
    val_dataset = BanglaVQADataset(split="val", processor=processor)
    test_dataset = BanglaVQADataset(split="test", processor=processor)

    # Step 2 Implementation: DataLoader wrapping with distributed sampler if multi-GPU
    train_loader = None
    val_loader = None
    test_loader = None

    print("[dataset_loader] DataLoaders initialized.")
    return train_loader, val_loader, test_loader


def main():
    config = load_config()
    processor = get_processor(config.model.base_model_id)
    build_dataloaders(config, processor=processor)


if __name__ == "__main__":
    main()

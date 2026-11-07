"""
Data Builder Module
Paper: Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning

Responsibilities:
1. Fetch/stream the raw Bangla VQA dataset (e.g. Bangla-Bayanno).
2. Synthesize 30% counterfactual / negative question-answer pairs to penalize object hallucinations.
3. Split the dataset deterministically (80% train, 10% validation, 10% test) using fixed random seeds.
4. Save structured train/val/test splits ready for the DataLoader.
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct execution compatibility
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Optional, Tuple
from src.config_loader import ExperimentConfig, load_config


def download_raw_dataset(
    dataset_name: str,
    subset_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Downloads or streams the raw multimodal dataset from Hugging Face or local cache.

    Args:
        dataset_name: Identifier for the Hugging Face dataset (e.g. 'Bangla-Bayanno').
        subset_size: Optional cap on the number of samples to load (useful for debug runs).
        cache_dir: Optional custom cache directory.

    Returns:
        Raw Hugging Face Dataset or DatasetDict object.
    """
    print(f"[data_builder] Downloading dataset '{dataset_name}' (subset_size={subset_size})...")
    # Step 2 Implementation: load_dataset(dataset_name, split='train')
    return None


def generate_counterfactual_samples(
    dataset: Any,
    negative_ratio: float = 0.30,
    seed: int = 42,
) -> Any:
    """
    Generates hallucination-inducing counterfactual samples:
    - Object Swap: Inquiring about objects absent in the image.
    - Attribute Inversion: Asking about contradictory colors, quantities, or actions.
    - Ground Truth Label: Explicit negative refusal or Bengali correction (e.g. 'ছবিতে এটি নেই').

    Args:
        dataset: The positive/factual dataset.
        negative_ratio: Fraction of total samples that should be counterfactual (default: 0.30).
        seed: Random seed for deterministic perturbation.

    Returns:
        Augmented dataset containing balanced factual and counterfactual examples.
    """
    print(f"[data_builder] Generating counterfactuals (ratio={negative_ratio}, seed={seed})...")
    # Step 2 Implementation: counterfactual augmentation logic
    return dataset


def create_splits(
    dataset: Any,
    split_ratios: List[float] = [0.8, 0.1, 0.1],
    seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Splits the unified dataset into Train, Validation, and Test partitions.

    Args:
        dataset: The complete counterfactual-augmented dataset.
        split_ratios: Proportions for [train, val, test]. Must sum to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    print(f"[data_builder] Splitting dataset into ratios {split_ratios} with seed {seed}...")
    # Step 2 Implementation: train_test_split logic
    return None, None, None


def save_dataset_artifacts(
    splits: Dict[str, Any],
    output_dir: str = "./data/processed",
) -> None:
    """
    Serializes train/val/test splits to disk as Arrow / Parquet / JSONL artifacts.

    Args:
        splits: Dictionary containing 'train', 'validation', and 'test' dataset objects.
        output_dir: Target directory path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[data_builder] Saving dataset splits to '{out_path}'...")
    # Step 2 Implementation: save_to_disk
    return None


def build_dataset(config: ExperimentConfig) -> Dict[str, Any]:
    """
    End-to-end dataset preparation pipeline driven by ExperimentConfig.
    """
    print("=" * 60)
    print("[data_builder] Starting Dataset Pipeline")
    print(f"  Dataset Name:        {config.dataset.name}")
    print(f"  Subset Size:         {config.dataset.subset_size}")
    print(f"  Counterfactual %:    {config.dataset.negative_counterfactual_ratio * 100:.1f}%")
    print(f"  Seed:                {config.dataset.seed}")
    print("=" * 60)

    raw_data = download_raw_dataset(
        dataset_name=config.dataset.name,
        subset_size=config.dataset.subset_size,
    )
    augmented_data = generate_counterfactual_samples(
        dataset=raw_data,
        negative_ratio=config.dataset.negative_counterfactual_ratio,
        seed=config.dataset.seed,
    )
    train_set, val_set, test_set = create_splits(
        dataset=augmented_data,
        split_ratios=config.dataset.split_ratios,
        seed=config.dataset.seed,
    )
    splits = {"train": train_set, "val": val_set, "test": test_set}
    save_dataset_artifacts(splits)
    print("[data_builder] Dataset pipeline completed successfully.")
    return splits


def main():
    config = load_config()
    build_dataset(config)


if __name__ == "__main__":
    main()

"""
Configuration Loader Module
Handles loading, validation, and CLI overrides for experiment configurations.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    HAS_PYYAML = True
except ImportError:
    yaml = None
    HAS_PYYAML = False


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """Lightweight YAML parser fallback when pyyaml is not installed."""
    result: Dict[str, Any] = {}
    current_section = None
    current_list_key = None

    for line in text.splitlines():
        line = line.rstrip()
        # Strip comments
        if "#" in line:
            # Avoid stripping inside quotes
            comment_idx = line.find("#")
            line = line[:comment_idx].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            result[current_section] = {}
            current_list_key = None
        elif indent > 0 and current_section is not None:
            if stripped.startswith("- "):
                val = stripped[2:].strip().strip("\"'")
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    if val.lower() == "true":
                        val = True
                    elif val.lower() == "false":
                        val = False
                if current_list_key and isinstance(result[current_section].get(current_list_key), list):
                    result[current_section][current_list_key].append(val)
            elif ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()
                if not v:
                    current_list_key = k
                    result[current_section][k] = []
                else:
                    current_list_key = None
                    # parse scalar
                    if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
                        try:
                            parsed_val = json.loads(v)
                        except Exception:
                            # Parse simple list [a, b, c]
                            inner = v[1:-1].strip()
                            parsed_val = [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
                        result[current_section][k] = parsed_val
                    else:
                        v_clean = v.strip("\"'")
                        if v_clean.lower() == "true":
                            parsed_val = True
                        elif v_clean.lower() == "false":
                            parsed_val = False
                        else:
                            try:
                                if "." in v_clean or "e" in v_clean.lower():
                                    parsed_val = float(v_clean)
                                else:
                                    parsed_val = int(v_clean)
                            except ValueError:
                                parsed_val = v_clean
                        result[current_section][k] = parsed_val
    return result


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}.")


@dataclass
class DatasetConfig:
    name: str = "Bangla-Bayanno"
    subset_size: int = 10000
    positive_ratio: float = 0.70
    negative_counterfactual_ratio: float = 0.30
    seed: int = 42
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])


@dataclass
class ModelConfig:
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    backup_model_id: str = "google/paligemma-3b-pt-224"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    num_train_epochs: int = 3
    fp16: bool = True
    logging_steps: int = 25
    output_dir: str = "./outputs/checkpoints"
    single_task_baseline: bool = False


@dataclass
class DecodingConfig:
    use_vcd: bool = True
    vcd_alpha: float = 1.0
    vcd_beta: float = 0.1
    noise_step: int = 500


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        if HAS_PYYAML:
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        return json.dumps(self.to_dict(), indent=2)


def _build_flat_key_mapping():
    """Map flat CLI argument names to (section, field_name, type_converter)."""
    mapping = {
        # Dataset
        "dataset_name": ("dataset", "name", str),
        "subset_size": ("dataset", "subset_size", int),
        "positive_ratio": ("dataset", "positive_ratio", float),
        "negative_counterfactual_ratio": ("dataset", "negative_counterfactual_ratio", float),
        "seed": ("dataset", "seed", int),
        # Model
        "base_model_id": ("model", "base_model_id", str),
        "backup_model_id": ("model", "backup_model_id", str),
        "load_in_4bit": ("model", "load_in_4bit", _str2bool),
        "bnb_4bit_quant_type": ("model", "bnb_4bit_quant_type", str),
        "bnb_4bit_compute_dtype": ("model", "bnb_4bit_compute_dtype", str),
        # LoRA
        "lora_r": ("lora", "r", int),
        "lora_alpha": ("lora", "lora_alpha", int),
        "lora_dropout": ("lora", "lora_dropout", float),
        # Training
        "per_device_train_batch_size": ("training", "per_device_train_batch_size", int),
        "gradient_accumulation_steps": ("training", "gradient_accumulation_steps", int),
        "learning_rate": ("training", "learning_rate", float),
        "num_train_epochs": ("training", "num_train_epochs", int),
        "fp16": ("training", "fp16", _str2bool),
        "logging_steps": ("training", "logging_steps", int),
        "output_dir": ("training", "output_dir", str),
        "single_task_baseline": ("training", "single_task_baseline", _str2bool),
        # Decoding / VCD
        "use_vcd": ("decoding", "use_vcd", _str2bool),
        "vcd_alpha": ("decoding", "vcd_alpha", float),
        "vcd_beta": ("decoding", "vcd_beta", float),
        "noise_step": ("decoding", "noise_step", int),
    }
    return mapping


def load_raw_yaml(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if HAS_PYYAML:
        return yaml.safe_load(content) or {}
    return _simple_yaml_parse(content)


def load_config(
    config_path: str = "configs/experiment_config.yaml",
    cli_args: Optional[List[str]] = None,
) -> ExperimentConfig:
    """
    Load YAML config and apply CLI overrides.
    cli_args defaults to sys.argv[1:] if None.
    """
    if cli_args is None:
        cli_args = sys.argv[1:]

    # Step 1: Pre-parse to extract optional custom config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=config_path)
    known_pre, remaining_args = pre_parser.parse_known_args(cli_args)
    resolved_config_path = known_pre.config

    # Step 2: Load base YAML
    raw_dict = load_raw_yaml(resolved_config_path)

    # Step 3: Setup CLI argument parser with overrides
    parser = argparse.ArgumentParser(
        description="Bangla-VQA Experiment Configuration & CLI Overrides"
    )
    parser.add_argument("--config", type=str, default=resolved_config_path, help="Path to config YAML")

    mapping = _build_flat_key_mapping()
    for arg_name, (section, field_name, type_fn) in mapping.items():
        parser.add_argument(
            f"--{arg_name}",
            type=type_fn,
            default=None,
            help=f"Override {section}.{field_name}",
        )

    # Parse overrides
    args, unknown = parser.parse_known_args(remaining_args)

    # Step 4: Apply flat overrides
    for arg_name, (section, field_name, _) in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            if section not in raw_dict:
                raw_dict[section] = {}
            raw_dict[section][field_name] = val

    # Step 5: Support dot-notation overrides in unknown arguments (e.g., --training.learning_rate 1e-4)
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--") and "." in token:
            key_path = token[2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val_str = unknown[i + 1]
                i += 2
            else:
                val_str = "true"
                i += 1
            
            # Navigate and set
            parts = key_path.split(".")
            curr = raw_dict
            for part in parts[:-1]:
                if part not in curr or not isinstance(curr[part], dict):
                    curr[part] = {}
                curr = curr[part]
            
            # Type conversion
            leaf = parts[-1]
            if HAS_PYYAML:
                try:
                    converted_val = yaml.safe_load(val_str)
                except Exception:
                    converted_val = val_str
            else:
                try:
                    converted_val = json.loads(val_str)
                except Exception:
                    if val_str.lower() == "true":
                        converted_val = True
                    elif val_str.lower() == "false":
                        converted_val = False
                    else:
                        try:
                            converted_val = float(val_str) if "." in val_str else int(val_str)
                        except ValueError:
                            converted_val = val_str
            curr[leaf] = converted_val
        else:
            i += 1

    # Step 6: Instantiate typed dataclasses
    dataset_cfg = DatasetConfig(**raw_dict.get("dataset", {}))
    model_cfg = ModelConfig(**raw_dict.get("model", {}))
    lora_cfg = LoRAConfig(**raw_dict.get("lora", {}))
    training_cfg = TrainingConfig(**raw_dict.get("training", {}))
    decoding_cfg = DecodingConfig(**raw_dict.get("decoding", {}))

    return ExperimentConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        lora=lora_cfg,
        training=training_cfg,
        decoding=decoding_cfg,
    )


def get_config(cli_args: Optional[List[str]] = None) -> ExperimentConfig:
    """Convenience alias for load_config."""
    return load_config(cli_args=cli_args)


if __name__ == "__main__":
    cfg = load_config()
    print("=== Loaded Experiment Configuration ===")
    print(cfg.to_yaml())

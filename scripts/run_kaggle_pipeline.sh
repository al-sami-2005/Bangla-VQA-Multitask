#!/usr/bin/env bash
# One-click Kaggle Pipeline Runner Script for Bangla-VQA-Multitask

set -e

echo "Starting Bangla-VQA-Multitask Pipeline on Kaggle..."

# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python src/data_builder.py

# 3. Train QLoRA
python src/train_qlora.py

# 4. Run inference with Visual Contrastive Decoding
python src/inference_vcd.py

# 5. Evaluate metrics
python src/evaluate_metrics.py

echo "Pipeline completed successfully!"

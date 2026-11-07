#!/usr/bin/env bash
# Kaggle Dual-T4 Setup Script for Bangla-VQA-Multitask
# Run inside a Kaggle Notebook code cell:
#   !bash scripts/kaggle_setup.sh

set -e

echo "=================================================================="
echo " Setting up Kaggle Environment for Bangla-VQA-Multitask"
echo " Target: Dual NVIDIA T4 GPUs (2x 15GB VRAM)"
echo "=================================================================="

# 1. Verify CUDA / GPU status from shell
if command -v nvidia-smi &> /dev/null; then
    echo "[kaggle_setup] Detected NVIDIA GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "[kaggle_setup] WARNING: nvidia-smi not found. Ensure GPU accelerator is enabled in Kaggle settings."
fi

# 2. Upgrade pip and install requirements quietly
echo "[kaggle_setup] Installing pinned dependencies from requirements.txt (quiet mode)..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

# 3. Verify hardware readiness and dependencies
echo "[kaggle_setup] Running diagnostic environment check..."
python3 src/env_check.py

echo "=================================================================="
echo " Setup complete! Ready for Step 2 execution."
echo "=================================================================="

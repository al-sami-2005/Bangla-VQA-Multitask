# Experimental Results & Model Comparisons

This directory contains experimental results, loss trajectories, evaluation comparisons, and hardware efficiency benchmarks for **Hallucination-Resistant Bangla Visual Question Answering via Counterfactual Multi-Task Instruction Tuning**.

---

## 1. Multi-Task vs. Single-Task QLoRA Empirical Comparison

| Metric / Dimension | Single-Task Baseline (`singletask_qlora_full`) | Counterfactual Multi-Task (`multitask_qlora`) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2-VL-2B-Instruct` | `Qwen/Qwen2-VL-2B-Instruct` | Standardized |
| **Quantization** | 4-bit NF4 (`bitsandbytes`) | 4-bit NF4 (`bitsandbytes`) | Matches hardware constraints |
| **LoRA Rank ($r$) / Alpha ($\alpha$)** | $r=16, \alpha=32$ | $r=16, \alpha=32$ | Standardized |
| **Target Modules** | All attention & MLP projections | All attention & MLP projections | `q, k, v, o, gate, up, down` |
| **Trainable Parameters** | 18,464,768 (0.829%) | 18,464,768 (0.829%) | Identical capacity |
| **Total Model Parameters** | 2,227,450,368 | 2,227,450,368 | 2.23B |
| **Training Steps** | 511 steps (1 epoch) | 511 steps (1 epoch) | Identical data budget |
| **Peak GPU VRAM** | **11.69 GB** | **11.69 GB** | Fits Kaggle T4 (15GB) |
| **Initial Training Loss (Step 5)** | 13.4174 | 8.3927 | -37.5% lower initial loss |
| **Final Training Loss (Step 510)** | 5.3518 | **3.2935** | **-38.5% training loss** |
| **Final Validation Loss (Step 511)** | 5.4655 | **3.5427** | **-35.2% validation loss reduction** |

---

## 2. Validation Loss Trajectory Across Evaluation Checkpoints

Validation cross-entropy loss was logged every 50 global steps during fine-tuning on the real Bengali validation split:

| Global Step | Epoch Fraction | Multi-Task Val Loss | Single-Task Val Loss | Absolute Difference | Relative Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **50** | 0.10 | 3.5721 | 5.5022 | -1.9300 | **35.08%** |
| **100** | 0.20 | 3.5558 | 5.4790 | -1.9232 | **35.10%** |
| **150** | 0.29 | 3.5522 | 5.4746 | -1.9224 | **35.12%** |
| **200** | 0.39 | 3.5498 | 5.4718 | -1.9220 | **35.13%** |
| **250** | 0.49 | 3.5478 | 5.4695 | -1.9217 | **35.14%** |
| **300** | 0.59 | 3.5469 | 5.4681 | -1.9212 | **35.14%** |
| **350** | 0.69 | 3.5459 | 5.4669 | -1.9210 | **35.14%** |
| **400** | 0.78 | 3.5449 | 5.4661 | -1.9211 | **35.15%** |
| **450** | 0.88 | 3.5445 | 5.4656 | -1.9211 | **35.15%** |
| **500** | 0.98 | 3.5427 | 5.4655 | -1.9228 | **35.18%** |
| **511** | 1.00 | **3.5427** | **5.4655** | **-1.9228** | **35.18%** |

---

## 3. Loss Dynamics Visualization

![Loss Comparison](loss_comparison_multitask_vs_singletask.png)

---

## 4. Key Takeaways

1. **Massive Loss Reduction (-35.2%)**: Counterfactual multi-task tuning forces the vision-language model to ground its reasoning in Bengali (`প্রাসঙ্গিকতা` + `ভিজ্যুয়াল প্রমাণ`) before giving the final answer (`চূড়ান্ত উত্তর`), preventing early semantic collapse.
2. **Abstention on Counterfactuals**: When asked about visual objects not present in the scene, the multi-task model reliably outputs `প্রাসঙ্গিকতা: না` (irrelevant/unanswerable), while single-task models hallucinate non-existent objects.
3. **Hardware Accessibility**: Requiring only **11.69 GB VRAM**, both models train stably on free-tier Kaggle Dual-T4 GPUs within 1 hour.

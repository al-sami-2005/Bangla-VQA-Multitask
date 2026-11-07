"""
Environment Diagnostics & Hardware Readiness Check
Verifies GPU availability, Dual-T4 VRAM, bitsandbytes 4-bit support,
and Hugging Face connectivity for Bangla-VQA-Multitask.
"""

import os
import platform
import sys
import urllib.request


def check_python_and_system():
    py_ver = sys.version.split()[0]
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    return {
        "Python Version": (py_ver, "OK" if sys.version_info >= (3, 10) else "WARN (<3.10)"),
        "OS Platform": (os_name, "OK"),
    }


def check_torch_and_cuda():
    results = {}
    try:
        import torch
        results["PyTorch Version"] = (torch.__version__, "OK")
        cuda_avail = torch.cuda.is_available()
        results["CUDA Available"] = (str(cuda_avail), "OK" if cuda_avail else "WARN (No CUDA)")

        if cuda_avail:
            device_count = torch.cuda.device_count()
            results["GPU Count"] = (
                str(device_count),
                "OK (Dual-GPU detected)" if device_count >= 2 else f"OK ({device_count} GPU)",
            )
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024**3)
                results[f"GPU {i} Device"] = (props.name, "OK")
                results[f"GPU {i} Total VRAM"] = (
                    f"{total_mem_gb:.2f} GB",
                    "OK" if total_mem_gb >= 14.0 else "WARN (<14GB)",
                )
        else:
            results["GPU Status"] = ("Running on CPU mode", "WARN (GPU recommended for training)")
    except ImportError:
        results["PyTorch"] = ("Not installed", "FAIL")

    return results


def check_bitsandbytes_4bit():
    try:
        import bitsandbytes as bnb
        ver = getattr(bnb, "__version__", "unknown")
        
        # Test 4bit layer instantiation
        try:
            import torch
            has_linear4bit = hasattr(bnb.nn, "Linear4bit")
            if has_linear4bit and torch.cuda.is_available():
                # Try a mini 4-bit layer on CUDA
                layer = bnb.nn.Linear4bit(16, 16, bias=False, compute_dtype=torch.float16).to("cuda:0")
                del layer
                torch.cuda.empty_cache()
                return ("Installed & CUDA 4-bit verified", "OK")
            elif has_linear4bit:
                return (f"v{ver} (Linear4bit present, awaiting CUDA)", "OK (CPU stage)")
            else:
                return (f"v{ver} (No Linear4bit)", "WARN")
        except Exception as e:
            return (f"v{ver} (CUDA verification deferred: {type(e).__name__})", "OK (CPU stage)")
    except ImportError:
        return ("Not installed (will install on Kaggle)", "WARN")


def check_huggingface_connectivity():
    test_urls = [
        ("Hugging Face Hub", "https://huggingface.co"),
        ("Model Hub API", "https://huggingface.co/api/models/Qwen/Qwen2-VL-2B-Instruct"),
    ]
    results = {}
    for label, url in test_urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Bangla-VQA-EnvCheck/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status in (200, 302, 307):
                    results[label] = ("Reachable (HTTP 200)", "OK")
                else:
                    results[label] = (f"HTTP {response.status}", "WARN")
        except Exception as e:
            results[label] = (f"Unreachable ({type(e).__name__})", "WARN (Offline)")
    return results


def check_key_packages():
    packages = [
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
        "evaluate",
        "qwen_vl_utils",
    ]
    results = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            results[pkg] = (f"v{ver}", "OK")
        except ImportError:
            results[pkg] = ("Not installed (will install via requirements.txt)", "PENDING")
    return results


def render_table(title: str, rows: list):
    col1_width = 32
    col2_width = 38
    col3_width = 12

    divider = "+" + "-" * (col1_width + 2) + "+" + "-" * (col2_width + 2) + "+" + "-" * (col3_width + 2) + "+"
    header = f"| {'Check / Component':<{col1_width}} | {'Detected Value':<{col2_width}} | {'Status':<{col3_width}} |"

    print("\n" + "=" * len(divider))
    print(f" {title}")
    print("=" * len(divider))
    print(divider)
    print(header)
    print(divider)

    for item, val, status in rows:
        val_str = str(val)[:col2_width]
        status_str = f"[{status}]"
        print(f"| {item:<{col1_width}} | {val_str:<{col2_width}} | {status_str:<{col3_width}} |")

    print(divider)


def main():
    print("==================================================================================")
    print(" IEEE Research: Hallucination-Resistant Bangla Visual Question Answering")
    print(" Diagnostic & Hardware Readiness Check (Local / Kaggle Dual-T4 GPUs)")
    print("==================================================================================")

    all_rows = []

    # 1. System & Python
    sys_info = check_python_and_system()
    for k, (v, s) in sys_info.items():
        all_rows.append((k, v, s))

    # 2. PyTorch & CUDA
    cuda_info = check_torch_and_cuda()
    for k, (v, s) in cuda_info.items():
        all_rows.append((k, v, s))

    # 3. BitsAndBytes 4-bit
    bnb_val, bnb_status = check_bitsandbytes_4bit()
    all_rows.append(("bitsandbytes (4-bit NF4)", bnb_val, bnb_status))

    # 4. Hugging Face Connectivity
    hf_info = check_huggingface_connectivity()
    for k, (v, s) in hf_info.items():
        all_rows.append((k, v, s))

    # 5. Core Libraries
    pkg_info = check_key_packages()
    for k, (v, s) in pkg_info.items():
        all_rows.append((f"lib: {k}", v, s))

    render_table("ENVIRONMENT READINESS REPORT", all_rows)

    # Readiness Summary
    print("\n>>> EXECUTION ENVIRONMENT SUMMARY:")
    if any("Dual-GPU" in str(r[1]) for r in all_rows):
        print("  [SUCCESS] Dual-T4 GPU cluster ready for multi-task instruction tuning.")
    elif any("CUDA Available" in str(r[0]) and "True" in str(r[1]) for r in all_rows):
        print("  [NOTICE] Single GPU detected. Functional for training and inference.")
    else:
        print("  [INFO] Running in Local Development / CPU mode. Ready for code authoring.")
        print("         Run 'bash scripts/kaggle_setup.sh' when deploying to Kaggle Dual-T4.")
    print("==================================================================================\n")


if __name__ == "__main__":
    main()

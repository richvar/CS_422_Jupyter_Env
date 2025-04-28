import torch
import gc
import subprocess

def run_nvidia_smi(prefix=""):
    print(f"\n{prefix}🔍 NVIDIA-SMI output:")
    try:
        result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"{prefix}❌ Failed to run nvidia-smi:\n{e}")

def print_cuda_memory(prefix=""):
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available.")
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"{prefix}Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB")

def cleanup_cuda():
    print("\n🧹 Checking CUDA memory before cleanup:")
    print_cuda_memory(prefix="  ")
    run_nvidia_smi(prefix="  ")

    print("\n♻️ Cleaning up...")
    gc.collect()
    torch.cuda.empty_cache()

    print("\n🧹 Checking CUDA memory after cleanup:")
    print_cuda_memory(prefix="  ")
    run_nvidia_smi(prefix="  ")

if __name__ == "__main__":
    cleanup_cuda()

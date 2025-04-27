
import os
from datetime import datetime
import subprocess

# Set working directory to the cloned repo (edit if needed)
os.chdir("stylegan2-ada-pytorch")

# Define results/log output directory
log_dir = "./results"
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

os.makedirs(log_dir, exist_ok=True)

# Run training
cmd = [
    "python3", "train.py",
    "--outdir=./results",
    "--data=./datasets/post-impressionist",
    "--gpus=1",
    "--snap=1",  # will use default tick size (50 kimg)
    "--resume=ffhq1024",
    "--cfg=auto",
    "--aug=ada"
]

with open(log_file, "w") as f:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in process.stdout:
        print(line, end="")
        f.write(line)

print(f"✅ Training started. Logs saved to: {log_file}")

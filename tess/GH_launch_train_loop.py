

import os
from datetime import datetime

# Fix Click's UTF-8 issue (important for StyleGAN2-ADA)
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

# Create output directory if it doesn't exist
log_dir = "./results"
os.makedirs(log_dir, exist_ok=True)

# Create log file with timestamp
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Set up the training command
cmd = f"""
python3 train.py \\
  --outdir=./results \\
  --data=./datasets/post-impressionist \\
  --gpus=1 \\
  --snap=1 \\
  --cfg=auto \\
  --resume=./results/00006-post-impressionist-auto1-ada-resumecustom/network-snapshot-001100.pkl \\
  --kimg-per-tick=10 \\
  --metrics=none \\
  --workers=1 \\
  --aug=ada | tee {log_file}
"""

# Run the command
os.system(cmd)

print(f"✅ Training started! Logs will be saved to:\n  {log_file}")
import os
from datetime import datetime

# Set environment variables, ASCII bug otherwise
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

# Define results and log output directory
log_dir = "./results"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Make sure to update transfer_weights directory
cmd = f"""
python3 train.py \\
  --outdir=./results \\
  --data=./datasets/post-impressionist \\
  --gpus=1 \\
  --snap=1 \\
  --resume=../transfer_weights/ffhq-res512-mirror-stylegan2-noaug.pkl \\
  --cfg=auto \\
  --kimg-per-tick=10 \\
  --metrics=none \\
  --workers=1 \\
  --aug=ada | tee {log_file}
"""

os.system(cmd)

print(f"Training started. Logs will be saved to: {log_file}")

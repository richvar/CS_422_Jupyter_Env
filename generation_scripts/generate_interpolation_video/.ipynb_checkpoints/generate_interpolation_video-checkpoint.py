import os
import sys
import torch
import numpy as np
import subprocess
from natsort import natsorted

# --- CONFIG ---

pkl_folder = '/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch/results/00008-post-impressionist-auto1-ada-resumecustom'  # Change this as needed
output_file = 'training_interpolation.webm'
truncation_psi = 0.5
fixed_seed = 1758362
fps = 10

# --- Setup imports ---
sys.path.append('/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch')

import dnnlib
import legacy

# --- Set device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --- Set random seed and sample one z ---
np.random.seed(fixed_seed)
torch.manual_seed(fixed_seed)
fixed_z = torch.from_numpy(np.random.randn(1, 512)).to(device)

# --- Find all .pkl files ---
pkl_files = [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')]
pkl_files = natsorted(pkl_files)

print(f'Found {len(pkl_files)} pkl snapshots.')

# --- Load first network to get image size ---
with open(pkl_files[0], 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

img = G(fixed_z, None, truncation_psi=truncation_psi, noise_mode='const')
_, C, H, W = img.shape  # Get resolution

# --- Setup ffmpeg subprocess ---
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # Overwrite output
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}',  # Size
    '-r', str(fps),  # Frames per second
    '-i', '-',  # Input comes from stdin
    '-an',  # No audio
    '-vcodec', 'libvpx',  # VP9 codec
    '-b:v', '5M',  # Increase bitrate (e.g., 5M for better quality, you can increase this value further)
    '-crf', '14',  # Lower CRF for better quality (lower = better quality)
    output_file
]


print(f'Starting ffmpeg to write {output_file}...')

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# --- Generate and pipe frames directly, but only every 5th .pkl file ---
for idx, pkl_path in enumerate(pkl_files):
    if idx % 5 != 0:  # Skip every 5th file
        continue

    print(f'[{idx+1}/{len(pkl_files)}] Processing {pkl_path}...')

    with open(pkl_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    img = G(fixed_z, None, truncation_psi=truncation_psi, noise_mode='const')
    img = (img + 1) * (255/2)
    img = img.clamp(0, 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).cpu().numpy()  # (H, W, C) and CPU numpy array

    # Stream raw RGB data to ffmpeg
    proc.stdin.write(img.tobytes())

# --- Finalize video ---
proc.stdin.close()
proc.wait()

print('✅ Done! Video created:', output_file)

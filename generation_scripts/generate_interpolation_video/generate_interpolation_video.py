import os
import sys
import torch
import numpy as np
import subprocess
from natsort import natsorted

# --- CONFIG ---

pkl_folder = '/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch/results/00008-post-impressionist-auto1-ada-resumecustom'  # Change this as needed
output_file = 'training_interpolation.webm'
truncation_psi = 0.5  # You can adjust this to control image diversity vs. quality
fixed_seed = 1758362  # Fixed seed to keep generating the same latent vector
fps = 10  # Frames per second for the video
num_random_images = 10  # Number of random images to generate

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
fixed_z = torch.from_numpy(np.random.randn(1, 512)).to(device)  # Latent vector 'z'

# --- Find all .pkl files ---
pkl_files = [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')]
pkl_files = natsorted(pkl_files)

print(f'Found {len(pkl_files)} pkl snapshots.')

# --- Load first network to get image size ---
with open(pkl_files[0], 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

img = G(fixed_z, None, truncation_psi=truncation_psi, noise_mode='const')
_, C, H, W = img.shape  # Get resolution

# --- Setup ffmpeg subprocess for high quality ---
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # Overwrite output
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}',  # Image size
    '-r', str(fps),  # Frames per second
    '-i', '-',  # Input comes from stdin
    '-an',  # No audio
    '-vcodec', 'libvpx',  # VP9 codec (WebM)
    '-b:v', '5M',  # Bitrate (higher = better quality)
    '-crf', '14',  # Quality setting (lower = better quality)
    output_file
]

print(f'Starting ffmpeg to write {output_file}...')

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# --- Generate and pipe frames directly, but only every 5th .pkl file ---
for idx, pkl_path in enumerate(pkl_files):
    if idx % 5 != 0:  # Skip every 5th file (you can change this to 3 or another number)
        continue

    print(f'[{idx+1}/{len(pkl_files)}] Processing {pkl_path}...')

    # Load the model from .pkl snapshot
    with open(pkl_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Generate image using the fixed latent vector
    img = G(fixed_z, None, truncation_psi=truncation_psi, noise_mode='const')

    # Post-process the image
    img = (img + 1) * (255 / 2)  # Rescale to [0, 255]
    img = img.clamp(0, 255).to(torch.uint8)  # Clamp pixel values to valid range
    img = img[0].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) numpy array

    # Stream the raw RGB data to ffmpeg
    proc.stdin.write(img.tobytes())

# --- Finalize the video ---
proc.stdin.close()
proc.wait()

print(f'✅ Done! Video created: {output_file}')

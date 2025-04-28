import os
import sys
import shutil
import torch
import numpy as np
from torchvision.utils import save_image

# --- CONFIG ---

richard_pkl_path = '/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch/results/00008-post-impressionist-auto1-ada-resumecustom/network-snapshot-010640.pkl'
gabe_pkl_path = '/home/s25vargason1/gabe/stylegan2-ada-pytorch/results/00009-post-impressionist-auto1-gamma30-kimg25000-batch8-ada-resumecustom/network-snapshot-001064.pkl'
tess_pkl_path = '/home/s25vargason1/tess/stylegan2-ada-pytorch/results/00008-post-impressionist-auto1-ada-resumecustom/network-snapshot-000330.pkl'

output_dir = './generated_images'


truncation_psi = 0.6 # <-- Higher values lead to more diverse, but may produce lower quality
                      # Lower values lead to more similar images (lower diversity), but higher quality

# List of custom seeds (leave empty to auto-generate 5 random ones)
seeds = [
         # 42, 
         # 123, 
         # 999, 
         # 2025, 
         # 777
        ]

num_random_images = 10  # Number of images to generate if seeds list is empty

teammate = 'tess'

# --- Choose which teammate's model to load ---
if teammate.lower() == 'richard':
    pkl_path = richard_pkl_path
elif teammate.lower() == 'gabe':
    pkl_path = gabe_pkl_path
elif teammate.lower() == 'tess':
    pkl_path = tess_pkl_path
else:
    raise ValueError(f"Unknown teammate '{teammate}'. Please choose Richard, Gabe, or Tess.")

print(f'✅ Selected model for {teammate}: {pkl_path}')

# --- Setup imports ---
sys.path.append('/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch')

import dnnlib
import legacy

# --- Set device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --- Prepare output directory (clean it) ---
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Delete the whole folder
os.makedirs(output_dir, exist_ok=True)
print(f'Output folder {output_dir} prepared.')

# --- Load network ---
print(f'Loading network from {pkl_path}...')
with open(pkl_path, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
print('✅ Network loaded!')

# --- If seeds is empty, generate random seeds ---
if not seeds:
    print(f'No seeds provided, generating {num_random_images} random images...')
    seeds = list(np.random.randint(0, 2**31, size=num_random_images))

# --- Generate images for each seed ---
for seed in seeds:
    print(f'Generating image for seed {seed}...')

    # Set the random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sample latent vector z
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    c = None  # No labels (unconditional model)

    # Generate image
    img = G(z, c, truncation_psi=truncation_psi, noise_mode='const')

    # Post-process
    img = (img + 1) * (255/2)
    img = img.clamp(0, 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).cpu()

    # Save with seed number as filename
    save_path = os.path.join(output_dir, f'{seed}.jpg')
    save_image(img.permute(2, 0, 1).float() / 255.0, save_path)

    print(f'Saved {save_path}')

print('✅ Done generating all images!')

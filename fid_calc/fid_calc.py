import os
import re
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy.linalg
import gc

import sys
sys.path.insert(0, '/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch')


# === CONFIGURATION ===
real_images_dir = '/home/s25vargason1/richard/512_basic_training/images/512images'
snapshots_dir = '/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch/results/00008-post-impressionist-auto1-ada-resumecustom'
output_csv = './fid_scores_simple.csv'
batch_size = 1
num_fakes = 4000
min_kimg = 100
snapshot_interval = 3  # every 3rd snapshot
inception_path = '/home/s25vargason1/fid_calc/inception-2015-12-05.pt'  # local inception model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD INCEPTION MODEL ===
print("📥 Loading Inception model...")
inception = torch.jit.load(inception_path).eval().to(device)

# === LOAD REAL IMAGES ===
print("📥 Loading real images...")
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

real_dataset = ImageFolder(root=real_images_dir, transform=transform)
real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

# === HELPER FUNCTIONS ===
def compute_features(loader):
    features = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feat = inception(imgs, return_features=True)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def calculate_fid(mu1, sigma1, mu2, sigma2):
    m = np.square(mu1 - mu2).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return float(fid)

# === PRECOMPUTE REAL FEATURES ===
print("🧮 Computing real image features...")
real_features = compute_features(real_loader)
mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)

# === FIND SNAPSHOTS ===
snapshot_pattern = re.compile(r'network-snapshot-(\d+)\.pkl')
snapshots = []

all_snapshots = []
for filename in sorted(os.listdir(snapshots_dir)):
    match = snapshot_pattern.match(filename)
    if match:
        kimg = int(match.group(1))
        if kimg >= min_kimg:
            all_snapshots.append((kimg, os.path.join(snapshots_dir, filename)))

snapshots = all_snapshots[::snapshot_interval]

# === EVALUATE EACH SNAPSHOT ===
fid_scores = []

for kimg, snapshot_path in snapshots:
    print(f"\n🔍 Evaluating snapshot {kimg} kimg: {snapshot_path}")
    
    # clear out memory leak
    if 'G' in globals():
        del G
    gc.collect()
    torch.cuda.empty_cache()

    with open(snapshot_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device).eval()

    # Generate fakes
    print(f"🖼️ Generating {num_fakes} fake images...")
    fake_images = []
    with torch.no_grad():
        for _ in range(num_fakes // batch_size):
            z = torch.randn([batch_size, G.z_dim], device=device)
            c = torch.zeros([batch_size, G.c_dim], device=device)
            img = G(z=z, c=c, truncation_psi=1.0, noise_mode='const')
            img = (img * 0.5 + 0.5).clamp(0, 1)
            fake_images.append(img)
    fake_images = torch.cat(fake_images, dim=0)

    # Compute fake features
    print("🧮 Computing fake features...")
    fake_features = []
    for i in range(0, num_fakes, batch_size):
        imgs = fake_images[i:i+batch_size]
        feat = inception(imgs, return_features=True)
        fake_features.append(feat.cpu().numpy())
    fake_features = np.concatenate(fake_features, axis=0)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID
    fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"✅ FID at {kimg} kimg: {fid:.2f}")

    fid_scores.append((kimg, fid))

# === SAVE FID RESULTS ===
print("\n📄 Writing FID scores to CSV...")
with open(output_csv, 'w') as f:
    f.write("kimg,FID\n")
    for kimg, fid in fid_scores:
        f.write(f"{kimg},{fid:.6f}\n")

print(f"✅ All done! Results saved to {output_csv}")

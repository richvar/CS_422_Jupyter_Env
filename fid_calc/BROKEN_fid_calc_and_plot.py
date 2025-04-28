import os
import subprocess
import re
import csv

try:
    import matplotlib.pyplot as plt
    plotting_available = True
except ImportError:
    print("⚠️ matplotlib not available, skipping plot.")
    plotting_available = False

# === CONFIGURATION ===
snapshots_dir = '/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch/results/00008-post-impressionist-auto1-ada-resumecustom'
real_data_dir = '/home/s25vargason1/richard/512_basic_training/images/512images'
calc_metrics_path = '/home/s25vargason1/richard/512_basic_training/stylegan2-ada-pytorch/calc_metrics.py'
output_csv = './fid_scores.csv'
plot_output = './fid_plot.png'
gpus = 1
metric = 'fid4k_full'
min_kimg = 100  # Skip snapshots before 100 kimg
snapshot_interval = 3  # Only evaluate every 3rd snapshot

# === CHECK REAL DATASET ===
print("\n🔎 Checking real dataset...")
try:
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=real_data_dir,
        transform=transform,
    )

    print(f"✅ Found {len(dataset)} real images in {real_data_dir}")
except Exception as e:
    print(f"❌ Failed to load real dataset: {e}")
    exit(1)


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

# Pick every 3rd snapshot
snapshots = all_snapshots[::snapshot_interval]

# === EVALUATE FID FOR EACH SNAPSHOT ===
fid_scores = []

for kimg, snapshot_path in snapshots:
    print(f"\n🔍 Evaluating FID for snapshot at {kimg} kimg: {snapshot_path}")

    try:
        result = subprocess.run([
            'python', calc_metrics_path,
            f'--gpus={gpus}',
            f'--data={real_data_dir}',
            f'--metrics={metric}',
            f'--network={snapshot_path}'
        ], text=True, check=True)

        for line in result.stdout.splitlines():
            if line.startswith(f'{metric}:'):
                fid = float(line.split(':')[1].strip())
                print(f"✅ FID at {kimg} kimg = {fid}")
                fid_scores.append((kimg, fid))
                break

    except subprocess.CalledProcessError as e:
        print(f"❌ Error evaluating {snapshot_path}:")
        print(e.stdout)
        print(e.stderr)

# === SAVE RESULTS TO CSV ===
print("\n📄 Writing FID scores to CSV...")
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['kimg', 'FID'])
    writer.writerows(fid_scores)

print(f"✅ FID scores saved to: {output_csv}")

# === PLOT FID CURVE (optional) ===
if plotting_available:
    print("\n📈 Plotting FID vs kimg...")
    fid_scores.sort()
    kimgs, fids = zip(*fid_scores)

    plt.figure(figsize=(10,6))
    plt.plot(kimgs, fids, marker='o')
    plt.xlabel('Training Progress (kimg)')
    plt.ylabel('FID (lower is better)')
    plt.title('FID over Training Progress')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_output)
    plt.show()

    print(f"✅ Plot saved to: {plot_output}")
else:
    print("⚡ Skipping plot because matplotlib is not installed.")

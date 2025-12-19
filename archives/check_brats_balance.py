import numpy as np
from pathlib import Path
from tqdm import tqdm

# Check BraTS dataset balance
brats_dir = Path('data/processed/brats2d')
files = list(brats_dir.glob('*.npz'))

print(f"Checking {len(files)} BraTS files...\n")

tumor_count = 0
no_tumor_count = 0

for f in tqdm(files, desc="Scanning"):
    data = np.load(f, allow_pickle=True)
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    if metadata.get('has_tumor', False):
        tumor_count += 1
    else:
        no_tumor_count += 1

print(f"\n{'='*50}")
print(f"Dataset Balance:")
print(f"{'='*50}")
print(f"Total:    {len(files):,} slices")
print(f"Tumor:    {tumor_count:,} ({tumor_count/len(files)*100:.1f}%)")
print(f"No-tumor: {no_tumor_count:,} ({no_tumor_count/len(files)*100:.1f}%)")
print(f"{'='*50}")

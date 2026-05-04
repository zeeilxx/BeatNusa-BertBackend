import numpy as np
from pathlib import Path

processed = Path("data/processed")

total_pos = 0
total_neg = 0
total_frames = 0

for npz_path in sorted(processed.glob("*.npz")):
    if "__" in npz_path.stem:
        continue
    d = np.load(npz_path, allow_pickle=True)
    ev = d['event']
    n_frames = ev.shape[0]
    n_pos = int(ev.sum())
    total_pos += n_pos
    total_neg += n_frames - n_pos
    total_frames += n_frames
    print(f"{npz_path.stem:30s} frames={n_frames:6d} pos={n_pos:4d} ratio={n_pos/max(1,n_frames):.4f}")

print()
print(f"Total original songs: {len(list(processed.glob('*.npz'))) - len(list(processed.glob('*__*.npz')))}")
print(f"Total frames: {total_frames}")
print(f"Positive: {total_pos}, Negative: {total_neg}")
print(f"Pos ratio: {total_pos/total_frames:.6f}")
print(f"Neg/Pos: {total_neg/total_pos:.1f}")

# Check sample counts for train dataset
import sys
sys.path.insert(0, str(Path('src')))
from beatbert.configs import load_config, resolve_paths
cfg = resolve_paths(load_config('configs/default.yaml'), Path('.').resolve())
from beatbert.data.dataset import BeatmapDataset
train_ds = BeatmapDataset(cfg, 'train')
val_ds = BeatmapDataset(cfg, 'val')
print(f"\nTrain samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")

# Check event distribution in train samples
pos_count = 0
neg_count = 0
for i in range(min(20, len(train_ds))):
    s = train_ds[i]
    ev = s['event']
    mask = s['attention_mask']
    valid = ev[mask > 0.5]
    pos_count += int((valid > 0.5).sum())
    neg_count += int((valid <= 0.5).sum())

print(f"\nFirst 20 train samples - pos: {pos_count}, neg: {neg_count}, ratio: {pos_count/max(1,pos_count+neg_count):.4f}")

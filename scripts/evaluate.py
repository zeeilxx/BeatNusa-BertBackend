from pathlib import Path
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beatbert.configs import load_config, resolve_paths
from beatbert.data.dataset import BeatmapDataset
from beatbert.models.beatmap_model import BeatmapModel
from beatbert.training.metrics import decode_events, event_level_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    cfg = resolve_paths(load_config(args.config), ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['general']['device'] != 'cpu' else 'cpu')
    ds = BeatmapDataset(cfg, 'test')
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model = BeatmapModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    metrics = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['mel'], batch['attention_mask'])
            pred_events = decode_events(
                outputs['event_logits'].squeeze(0).cpu().numpy(),
                outputs['lane_logits'].squeeze(0).cpu().numpy(),
                batch['frame_times_ms'].squeeze(0).cpu().numpy(),
                cfg['eval']['event_threshold'],
            )
            true_idxs = torch.where(batch['event'].squeeze(0) > 0.5)[0].cpu().numpy()
            frame_times = batch['frame_times_ms'].squeeze(0).cpu().numpy()
            lanes = batch['lane'].squeeze(0).cpu().numpy()
            true_events = [(float(frame_times[i]), int(lanes[i])) for i in true_idxs if frame_times[i] >= 0]
            metrics.append(event_level_metrics(pred_events, true_events, cfg['eval']['tolerance_ms']))

    avg = {k: float(np.nanmean([m[k] for m in metrics])) for k in metrics[0]}
    for k, v in avg.items():
        print(f'{k}: {v:.4f}')


if __name__ == '__main__':
    main()

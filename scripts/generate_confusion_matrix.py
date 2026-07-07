"""
Generate Confusion Matrix for Event Note Classification (BeatmapBERT).

Evaluates the model on validation and/or test sets, producing:
  - Confusion matrix images (PNG) ready for thesis
  - Summary text with TP, FP, FN, TN, precision, recall, F1

Uses the same evaluation logic as trainer.py:
  - frame_classification_metrics with sigmoid threshold on event_logits
  - threshold from config: eval.event_threshold (0.42)

Output directory: thesis_artifacts/confusion_matrix/

Fix notes:
  - num_workers=0 to avoid MemoryError on Windows (NPZ files are 30-80MB each,
    multi-worker DataLoader duplicates memory per forked process)
  - batch_size=8 to reduce peak GPU memory
  - Full diagnostic output before inference
"""

from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─── Project path setup ───
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beatbert.configs import load_config, resolve_paths
from beatbert.data.dataset import BeatmapDataset
from beatbert.models.beatmap_model import BeatmapModel


# ─────────────────────────────────────────────────────────────
#  DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def print_diagnostic_header(cfg: Dict, config_path: Path, checkpoint_path: Path):
    """Print comprehensive diagnostic information before any inference."""
    print('=' * 70)
    print('  DIAGNOSTIC: Environment & Paths')
    print('=' * 70)
    print(f'  CWD                  : {os.getcwd()}')
    print(f'  ROOT (project)       : {ROOT}')
    print(f'  Config path          : {config_path}')
    print(f'    exists?            : {config_path.exists()}')
    print(f'  Checkpoint path      : {checkpoint_path}')
    print(f'    exists?            : {checkpoint_path.exists()}')

    splits_dir = Path(cfg['paths']['splits_dir'])
    val_csv = splits_dir / 'val.csv'
    test_csv = splits_dir / 'test.csv'
    print(f'  Splits dir           : {splits_dir}')
    print(f'  val.csv path         : {val_csv}')
    print(f'    exists?            : {val_csv.exists()}')
    print(f'  test.csv path        : {test_csv}')
    print(f'    exists?            : {test_csv.exists()}')

    import pandas as pd
    for name, csv_path in [('val', val_csv), ('test', test_csv)]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f'\n  --- {name}.csv ---')
            print(f'    Rows    : {len(df)}')
            print(f'    Columns : {list(df.columns)}')
            print(f'    First 5 rows:')
            for i, row in df.head().iterrows():
                print(f'      [{i}] {row["song_id"]}')

    processed_dir = Path(cfg['paths']['processed_dir'])
    npz_files = list(processed_dir.glob('*.npz'))
    print(f'\n  Processed dir        : {processed_dir}')
    print(f'    exists?            : {processed_dir.exists()}')
    print(f'    .npz files found   : {len(npz_files)}')


def diagnose_dataset(ds: BeatmapDataset, split_name: str):
    """Diagnose dataset: print length and inspect first sample."""
    print(f'\n  --- Dataset diagnostic: {split_name} ---')
    print(f'    len(dataset)       : {len(ds)}')
    print(f'    num song_paths     : {len(ds.song_paths)}')

    if len(ds) == 0:
        print(f'    *** ERROR: Dataset is EMPTY! ***')
        print(f'    song_paths found:')
        for p in ds.song_paths[:5]:
            print(f'      {p} (exists={p.exists()})')
        raise RuntimeError(f'Dataset "{split_name}" has 0 samples. Cannot proceed.')

    # Inspect first sample
    print(f'    Inspecting sample [0]...')
    sample = ds[0]
    for key, tensor in sample.items():
        print(f'      {key:20s} : shape={tuple(tensor.shape)}, dtype={tensor.dtype}')

    event = sample['event']
    mask = sample['attention_mask']
    valid = mask.bool()
    valid_event = event[valid]
    n_pos = int((valid_event == 1).sum())
    n_neg = int((valid_event == 0).sum())
    print(f'      valid frames         : {int(valid.sum())}')
    print(f'      event positive (1)   : {n_pos}')
    print(f'      event negative (0)   : {n_neg}')

    # Source file info
    path, start, end = ds.samples[0]
    print(f'      source file          : {path.name}')
    print(f'      frame range          : [{start}, {end})')


def diagnose_dataloader(loader: DataLoader, split_name: str):
    """Print DataLoader info."""
    print(f'\n  --- DataLoader diagnostic: {split_name} ---')
    print(f'    num batches        : {len(loader)}')
    print(f'    batch_size         : {loader.batch_size}')
    print(f'    num_workers        : {loader.num_workers}')


# ─────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────

def collect_predictions(
    model: BeatmapModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, np.ndarray]:
    """Run inference on all batches and collect frame-level predictions & labels.

    Only valid (non-padding) frames are included.
    """
    all_preds = []
    all_labels = []
    total_batches = len(loader)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['mel'], batch['attention_mask'])

            # Sigmoid → threshold → binary prediction (same as trainer.py)
            event_prob = torch.sigmoid(outputs['event_logits'])
            event_pred = (event_prob >= threshold).long()
            event_true = batch['event'].long()
            mask = batch['attention_mask'].bool()

            # Extract valid frames only (exclude padding)
            for i in range(event_pred.shape[0]):
                valid = mask[i]
                all_preds.append(event_pred[i][valid].cpu().numpy())
                all_labels.append(event_true[i][valid].cpu().numpy())

            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == total_batches:
                print(f'    Batch {batch_idx + 1}/{total_batches} done')

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {'preds': preds, 'labels': labels}


# ─────────────────────────────────────────────────────────────
#  CONFUSION MATRIX COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> Dict[str, int]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def compute_metrics_from_cm(cm: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = cm['TP'], cm['FP'], cm['FN']
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}


# ─────────────────────────────────────────────────────────────
#  VALIDATION OF RESULTS
# ─────────────────────────────────────────────────────────────

def validate_results(preds: np.ndarray, labels: np.ndarray, cm: Dict[str, int], split_name: str):
    """Validate confusion matrix results before generating images."""
    total_frames = len(preds)
    cm_total = cm['TP'] + cm['FP'] + cm['FN'] + cm['TN']

    print(f'\n  --- Validation of results: {split_name} ---')
    print(f'    Total valid frames     : {total_frames:,}')
    print(f'    Actual Note (1)        : {int((labels == 1).sum()):,}')
    print(f'    Actual No Note (0)     : {int((labels == 0).sum()):,}')
    print(f'    Predicted Note (1)     : {int((preds == 1).sum()):,}')
    print(f'    Predicted No Note (0)  : {int((preds == 0).sum()):,}')
    print(f'    TP = {cm["TP"]:>10,}')
    print(f'    FP = {cm["FP"]:>10,}')
    print(f'    FN = {cm["FN"]:>10,}')
    print(f'    TN = {cm["TN"]:>10,}')
    print(f'    CM total (TP+FP+FN+TN): {cm_total:,}')

    # Check 1: total frames > 0
    if total_frames == 0:
        raise RuntimeError(f'FATAL: Total frames = 0 for {split_name}. No data was processed.')

    # Check 2: CM total == total frames
    if cm_total != total_frames:
        raise RuntimeError(
            f'FATAL: CM total ({cm_total}) != total frames ({total_frames}). '
            f'Data integrity error.'
        )

    # Check 3: no NaN
    for key, val in cm.items():
        if np.isnan(val):
            raise RuntimeError(f'FATAL: {key} is NaN.')

    print(f'    ✓ All checks passed.')


# ─────────────────────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: Dict[str, int],
    metrics: Dict[str, float],
    split_name: str,
    threshold: float,
    output_path: Path,
):
    """Create a publication-quality confusion matrix plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Build 2x2 matrix: rows=Actual, cols=Predicted
    # Layout:  [[TN, FP],
    #           [FN, TP]]
    matrix = np.array([
        [cm['TN'], cm['FP']],
        [cm['FN'], cm['TP']],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap, aspect='equal')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    total = matrix.sum()
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = count / total * 100 if total > 0 else 0
            text_color = 'white' if count > matrix.max() * 0.5 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.2f}%)',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=text_color)

    class_labels = ['No Note (0)', 'Note (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_yticklabels(class_labels, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold', labelpad=10)

    split_display = split_name.capitalize()
    title = (
        f'Confusion Matrix — Event Note Classification ({split_display} Set)\n'
        f'Threshold = {threshold} | '
        f'Precision = {metrics["precision"]:.4f} | '
        f'Recall = {metrics["recall"]:.4f} | '
        f'F1 = {metrics["f1"]:.4f}'
    )
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved image: {output_path}')


# ─────────────────────────────────────────────────────────────
#  PER-SPLIT EVALUATION
# ─────────────────────────────────────────────────────────────

def generate_for_split(
    split_name: str,
    model: BeatmapModel,
    cfg: Dict,
    device: torch.device,
    threshold: float,
    output_dir: Path,
    checkpoint_metrics: Dict | None = None,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Generate confusion matrix for a given split."""
    print(f'\n{"=" * 70}')
    print(f'  Processing: {split_name} set')
    print(f'{"=" * 70}')

    # ── Build dataset (same class as trainer.py) ──
    ds = BeatmapDataset(cfg, split_name)
    diagnose_dataset(ds, split_name)

    # ── DataLoader: num_workers=0 to avoid MemoryError on Windows ──
    # NPZ files are 30-80MB each. Multi-worker DataLoader on Windows uses
    # spawn (not fork), which duplicates memory per worker process.
    # With 30k+ samples and 2 workers, this exceeds available RAM.
    loader = DataLoader(
        ds,
        batch_size=8,        # smaller to reduce peak memory
        shuffle=False,
        num_workers=0,        # CRITICAL: avoid MemoryError
        pin_memory=False,
    )
    diagnose_dataloader(loader, split_name)

    # ── Run inference ──
    print(f'\n  Running inference (threshold={threshold})...')
    result = collect_predictions(model, loader, device, threshold)
    preds = result['preds']
    labels = result['labels']

    # ── Compute confusion matrix ──
    cm = compute_confusion_matrix(preds, labels)
    metrics = compute_metrics_from_cm(cm)

    # ── Validate ──
    validate_results(preds, labels, cm, split_name)

    print(f'\n  Metrics (from confusion matrix):')
    print(f'    Precision = {metrics["precision"]:.6f}')
    print(f'    Recall    = {metrics["recall"]:.6f}')
    print(f'    F1-score  = {metrics["f1"]:.6f}')

    # ── Consistency check with checkpoint ──
    if checkpoint_metrics and split_name == 'val':
        print(f'\n  Consistency check with checkpoint best.pt:')
        ckpt_p = checkpoint_metrics.get('precision', 0)
        ckpt_r = checkpoint_metrics.get('recall', 0)
        ckpt_f1 = checkpoint_metrics.get('f1', 0)
        print(f'    Checkpoint Precision = {ckpt_p:.6f}')
        print(f'    Checkpoint Recall    = {ckpt_r:.6f}')
        print(f'    Checkpoint F1        = {ckpt_f1:.6f}')

        diff_p = abs(metrics['precision'] - ckpt_p)
        diff_r = abs(metrics['recall'] - ckpt_r)
        diff_f1 = abs(metrics['f1'] - ckpt_f1)
        print(f'    Δ Precision = {diff_p:.6f}')
        print(f'    Δ Recall    = {diff_r:.6f}')
        print(f'    Δ F1        = {diff_f1:.6f}')

        if max(diff_p, diff_r, diff_f1) < 0.05:
            print(f'    ✓ Results are consistent with checkpoint metrics.')
        else:
            print(f'    ⚠ Difference detected. Possible causes:')
            print(f'      1. Checkpoint stores batch-averaged metrics (mean of per-batch P/R/F1)')
            print(f'         while this script computes globally over all frames.')
            print(f'      2. Dataset negative_sample_ratio filters some negative samples,')
            print(f'         changing the TN/FP distribution between runs.')
            print(f'      3. Both are valid — this script gives the TRUE global confusion matrix.')

    # ── Plot ──
    img_path = output_dir / f'confusion_matrix_event_note_{split_name}.png'
    plot_confusion_matrix(cm, metrics, split_name, threshold, img_path)

    return cm, metrics


# ─────────────────────────────────────────────────────────────
#  REPORT GENERATION
# ─────────────────────────────────────────────────────────────

def write_report(
    output_dir: Path,
    threshold: float,
    checkpoint_epoch: int,
    checkpoint_metrics: Dict,
    val_cm: Dict[str, int] | None,
    val_metrics: Dict[str, float] | None,
    val_total_frames: int,
    test_cm: Dict[str, int] | None,
    test_metrics: Dict[str, float] | None,
    test_total_frames: int,
):
    """Write confusion_matrix_report.md with all results."""
    lines = []
    lines.append('# Confusion Matrix Report — Event Note Classification\n')
    lines.append(f'**Model**: BeatmapBERT (CNN + Transformer Encoder)\n')
    lines.append(f'**Checkpoint**: best.pt (epoch {checkpoint_epoch})\n')
    lines.append(f'**Threshold**: {threshold}\n')
    lines.append(f'**Date**: Generated by generate_confusion_matrix.py\n')
    lines.append('')

    if val_cm and val_metrics:
        lines.append('## Validation Set\n')
        lines.append(f'- Total frames evaluated: **{val_total_frames:,}**\n')
        lines.append('')
        lines.append('| | **Predicted: No Note** | **Predicted: Note** |')
        lines.append('|---|---:|---:|')
        lines.append(f'| **Actual: No Note** | TN = {val_cm["TN"]:,} | FP = {val_cm["FP"]:,} |')
        lines.append(f'| **Actual: Note** | FN = {val_cm["FN"]:,} | TP = {val_cm["TP"]:,} |')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|---|---|')
        lines.append(f'| Precision | {val_metrics["precision"]:.6f} |')
        lines.append(f'| Recall | {val_metrics["recall"]:.6f} |')
        lines.append(f'| F1-score | {val_metrics["f1"]:.6f} |')
        lines.append('')

        ckpt_p = checkpoint_metrics.get('precision', 0)
        ckpt_r = checkpoint_metrics.get('recall', 0)
        ckpt_f1 = checkpoint_metrics.get('f1', 0)
        lines.append('### Consistency with Checkpoint\n')
        lines.append('| Metric | Checkpoint | This Eval | Δ |')
        lines.append('|---|---|---|---|')
        lines.append(f'| Precision | {ckpt_p:.6f} | {val_metrics["precision"]:.6f} | {abs(val_metrics["precision"] - ckpt_p):.6f} |')
        lines.append(f'| Recall | {ckpt_r:.6f} | {val_metrics["recall"]:.6f} | {abs(val_metrics["recall"] - ckpt_r):.6f} |')
        lines.append(f'| F1 | {ckpt_f1:.6f} | {val_metrics["f1"]:.6f} | {abs(val_metrics["f1"] - ckpt_f1):.6f} |')
        lines.append('')
        lines.append('> **Note**: Differences are expected because the checkpoint stores metrics')
        lines.append('> averaged per-batch during validation, while this script computes metrics')
        lines.append('> globally across all frames. The dataset `negative_sample_ratio` also')
        lines.append('> affects which negative samples are included.\n')

    if test_cm and test_metrics:
        lines.append('## Test Set\n')
        lines.append(f'- Total frames evaluated: **{test_total_frames:,}**\n')
        lines.append('')
        lines.append('| | **Predicted: No Note** | **Predicted: Note** |')
        lines.append('|---|---:|---:|')
        lines.append(f'| **Actual: No Note** | TN = {test_cm["TN"]:,} | FP = {test_cm["FP"]:,} |')
        lines.append(f'| **Actual: Note** | FN = {test_cm["FN"]:,} | TP = {test_cm["TP"]:,} |')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|---|---|')
        lines.append(f'| Precision | {test_metrics["precision"]:.6f} |')
        lines.append(f'| Recall | {test_metrics["recall"]:.6f} |')
        lines.append(f'| F1-score | {test_metrics["f1"]:.6f} |')
        lines.append('')

    md_path = output_dir / 'confusion_matrix_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved report: {md_path}')


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    config_path = ROOT / 'configs' / 'local.yaml'
    checkpoint_path = ROOT / 'checkpoints' / 'best.pt'
    output_dir = ROOT / 'thesis_artifacts' / 'confusion_matrix'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also prepare a debug log file
    log_path = output_dir / 'evaluation_debug.log'
    # Tee stdout to both console and log file
    import io

    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers
        def write(self, s):
            for w in self.writers:
                w.write(s)
                w.flush()
        def flush(self):
            for w in self.writers:
                w.flush()

    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = TeeWriter(sys.__stdout__, log_file)

    try:
        print('=' * 70)
        print('  BeatmapBERT — Confusion Matrix Generator')
        print('=' * 70)

        # ── Load config ──
        print(f'\nLoading config from {config_path}...')
        cfg = resolve_paths(load_config(str(config_path)), ROOT)

        # ── Load checkpoint ──
        print(f'Loading checkpoint from {checkpoint_path}...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')

        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        checkpoint_metrics = ckpt.get('val_metrics', {})
        checkpoint_epoch = ckpt.get('epoch', 'N/A')

        print(f'Checkpoint epoch: {checkpoint_epoch}')
        print(f'Checkpoint val_metrics: {checkpoint_metrics}')

        threshold = float(cfg['eval']['event_threshold'])
        print(f'Evaluation threshold: {threshold}')

        # ── Diagnostics ──
        print_diagnostic_header(cfg, config_path, checkpoint_path)

        # ── Build model ──
        print('\nBuilding model...')
        model = BeatmapModel(cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Model parameters: {total_params:,}')

        # ── Validation set (primary) ──
        val_cm, val_metrics = generate_for_split(
            'val', model, cfg, device, threshold, output_dir,
            checkpoint_metrics=checkpoint_metrics,
        )

        # Count total frames for report
        val_total = val_cm['TP'] + val_cm['FP'] + val_cm['FN'] + val_cm['TN']

        # ── Test set (secondary) ──
        test_cm = None
        test_metrics = None
        test_total = 0
        try:
            test_cm, test_metrics = generate_for_split(
                'test', model, cfg, device, threshold, output_dir,
            )
            test_total = test_cm['TP'] + test_cm['FP'] + test_cm['FN'] + test_cm['TN']
        except FileNotFoundError as e:
            print(f'\n  ⚠ Test set not available: {e}')
        except Exception as e:
            print(f'\n  ⚠ Test set evaluation failed: {e}')
            traceback.print_exc()

        # ── Write report ──
        write_report(
            output_dir, threshold, checkpoint_epoch, checkpoint_metrics,
            val_cm, val_metrics, val_total,
            test_cm, test_metrics, test_total,
        )

        # ── Save JSON summary ──
        summary = {
            'threshold': threshold,
            'checkpoint_epoch': checkpoint_epoch,
            'validation': {
                'total_frames': val_total,
                'confusion_matrix': val_cm,
                'metrics': {k: round(v, 6) for k, v in val_metrics.items()},
            },
        }
        if test_cm and test_metrics:
            summary['test'] = {
                'total_frames': test_total,
                'confusion_matrix': test_cm,
                'metrics': {k: round(v, 6) for k, v in test_metrics.items()},
            }
        json_path = output_dir / 'confusion_matrix_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'  Saved JSON: {json_path}')

        print(f'\n{"=" * 70}')
        print(f'  All outputs saved to: {output_dir}')
        print(f'  Debug log saved to  : {log_path}')
        print(f'{"=" * 70}')

    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == '__main__':
    main()

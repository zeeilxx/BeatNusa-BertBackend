from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from beatbert.data.dataset import BeatmapDataset
from beatbert.models.beatmap_model import BeatmapModel
from beatbert.training.losses import compute_losses
from beatbert.training.metrics import frame_classification_metrics

console = Console()


def _auto_device(cfg: Dict) -> torch.device:
    requested = cfg['general']['device']
    if requested != 'auto':
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _estimate_pos_weight(train_dataset: BeatmapDataset) -> float:
    """Estimasi pos_weight dengan sampling cepat (tidak perlu baca semua file)."""
    import numpy as np
    pos = 0.0
    neg = 0.0
    # Sampling maksimal 200 file, bukan seluruh dataset
    all_paths = list({s[0] for s in train_dataset.samples})  # unique paths
    rng = np.random.RandomState(42)
    sample_paths = rng.choice(all_paths, size=min(200, len(all_paths)), replace=False)
    for path in sample_paths:
        with np.load(path) as data:
            event = data['event']
            pos += float((event == 1).sum())
            neg += float((event == 0).sum())
    return max(1.0, neg / max(1.0, pos))


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, steps_per_epoch: int):
    """Cosine Annealing LR scheduler with linear warmup.

    Warmup membantu training stabil di awal epoch pertama.
    Cosine annealing menurunkan LR secara gradual untuk konvergensi yang lebih baik.
    """
    warmup_epochs = int(cfg['train'].get('warmup_epochs', 3))
    total_epochs = int(cfg['train']['epochs'])
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: dari 0 ke 1
            return float(current_step) / max(1, warmup_steps)
        # Cosine annealing: dari 1 ke min_lr_ratio
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        min_lr_ratio = float(cfg['train'].get('min_lr_ratio', 0.01))
        import math
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class TrainArtifacts:
    model: BeatmapModel
    device: torch.device


def build_dataloaders(cfg: Dict):
    train_ds = BeatmapDataset(cfg, 'train')
    val_ds = BeatmapDataset(cfg, 'val')
    nw = int(cfg['general']['num_workers'])
    loader_kwargs = dict(
        batch_size=cfg['train']['batch_size'],
        num_workers=nw,
        pin_memory=cfg['general']['pin_memory'],
        persistent_workers=(nw > 0),   # Keep workers alive — hindari overhead respawn
        prefetch_factor=4 if nw > 0 else None,  # Pre-load 4 batch ke depan
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    
    # Val_loader juga pakai worker agar tidak bottleneck di disk I/O.
    # Tanpa worker, setiap NPZ file dibaca di main thread → 15-20 menit gap antar epoch!
    val_nw = max(2, nw // 2)  # Minimal 2 worker, atau setengah dari train workers
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=cfg['train']['batch_size'],
        num_workers=val_nw,
        pin_memory=cfg['general']['pin_memory'],
        persistent_workers=(val_nw > 0),
        prefetch_factor=4 if val_nw > 0 else None,
    )
    return train_ds, val_ds, train_loader, val_loader


def train(cfg: Dict) -> TrainArtifacts:
    device = _auto_device(cfg)
    console.print(f'[bold green]Device:[/bold green] {device}')

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg)
    console.print(f'[cyan]Train samples: {len(train_ds)}, Val samples: {len(val_ds)}[/cyan]')

    model = BeatmapModel(cfg).to(device)

    # Log jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f'[cyan]Model params: {trainable_params:,} trainable / {total_params:,} total[/cyan]')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )

    # LR Scheduler
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['train']['amp']) and device.type == 'cuda')

    if cfg['train']['event_positive_weight'] == 'auto':
        pos_weight_value = _estimate_pos_weight(train_ds)
        console.print(f'[cyan]Auto pos_weight: {pos_weight_value:.2f}[/cyan]')
    else:
        pos_weight_value = float(cfg['train']['event_positive_weight'])
    pos_weight = torch.tensor(pos_weight_value, device=device)

    use_focal = cfg['train'].get('use_focal_loss', True)
    if use_focal:
        console.print(
            f'[bold magenta]Using Focal Loss[/bold magenta] '
            f'alpha={cfg["train"].get("focal_alpha", 0.75)}, '
            f'gamma={cfg["train"].get("focal_gamma", 2.0)}'
        )

    best_val_f1 = 0.0
    patience = 0
    ckpt_dir = Path(cfg['paths']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor = cfg['train'].get('monitor_metric', 'f1')  # 'f1' or 'loss'
    
    start_epoch = 1
    resume_path = ckpt_dir / 'last.pt'
    if resume_path.exists():
        console.print(f'[bold yellow]Auto-Resume aktif: Melanjutkan dari {resume_path}[/bold yellow]')
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state['epoch'] + 1
        if monitor == 'f1' and 'val_metrics' in state:
            best_val_f1 = state['val_metrics']['f1']
        elif monitor == 'loss' and 'val_loss' in state:
            best_val_f1 = state['val_loss']

    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        # ============ TRAINING ============
        model.train()
        train_loss = 0.0
        train_event_loss = 0.0
        train_lane_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Train {epoch}/{cfg["train"]["epochs"]}')
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                outputs = model(batch['mel'], batch['attention_mask'])
                losses = compute_losses(outputs, batch, cfg, pos_weight)

            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += float(losses['loss'].item())
            train_event_loss += float(losses['event_loss'].item())
            train_lane_loss += float(losses['lane_loss'].item())

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{losses["loss"].item():.4f}', lr=f'{current_lr:.6f}')

        n_batches = max(1, len(train_loader))
        avg_train = train_loss / n_batches
        avg_event = train_event_loss / n_batches
        avg_lane = train_lane_loss / n_batches

        # ============ VALIDATION ============
        validate_every = int(cfg['train'].get('validate_every', 1))  # 0 = skip, N = setiap N epoch
        run_val = validate_every > 0 and (epoch % validate_every == 0 or epoch == cfg['train']['epochs'])

        if run_val:
            val_loss, val_metrics = validate(model, val_loader, device, cfg, pos_weight)

            current_lr = optimizer.param_groups[0]['lr']
            console.print(
                f'[bold cyan]Epoch {epoch:3d}[/bold cyan] '
                f'lr={current_lr:.6f} '
                f'train_loss={avg_train:.4f} (event={avg_event:.4f} lane={avg_lane:.4f}) '
                f'val_loss={val_loss:.4f} '
                f'val_f1={val_metrics["f1"]:.4f} '
                f'precision={val_metrics["precision"]:.4f} '
                f'recall={val_metrics["recall"]:.4f} '
                f'lane_acc={val_metrics["lane_acc"]:.4f}'
            )
        else:
            val_loss = 0.0
            val_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'lane_acc': 0.0}
            current_lr = optimizer.param_groups[0]['lr']
            console.print(
                f'[bold cyan]Epoch {epoch:3d}[/bold cyan] '
                f'lr={current_lr:.6f} '
                f'train_loss={avg_train:.4f} (event={avg_event:.4f} lane={avg_lane:.4f}) '
                f'[dim](validation skipped)[/dim]'
            )

        # ============ CHECKPOINTING ============
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': cfg,
            'val_metrics': val_metrics,
            'val_loss': val_loss,
        }
        torch.save(state, ckpt_dir / 'last.pt')

        if not run_val:
            # Tanpa validasi, simpan best.pt di epoch terakhir saja
            if epoch == cfg['train']['epochs']:
                torch.save(state, ckpt_dir / 'best.pt')
                console.print(f'  [green]✓ Final model saved as best.pt[/green]')
            continue

        # Monitor berdasarkan F1 (lebih tepat untuk imbalanced data)
        if monitor == 'f1':
            improved = val_metrics['f1'] > best_val_f1
            if improved:
                best_val_f1 = val_metrics['f1']
        else:
            improved = val_loss < best_val_f1 if best_val_f1 > 0 else True
            if epoch == 1:
                best_val_f1 = val_loss

        if improved:
            patience = 0
            torch.save(state, ckpt_dir / 'best.pt')
            console.print(f'  [green]✓ Best model saved (f1={val_metrics["f1"]:.4f})[/green]')
        else:
            patience += 1
            console.print(f'  [yellow]No improvement ({patience}/{cfg["train"]["early_stopping_patience"]})[/yellow]')
            if patience >= cfg['train']['early_stopping_patience']:
                console.print('[bold yellow]Early stopping triggered.[/bold yellow]')
                break


    console.print(f'[bold green]Training complete. Best val F1: {best_val_f1:.4f}[/bold green]')
    return TrainArtifacts(model=model, device=device)


def validate(model: BeatmapModel, loader: DataLoader, device: torch.device, cfg: Dict, pos_weight: torch.Tensor):
    model.eval()
    losses_accum = []
    metrics_accum = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['mel'], batch['attention_mask'])
            losses = compute_losses(outputs, batch, cfg, pos_weight)
            metrics = frame_classification_metrics(
                outputs['event_logits'].detach().cpu(),
                outputs['lane_logits'].detach().cpu(),
                batch['event'].detach().cpu(),
                batch['lane'].detach().cpu(),
                threshold=cfg['eval']['event_threshold'],
            )
            losses_accum.append(float(losses['loss'].item()))
            metrics_accum.append(metrics)
    mean_loss = sum(losses_accum) / max(1, len(losses_accum))
    mean_metrics = {
        k: sum(m[k] for m in metrics_accum) / max(1, len(metrics_accum))
        for k in metrics_accum[0]
    }
    return mean_loss, mean_metrics

#!/usr/bin/env python3
"""
compare_mp3_wav.py – Perbandingan MP3 vs WAV untuk skripsi BeatNusa.

Menggunakan pipeline aktual project:
  audio_service.py → ai_service._direct_inference → postprocess_events

Tidak mengubah kode utama, database, checkpoint, atau konfigurasi.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Ensure src/ is importable ──
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import librosa
import soundfile as sf

from beatbert.configs import load_config
from beatbert.models.beatmap_model import BeatmapModel
from beatbert.utils.audio import (
    load_audio,
    compute_log_mel,
    compute_rhythm_guides,
    extract_audio_features,
    frame_times_ms,
    AudioFeatures,
)
from beatbert.inference.predictor import _run_chunked_inference
from beatbert.inference.postprocess import postprocess_events
from beatbert.utils.seed import set_seed

# ── Matplotlib backend ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr


# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════
log = logging.getLogger("compare")


def setup_logging(output_dir: Path):
    log_path = output_dir / "execution_log.txt"
    handler_file = logging.FileHandler(str(log_path), encoding="utf-8")
    handler_file.setLevel(logging.DEBUG)
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler_file.setFormatter(fmt)
    handler_console.setFormatter(fmt)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_file)
    log.addHandler(handler_console)


# ═══════════════════════════════════════════════════════════════
# Environment / system info
# ═══════════════════════════════════════════════════════════════

def gather_environment(checkpoint_path: str, config_path: str, device: torch.device) -> Dict:
    info: Dict[str, Any] = {}
    info["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["os"] = f"{platform.system()} {platform.release()} {platform.version()}"
    info["cpu"] = platform.processor() or "tidak tersedia"
    # Try to get a more descriptive CPU name on Windows
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output("wmic cpu get name", shell=True, text=True)
            lines = [l.strip() for l in out.strip().split("\n") if l.strip() and l.strip().lower() != "name"]
            if lines:
                info["cpu"] = lines[0]
    except Exception:
        pass
    info["gpu"] = "tidak tersedia"
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    info["device_used"] = str(device)
    info["python_version"] = sys.version
    info["pytorch_version"] = torch.__version__
    info["librosa_version"] = librosa.__version__
    info["checkpoint_path"] = checkpoint_path
    info["config_path"] = config_path
    # Try to read epoch from checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        info["checkpoint_epoch"] = ckpt.get("epoch", "tidak tersedia")
    except Exception:
        info["checkpoint_epoch"] = "tidak tersedia"
    return info


# ═══════════════════════════════════════════════════════════════
# File validation helpers
# ═══════════════════════════════════════════════════════════════

def get_file_info(path: str) -> Dict[str, Any]:
    """Gather file-level metadata (no decoding)."""
    p = Path(path)
    info: Dict[str, Any] = {}
    info["filename"] = p.name
    info["format"] = p.suffix.lstrip(".").upper()
    info["size_bytes"] = p.stat().st_size
    info["size_mb"] = round(p.stat().st_size / (1024 * 1024), 4)

    # Use soundfile for WAV metadata
    ext = p.suffix.lower()
    if ext == ".wav":
        try:
            sf_info = sf.info(str(p))
            info["duration_seconds"] = sf_info.duration
            info["sample_rate"] = sf_info.samplerate
            info["channels"] = sf_info.channels
            info["codec"] = sf_info.subtype
            info["bitrate_kbps"] = "tidak tersedia (WAV)"
            info["bit_depth"] = sf_info.subtype  # e.g. PCM_16
        except Exception as e:
            info["duration_seconds"] = "error: " + str(e)
    elif ext == ".mp3":
        # Use librosa to get duration, mutagen for metadata
        try:
            y_tmp, sr_tmp = librosa.load(str(p), sr=None, mono=False, duration=None)
            if y_tmp.ndim == 1:
                info["channels"] = 1
            else:
                info["channels"] = y_tmp.shape[0]
            info["sample_rate"] = sr_tmp
            info["duration_seconds"] = len(y_tmp) / sr_tmp if y_tmp.ndim == 1 else y_tmp.shape[1] / sr_tmp
        except Exception as e:
            info["duration_seconds"] = "error: " + str(e)
        info["codec"] = "MP3"
        info["bit_depth"] = "tidak tersedia (MP3)"
        # Try mutagen for bitrate
        try:
            from mutagen.mp3 import MP3
            m = MP3(str(p))
            info["bitrate_kbps"] = round(m.info.bitrate / 1000)
        except ImportError:
            # Fallback: estimate from file size / duration
            try:
                dur = info.get("duration_seconds", 0)
                if isinstance(dur, (int, float)) and dur > 0:
                    info["bitrate_kbps"] = round(info["size_bytes"] * 8 / dur / 1000)
                else:
                    info["bitrate_kbps"] = "tidak tersedia"
            except Exception:
                info["bitrate_kbps"] = "tidak tersedia"
        except Exception:
            info["bitrate_kbps"] = "tidak tersedia"
    return info


# ═══════════════════════════════════════════════════════════════
# Backend file handling (simulates audio_service.py)
# ═══════════════════════════════════════════════════════════════

ALLOWED_FORMATS = {".mp3", ".wav", ".ogg"}


def simulate_backend_validation(path: str) -> Dict[str, Any]:
    """Simulate validate_upload() from audio_service.py."""
    p = Path(path)
    result: Dict[str, Any] = {}
    ext = p.suffix.lower()
    result["extension"] = ext
    result["is_allowed_format"] = ext in ALLOWED_FORMATS
    result["file_size_bytes"] = p.stat().st_size
    # Backend uses settings.MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
    max_bytes = 10 * 1024 * 1024
    result["max_upload_bytes"] = max_bytes
    result["size_within_limit"] = p.stat().st_size <= max_bytes
    result["validation_passed"] = result["is_allowed_format"] and result["size_within_limit"]
    return result


def simulate_backend_save(src_path: str, tmp_dir: str) -> Dict[str, Any]:
    """
    Simulate save_audio_file() from audio_service.py.
    For non-WAV: transcode to WAV using librosa.load(sr=None, mono=False) + soundfile.write PCM_16.
    For WAV: just copy (rename in backend).
    """
    p = Path(src_path)
    ext = p.suffix.lower()
    result: Dict[str, Any] = {}
    result["original_format"] = ext.lstrip(".")

    if ext != ".wav":
        # Transcode path (same as backend)
        t0 = time.perf_counter()
        y, sr = librosa.load(str(p), sr=None, mono=False)
        target_path = Path(tmp_dir) / (p.stem + "_transcoded.wav")
        sf.write(str(target_path), y.T if y.ndim > 1 else y, sr, subtype="PCM_16")
        elapsed = time.perf_counter() - t0

        result["transcoded"] = True
        result["transcoded_format"] = "wav"
        result["transcoded_path"] = str(target_path)
        result["transcoded_size_bytes"] = target_path.stat().st_size
        result["transcoding_time_s"] = elapsed
        # Info before / after
        result["before_sr"] = sr
        result["before_channels"] = y.shape[0] if y.ndim > 1 else 1
        # Read back to check
        sf_info = sf.info(str(target_path))
        result["after_sr"] = sf_info.samplerate
        result["after_channels"] = sf_info.channels
        result["after_subtype"] = sf_info.subtype
    else:
        t0 = time.perf_counter()
        target_path = Path(tmp_dir) / ("backend_" + p.name)
        if str(Path(p).resolve()) == str(target_path.resolve()):
            target_path = Path(tmp_dir) / ("backend_copy_" + p.name)
        shutil.copy2(str(p), str(target_path))
        elapsed = time.perf_counter() - t0
        result["transcoded"] = False
        result["transcoded_path"] = str(target_path)
        result["handling_time_s"] = elapsed
        result["note"] = "File sudah WAV, hanya di-copy (rename di backend asli)"
        sf_info = sf.info(str(target_path))
        result["after_sr"] = sf_info.samplerate
        result["after_channels"] = sf_info.channels
        result["after_subtype"] = sf_info.subtype

    return result


# ═══════════════════════════════════════════════════════════════
# Full pipeline runner
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    audio_path: str,
    model: BeatmapModel,
    cfg: Dict,
    device: torch.device,
    label: str,
) -> Dict[str, Any]:
    """
    Run the full inference pipeline on one audio file, matching ai_service._direct_inference exactly.
    Returns timing + intermediate results.
    """
    timings: Dict[str, float] = {}
    results: Dict[str, Any] = {"label": label, "audio_path": audio_path}

    pipeline_start = time.perf_counter()

    # ── 1. Load audio (same as load_audio in utils/audio.py) ──
    t0 = time.perf_counter()
    y, sr = load_audio(audio_path, cfg)
    timings["audio_load_s"] = time.perf_counter() - t0
    results["waveform_samples"] = len(y)
    results["waveform_sr"] = sr
    results["waveform_channels"] = 1  # mono enforced by config
    results["waveform_duration_s"] = len(y) / sr

    # ── 2. Compute log-mel spectrogram ──
    t0 = time.perf_counter()
    mel_raw = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        win_length=cfg["audio"]["win_length"],
        n_mels=cfg["audio"]["n_mels"],
        fmin=cfg["audio"]["fmin"],
        fmax=cfg["audio"]["fmax"],
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_raw, ref=np.max, top_db=cfg["audio"].get("top_db", 80))
    # Stats BEFORE normalization
    results["mel_pre_norm_mean"] = float(mel_db.mean())
    results["mel_pre_norm_std"] = float(mel_db.std())
    results["mel_min_pre_norm"] = float(mel_db.min())
    results["mel_max_pre_norm"] = float(mel_db.max())
    # Normalize (same as compute_log_mel)
    mel = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel = mel.astype(np.float32)
    timings["mel_spectrogram_s"] = time.perf_counter() - t0

    results["mel_shape"] = list(mel.shape)
    results["mel_time_frames"] = mel.shape[1]
    results["mel_post_norm_mean"] = float(mel.mean())
    results["mel_post_norm_std"] = float(mel.std())
    results["mel_min_post_norm"] = float(mel.min())
    results["mel_max_post_norm"] = float(mel.max())

    # ── 3. Rhythm guides ──
    t0 = time.perf_counter()
    onset_ms, beat_ms, bpm = compute_rhythm_guides(y, sr, cfg)
    timings["rhythm_guides_s"] = time.perf_counter() - t0
    results["num_onsets"] = len(onset_ms)
    results["num_beats"] = len(beat_ms)
    results["bpm"] = bpm

    # Total preprocessing time
    timings["preprocessing_total_s"] = (
        timings["audio_load_s"] + timings["mel_spectrogram_s"] + timings["rhythm_guides_s"]
    )

    # ── 4. Model inference ──
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    event_prob, lane_prob = _run_chunked_inference(model, mel, cfg, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["inference_s"] = time.perf_counter() - t0

    lane_pred = lane_prob.argmax(axis=-1)
    results["pred_frames"] = len(event_prob)
    results["event_prob_min"] = float(event_prob.min())
    results["event_prob_max"] = float(event_prob.max())
    results["event_prob_mean"] = float(event_prob.mean())
    results["event_prob_std"] = float(event_prob.std())

    eval_threshold = cfg["eval"]["event_threshold"]
    pp_threshold = cfg["postprocess"]["event_threshold"]
    results["frames_above_eval_threshold"] = int((event_prob >= eval_threshold).sum())
    results["frames_above_pp_threshold"] = int((event_prob >= pp_threshold).sum())

    # Lane distribution before post-processing
    f_ms = frame_times_ms(mel.shape[1], cfg)
    idxs_above = np.where(event_prob >= pp_threshold)[0]
    lane_dist_pre = Counter(int(lane_pred[i]) for i in idxs_above)
    results["lane_distribution_pre_pp"] = dict(sorted(lane_dist_pre.items()))
    results["num_candidates_pre_pp"] = len(idxs_above)

    # ── 5. Build raw_events (same as ai_service) ──
    t0 = time.perf_counter()
    raw_events: List[Dict] = []
    for idx in idxs_above:
        raw_events.append({
            "time_ms": float(f_ms[idx]),
            "lane": int(lane_pred[idx]),
            "confidence": float(event_prob[idx]),
        })

    # ── 6. Post-processing ──
    final_events = postprocess_events(raw_events, onset_ms, beat_ms, cfg)
    timings["postprocessing_s"] = time.perf_counter() - t0

    # ── 7. Build beatmap JSON (same as ai_service) ──
    t0 = time.perf_counter()
    notes = [
        {
            "time_ms": int(round(e["time_ms"])),
            "lane": int(e["lane"]),
            "type": "tap",
            "length_ms": 0,
        }
        for e in final_events
    ]
    lane_count = int(cfg["model"]["num_lanes"])
    duration_ms = len(y) / sr * 1000.0
    beatmap = {
        "bpm": bpm,
        "duration_ms": duration_ms,
        "offset_ms": 0,
        "lane_count": lane_count,
        "notes": notes,
        "note_count": len(notes),
    }
    timings["beatmap_json_build_s"] = time.perf_counter() - t0

    timings["pipeline_total_s"] = time.perf_counter() - pipeline_start

    # Post-processing stats
    results["pp_event_threshold"] = pp_threshold
    results["pp_min_gap_ms"] = cfg["postprocess"]["min_gap_ms"]
    results["pp_same_lane_min_gap_ms"] = cfg["postprocess"]["same_lane_min_gap_ms"]
    results["pp_onset_snap_tolerance_ms"] = cfg["postprocess"]["onset_snap_tolerance_ms"]
    results["pp_beat_snap_tolerance_ms"] = cfg["postprocess"]["beat_snap_tolerance_ms"]
    results["pp_max_density_nps"] = cfg["postprocess"]["max_density_notes_per_second"]
    results["num_notes_after_pp"] = len(notes)
    results["notes_removed"] = len(raw_events) - len(notes)
    results["notes_removed_pct"] = (
        round((len(raw_events) - len(notes)) / max(len(raw_events), 1) * 100, 2)
    )
    results["beatmap_duration_ms"] = duration_ms
    results["note_density_per_s"] = round(len(notes) / max(duration_ms / 1000, 1), 4)
    lane_dist_final = Counter(n["lane"] for n in notes)
    results["lane_counts"] = {i: lane_dist_final.get(i, 0) for i in range(lane_count)}
    results["beatmap_json_size_bytes"] = len(json.dumps(beatmap).encode("utf-8"))
    results["beatmap_success"] = True

    return {
        "timings": timings,
        "results": results,
        "mel": mel,
        "mel_db_unnorm": mel_db,
        "event_prob": event_prob,
        "lane_prob": lane_prob,
        "beatmap": beatmap,
        "notes": notes,
        "onset_ms": onset_ms,
        "beat_ms": beat_ms,
    }


# ═══════════════════════════════════════════════════════════════
# Note matching
# ═══════════════════════════════════════════════════════════════

def match_notes(
    notes_a: List[Dict], notes_b: List[Dict], tolerance_ms: float, require_same_lane: bool
) -> Dict[str, Any]:
    """
    One-to-one greedy matching: pair notes with smallest time difference first.
    """
    pairs = []
    for i, a in enumerate(notes_a):
        for j, b in enumerate(notes_b):
            dt = abs(a["time_ms"] - b["time_ms"])
            if dt <= tolerance_ms:
                if require_same_lane and a["lane"] != b["lane"]:
                    continue
                pairs.append((dt, i, j))
    pairs.sort(key=lambda x: x[0])
    used_a, used_b = set(), set()
    matched = []
    for dt, i, j in pairs:
        if i not in used_a and j not in used_b:
            matched.append((i, j, dt))
            used_a.add(i)
            used_b.add(j)

    unmatched_a = [i for i in range(len(notes_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(notes_b)) if j not in used_b]
    dts = [m[2] for m in matched]

    result: Dict[str, Any] = {
        "tolerance_ms": tolerance_ms,
        "require_same_lane": require_same_lane,
        "num_matched": len(matched),
        "num_unmatched_a": len(unmatched_a),
        "num_unmatched_b": len(unmatched_b),
    }
    if dts:
        result["mean_dt_ms"] = float(np.mean(dts))
        result["median_dt_ms"] = float(np.median(dts))
        result["max_dt_ms"] = float(np.max(dts))
    else:
        result["mean_dt_ms"] = None
        result["median_dt_ms"] = None
        result["max_dt_ms"] = None
    return result


# ═══════════════════════════════════════════════════════════════
# Spectrogram similarity
# ═══════════════════════════════════════════════════════════════

def spectrogram_similarity(mel_a: np.ndarray, mel_b: np.ndarray) -> Dict[str, Any]:
    """Compare two spectrograms, aligning to the shorter one."""
    min_frames = min(mel_a.shape[1], mel_b.shape[1])
    a = mel_a[:, :min_frames].flatten()
    b = mel_b[:, :min_frames].flatten()
    mae = float(np.mean(np.abs(a - b)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    try:
        corr, _ = pearsonr(a, b)
        corr = float(corr)
    except Exception:
        corr = None
    try:
        cos_sim = 1.0 - cosine_distance(a, b)
        cos_sim = float(cos_sim)
    except Exception:
        cos_sim = None

    return {
        "frames_compared": min_frames,
        "frames_mp3": mel_a.shape[1],
        "frames_wav": mel_b.shape[1],
        "extra_frames_mp3": max(0, mel_a.shape[1] - min_frames),
        "extra_frames_wav": max(0, mel_b.shape[1] - min_frames),
        "mae": mae,
        "rmse": rmse,
        "pearson_correlation": corr,
        "cosine_similarity": cos_sim,
    }


# ═══════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════

def plot_spectrogram(mel: np.ndarray, title: str, save_path: str, cfg: Dict):
    fig, ax = plt.subplots(figsize=(14, 5))
    hop = cfg["audio"]["hop_length"]
    sr = cfg["audio"]["sample_rate"]
    img = ax.imshow(
        mel, aspect="auto", origin="lower", cmap="magma",
        extent=[0, mel.shape[1] * hop / sr, 0, cfg["audio"]["n_mels"]],
    )
    ax.set_xlabel("Waktu (detik)", fontsize=12)
    ax.set_ylabel("Mel Band", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.colorbar(img, ax=ax, label="Amplitudo (dB, normalized)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram_comparison(mel_mp3: np.ndarray, mel_wav: np.ndarray, save_path: str, cfg: Dict):
    hop = cfg["audio"]["hop_length"]
    sr = cfg["audio"]["sample_rate"]
    vmin = min(mel_mp3.min(), mel_wav.min())
    vmax = max(mel_mp3.max(), mel_wav.max())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, mel, title in [
        (axes[0], mel_mp3, "Log-Mel Spectrogram – MP3"),
        (axes[1], mel_wav, "Log-Mel Spectrogram – WAV"),
    ]:
        img = ax.imshow(
            mel, aspect="auto", origin="lower", cmap="magma",
            extent=[0, mel.shape[1] * hop / sr, 0, cfg["audio"]["n_mels"]],
            vmin=vmin, vmax=vmax,
        )
        ax.set_ylabel("Mel Band", fontsize=11)
        ax.set_title(title, fontsize=13)
        fig.colorbar(img, ax=ax, label="Amplitudo")
    axes[1].set_xlabel("Waktu (detik)", fontsize=12)
    plt.suptitle("Perbandingan Spectrogram MP3 vs WAV", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram_difference(mel_mp3: np.ndarray, mel_wav: np.ndarray, save_path: str, cfg: Dict):
    hop = cfg["audio"]["hop_length"]
    sr = cfg["audio"]["sample_rate"]
    min_frames = min(mel_mp3.shape[1], mel_wav.shape[1])
    diff = mel_mp3[:, :min_frames] - mel_wav[:, :min_frames]

    fig, ax = plt.subplots(figsize=(14, 5))
    abs_max = max(abs(diff.min()), abs(diff.max()), 1e-6)
    img = ax.imshow(
        diff, aspect="auto", origin="lower", cmap="RdBu_r",
        extent=[0, min_frames * hop / sr, 0, cfg["audio"]["n_mels"]],
        vmin=-abs_max, vmax=abs_max,
    )
    ax.set_xlabel("Waktu (detik)", fontsize=12)
    ax.set_ylabel("Mel Band", fontsize=12)
    ax.set_title(f"Perbedaan Spectrogram (MP3 − WAV) | {min_frames} frame", fontsize=14)
    fig.colorbar(img, ax=ax, label="Selisih Amplitudo")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beatmap_timeline(notes_mp3: List[Dict], notes_wav: List[Dict], save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12"}
    for ax, notes, title in [
        (axes[0], notes_mp3, "Beatmap MP3"),
        (axes[1], notes_wav, "Beatmap WAV"),
    ]:
        for n in notes:
            t = n["time_ms"] / 1000.0
            lane = n["lane"]
            ax.scatter(t, lane, color=colors.get(lane, "gray"), s=12, alpha=0.8)
        ax.set_ylabel("Lane", fontsize=11)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_title(title, fontsize=13)
        ax.grid(axis="x", alpha=0.3)
    axes[1].set_xlabel("Waktu (detik)", fontsize=12)
    plt.suptitle("Perbandingan Timeline Beatmap MP3 vs WAV", fontsize=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lane_distribution(notes_mp3: List[Dict], notes_wav: List[Dict], save_path: str):
    lanes = [0, 1, 2, 3]
    mp3_counts = [sum(1 for n in notes_mp3 if n["lane"] == l) for l in lanes]
    wav_counts = [sum(1 for n in notes_wav if n["lane"] == l) for l in lanes]

    x = np.arange(len(lanes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, mp3_counts, width, label="MP3", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, wav_counts, width, label="WAV", color="#3498db", alpha=0.85)
    ax.set_xlabel("Lane", fontsize=12)
    ax.set_ylabel("Jumlah Note", fontsize=12)
    ax.set_title("Distribusi Lane: MP3 vs WAV", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Lane {l}" for l in lanes])
    ax.legend()
    # Add count labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(str(int(h)), xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timing_histogram(notes_mp3: List[Dict], notes_wav: List[Dict], tolerance_ms: float, save_path: str):
    """Histogram of absolute timestamp differences for matched notes (time only)."""
    match_info = match_notes(notes_mp3, notes_wav, tolerance_ms, require_same_lane=False)
    # Re-run matching to get actual dts for histogram
    pairs = []
    for i, a in enumerate(notes_mp3):
        for j, b in enumerate(notes_wav):
            dt = abs(a["time_ms"] - b["time_ms"])
            if dt <= tolerance_ms:
                pairs.append((dt, i, j))
    pairs.sort(key=lambda x: x[0])
    used_a, used_b = set(), set()
    dts = []
    for dt, i, j in pairs:
        if i not in used_a and j not in used_b:
            dts.append(dt)
            used_a.add(i)
            used_b.add(j)

    fig, ax = plt.subplots(figsize=(10, 5))
    if dts:
        ax.hist(dts, bins=min(50, len(dts)), color="#8e44ad", alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(dts), color="red", linestyle="--", label=f"Rata-rata: {np.mean(dts):.1f} ms")
        ax.axvline(np.median(dts), color="orange", linestyle="--", label=f"Median: {np.median(dts):.1f} ms")
        ax.legend()
    ax.set_xlabel("Selisih Absolut Timestamp (ms)", fontsize=12)
    ax.set_ylabel("Frekuensi", fontsize=12)
    ax.set_title(f"Histogram Selisih Timestamp Note yang Cocok (toleransi {tolerance_ms} ms)", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════

def _fmt_timing_table(mp3_timings: List[Dict], wav_timings: List[Dict], keys: List[str]) -> str:
    """Build markdown timing comparison table with mean, std, min, max."""
    header = "| Tahap | MP3 Rata-rata (s) | MP3 Std | MP3 Min | MP3 Max | WAV Rata-rata (s) | WAV Std | WAV Min | WAV Max |\n"
    sep = "|---|---|---|---|---|---|---|---|---|\n"
    rows = ""
    for key in keys:
        mp3_vals = [t[key] for t in mp3_timings if key in t]
        wav_vals = [t[key] for t in wav_timings if key in t]
        if not mp3_vals or not wav_vals:
            continue
        label = key.replace("_s", "").replace("_", " ").title()
        rows += (
            f"| {label} "
            f"| {np.mean(mp3_vals):.6f} | {np.std(mp3_vals):.6f} | {min(mp3_vals):.6f} | {max(mp3_vals):.6f} "
            f"| {np.mean(wav_vals):.6f} | {np.std(wav_vals):.6f} | {min(wav_vals):.6f} | {max(wav_vals):.6f} |\n"
        )
    return header + sep + rows


def generate_report(
    env: Dict, file_mp3: Dict, file_wav: Dict,
    backend_mp3: Dict, backend_wav: Dict,
    mp3_runs: List[Dict], wav_runs: List[Dict],
    spec_sim: Dict, matching_results: Dict,
    determinism: Dict, warnings_list: List[str],
    output_dir: Path, num_runs: int,
) -> str:
    """Generate the full comparison_report.md in Indonesian."""

    mp3_r = mp3_runs[0]["results"]
    wav_r = wav_runs[0]["results"]

    lines: List[str] = []
    lines.append("# Laporan Perbandingan MP3 vs WAV – BeatNusa\n\n")

    # ── 1. Informasi Pengujian ──
    lines.append("## 1. Informasi Pengujian\n\n")
    lines.append(f"| Parameter | Nilai |\n|---|---|\n")
    lines.append(f"| Tanggal dan waktu | {env['datetime']} |\n")
    lines.append(f"| Sistem operasi | {env['os']} |\n")
    lines.append(f"| CPU | {env['cpu']} |\n")
    lines.append(f"| GPU | {env['gpu']} |\n")
    lines.append(f"| CUDA tersedia | {env['cuda_available']} |\n")
    lines.append(f"| Device digunakan | {env['device_used']} |\n")
    lines.append(f"| Versi Python | {env['python_version'].split()[0]} |\n")
    lines.append(f"| Versi PyTorch | {env['pytorch_version']} |\n")
    lines.append(f"| Versi Librosa | {env['librosa_version']} |\n")
    lines.append(f"| Path checkpoint | {env['checkpoint_path']} |\n")
    lines.append(f"| Epoch checkpoint | {env['checkpoint_epoch']} |\n")
    lines.append(f"| Path konfigurasi | {env['config_path']} |\n")
    lines.append(f"| Jumlah pengulangan | {num_runs} |\n\n")

    # ── 2. Validasi Kesamaan Audio ──
    lines.append("## 2. Validasi Kesamaan Audio\n\n")
    dur_mp3 = file_mp3.get("duration_seconds", 0)
    dur_wav = file_wav.get("duration_seconds", 0)
    dur_diff = abs(dur_mp3 - dur_wav) if isinstance(dur_mp3, (int, float)) and isinstance(dur_wav, (int, float)) else "N/A"
    lines.append(f"- Durasi MP3: {dur_mp3:.4f} detik\n")
    lines.append(f"- Durasi WAV: {dur_wav:.4f} detik\n")
    if isinstance(dur_diff, float):
        lines.append(f"- Selisih durasi: {dur_diff*1000:.2f} ms\n")
        if dur_diff > 0.1:
            lines.append(f"- ⚠️ **Peringatan**: Selisih durasi > 100 ms. Kedua file mungkin tidak sejajar sempurna.\n")
        else:
            lines.append(f"- ✅ Selisih durasi ≤ 100 ms. Kedua file kemungkinan berasal dari lagu yang sama.\n")
    lines.append(f"- Pearson correlation spectrogram: {spec_sim.get('pearson_correlation', 'N/A')}\n")
    lines.append(f"- Cosine similarity spectrogram: {spec_sim.get('cosine_similarity', 'N/A')}\n\n")

    # ── 3. Perbandingan Karakteristik File ──
    lines.append("## 3. Perbandingan Karakteristik File\n\n")
    lines.append("| Karakteristik | MP3 | WAV |\n|---|---|---|\n")
    for key in ["filename", "format", "size_bytes", "size_mb", "duration_seconds",
                 "sample_rate", "channels", "codec", "bitrate_kbps", "bit_depth"]:
        lines.append(f"| {key} | {file_mp3.get(key, 'N/A')} | {file_wav.get(key, 'N/A')} |\n")
    lines.append("\n")

    # ── Backend handling ──
    lines.append("### Penanganan File oleh Backend\n\n")
    lines.append("**MP3:**\n\n")
    for k, v in backend_mp3.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n**WAV:**\n\n")
    for k, v in backend_wav.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n")

    # ── 4. Perbandingan Waktu Pemrosesan ──
    lines.append("## 4. Perbandingan Waktu Pemrosesan\n\n")
    timing_keys = [
        "audio_load_s", "mel_spectrogram_s", "rhythm_guides_s",
        "preprocessing_total_s", "inference_s", "postprocessing_s",
        "beatmap_json_build_s", "pipeline_total_s",
    ]
    mp3_timings = [r["timings"] for r in mp3_runs]
    wav_timings = [r["timings"] for r in wav_runs]
    lines.append(_fmt_timing_table(mp3_timings, wav_timings, timing_keys))
    lines.append("\n")

    # ── 5. Perbandingan Hasil Preprocessing ──
    lines.append("## 5. Perbandingan Hasil Preprocessing\n\n")
    lines.append("| Parameter | MP3 | WAV |\n|---|---|---|\n")
    preproc_keys = [
        "waveform_sr", "waveform_channels", "waveform_samples", "waveform_duration_s",
        "mel_shape", "mel_time_frames",
        "mel_min_pre_norm", "mel_max_pre_norm", "mel_pre_norm_mean", "mel_pre_norm_std",
        "mel_min_post_norm", "mel_max_post_norm", "mel_post_norm_mean", "mel_post_norm_std",
        "num_onsets", "num_beats", "bpm",
    ]
    for key in preproc_keys:
        mv = mp3_r.get(key, "N/A")
        wv = wav_r.get(key, "N/A")
        if isinstance(mv, float):
            mv = f"{mv:.6f}"
        if isinstance(wv, float):
            wv = f"{wv:.6f}"
        lines.append(f"| {key} | {mv} | {wv} |\n")
    lines.append("\n")

    lines.append("### Kemiripan Spectrogram\n\n")
    lines.append("| Metrik | Nilai |\n|---|---|\n")
    for key in ["frames_compared", "extra_frames_mp3", "extra_frames_wav", "mae", "rmse",
                 "pearson_correlation", "cosine_similarity"]:
        v = spec_sim.get(key, "N/A")
        if isinstance(v, float):
            v = f"{v:.8f}"
        lines.append(f"| {key} | {v} |\n")
    lines.append("\n")

    # ── 6. Perbandingan Prediksi Mentah ──
    lines.append("## 6. Perbandingan Prediksi Mentah\n\n")
    lines.append("| Parameter | MP3 | WAV |\n|---|---|---|\n")
    pred_keys = [
        "pred_frames", "event_prob_min", "event_prob_max", "event_prob_mean", "event_prob_std",
        "frames_above_eval_threshold", "frames_above_pp_threshold",
        "num_candidates_pre_pp",
    ]
    for key in pred_keys:
        mv = mp3_r.get(key, "N/A")
        wv = wav_r.get(key, "N/A")
        if isinstance(mv, float):
            mv = f"{mv:.6f}"
        if isinstance(wv, float):
            wv = f"{wv:.6f}"
        lines.append(f"| {key} | {mv} | {wv} |\n")
    lines.append("\n")
    lines.append("### Distribusi Lane Sebelum Post-Processing\n\n")
    lines.append("| Lane | MP3 | WAV |\n|---|---|---|\n")
    for lane in range(4):
        lines.append(f"| Lane {lane} | {mp3_r['lane_distribution_pre_pp'].get(lane, 0)} | {wav_r['lane_distribution_pre_pp'].get(lane, 0)} |\n")
    lines.append("\n")

    # Sample frames
    lines.append("### Contoh Frame Prediksi (5 frame pertama dengan probabilitas tertinggi)\n\n")
    ep_mp3 = mp3_runs[0]["event_prob"]
    ep_wav = wav_runs[0]["event_prob"]
    top5_mp3 = np.argsort(ep_mp3)[-5:][::-1]
    top5_wav = np.argsort(ep_wav)[-5:][::-1]
    lines.append("**MP3:**\n\n| Frame | Probabilitas Event | Lane Prediksi |\n|---|---|---|\n")
    lp_mp3 = mp3_runs[0]["lane_prob"]
    for idx in top5_mp3:
        lines.append(f"| {idx} | {ep_mp3[idx]:.6f} | {lp_mp3[idx].argmax()} |\n")
    lines.append("\n**WAV:**\n\n| Frame | Probabilitas Event | Lane Prediksi |\n|---|---|---|\n")
    lp_wav = wav_runs[0]["lane_prob"]
    for idx in top5_wav:
        lines.append(f"| {idx} | {ep_wav[idx]:.6f} | {lp_wav[idx].argmax()} |\n")
    lines.append("\n")

    # ── 7. Perbandingan Beatmap Akhir ──
    lines.append("## 7. Perbandingan Beatmap Akhir\n\n")
    lines.append("| Parameter | MP3 | WAV |\n|---|---|---|\n")
    bm_keys = [
        "pp_event_threshold", "pp_min_gap_ms", "pp_same_lane_min_gap_ms",
        "pp_onset_snap_tolerance_ms", "pp_beat_snap_tolerance_ms", "pp_max_density_nps",
        "num_candidates_pre_pp", "num_notes_after_pp", "notes_removed", "notes_removed_pct",
        "beatmap_duration_ms", "note_density_per_s",
        "beatmap_json_size_bytes", "beatmap_success",
    ]
    for key in bm_keys:
        lines.append(f"| {key} | {mp3_r.get(key, 'N/A')} | {wav_r.get(key, 'N/A')} |\n")
    lines.append("\n")
    lines.append("### Distribusi Lane Setelah Post-Processing\n\n")
    lines.append("| Lane | MP3 | WAV |\n|---|---|---|\n")
    for lane in range(4):
        lines.append(f"| Lane {lane} | {mp3_r['lane_counts'].get(lane, 0)} | {wav_r['lane_counts'].get(lane, 0)} |\n")
    lines.append("\n")

    # ── 8. Kemiripan Beatmap ──
    lines.append("## 8. Kemiripan Beatmap MP3 dan WAV\n\n")
    for tol in [25, 50, 100]:
        lines.append(f"### Toleransi {tol} ms\n\n")
        key_time = f"time_only_{tol}ms"
        key_lane = f"time_and_lane_{tol}ms"
        mt = matching_results.get(key_time, {})
        ml = matching_results.get(key_lane, {})
        lines.append("| Metrik | Waktu Saja | Waktu + Lane |\n|---|---|---|\n")
        lines.append(f"| Jumlah note MP3 | {len(mp3_runs[0]['notes'])} | {len(mp3_runs[0]['notes'])} |\n")
        lines.append(f"| Jumlah note WAV | {len(wav_runs[0]['notes'])} | {len(wav_runs[0]['notes'])} |\n")
        lines.append(f"| Note cocok | {mt.get('num_matched', 'N/A')} | {ml.get('num_matched', 'N/A')} |\n")
        lines.append(f"| Note MP3 tanpa pasangan | {mt.get('num_unmatched_a', 'N/A')} | {ml.get('num_unmatched_a', 'N/A')} |\n")
        lines.append(f"| Note WAV tanpa pasangan | {mt.get('num_unmatched_b', 'N/A')} | {ml.get('num_unmatched_b', 'N/A')} |\n")
        lines.append(f"| Rata-rata selisih timestamp (ms) | {mt.get('mean_dt_ms', 'N/A')} | {ml.get('mean_dt_ms', 'N/A')} |\n")
        lines.append(f"| Median selisih timestamp (ms) | {mt.get('median_dt_ms', 'N/A')} | {ml.get('median_dt_ms', 'N/A')} |\n")
        lines.append(f"| Maks selisih timestamp (ms) | {mt.get('max_dt_ms', 'N/A')} | {ml.get('max_dt_ms', 'N/A')} |\n")
        lines.append("\n")

    # Lane agreement on matched notes
    mt_100 = matching_results.get("time_only_100ms", {})
    ml_100 = matching_results.get("time_and_lane_100ms", {})
    if mt_100.get("num_matched", 0) > 0:
        lane_agreement = ml_100.get("num_matched", 0) / mt_100["num_matched"] * 100
        lines.append(f"**Tingkat kesesuaian lane** pada note yang cocok (100 ms): {lane_agreement:.2f}%\n\n")

    # Lane distribution similarity (cosine)
    mp3_lane_vec = [mp3_r["lane_counts"].get(i, 0) for i in range(4)]
    wav_lane_vec = [wav_r["lane_counts"].get(i, 0) for i in range(4)]
    try:
        lane_cos_sim = 1.0 - cosine_distance(mp3_lane_vec, wav_lane_vec)
        lines.append(f"**Kemiripan distribusi lane** (cosine similarity): {lane_cos_sim:.6f}\n\n")
    except Exception:
        lines.append("**Kemiripan distribusi lane**: tidak dapat dihitung\n\n")

    # Density difference
    density_diff = mp3_r["note_density_per_s"] - wav_r["note_density_per_s"]
    lines.append(f"**Perbedaan kepadatan note**: {density_diff:.4f} notes/detik\n\n")

    # Jaccard similarity
    total_notes = len(mp3_runs[0]["notes"]) + len(wav_runs[0]["notes"]) - mt_100.get("num_matched", 0)
    if total_notes > 0:
        jaccard = mt_100.get("num_matched", 0) / total_notes
        lines.append(f"**Jaccard similarity** (100 ms, waktu saja): {jaccard:.6f}\n\n")

    # ── 9. Determinisme ──
    lines.append("## 9. Determinisme Hasil\n\n")
    lines.append(f"- Note count konsisten antar-run (MP3): {determinism['mp3_note_count_consistent']}\n")
    lines.append(f"- Note count konsisten antar-run (WAV): {determinism['wav_note_count_consistent']}\n")
    lines.append(f"- Timestamp konsisten antar-run (MP3): {determinism['mp3_timestamps_consistent']}\n")
    lines.append(f"- Timestamp konsisten antar-run (WAV): {determinism['wav_timestamps_consistent']}\n")
    lines.append(f"- Lane konsisten antar-run (MP3): {determinism['mp3_lanes_consistent']}\n")
    lines.append(f"- Lane konsisten antar-run (WAV): {determinism['wav_lanes_consistent']}\n")
    if determinism.get("issues"):
        lines.append(f"\n⚠️ **Masalah Determinisme**: {determinism['issues']}\n")
    else:
        lines.append("\n✅ Semua hasil deterministik. Perubahan hanya terjadi pada waktu eksekusi.\n")
    lines.append("\n")

    # ── 10. Analisis ──
    lines.append("## 10. Analisis\n\n")
    # Format lebih kecil
    if file_mp3["size_bytes"] < file_wav["size_bytes"]:
        lines.append(f"- **Format lebih kecil**: MP3 ({file_mp3['size_mb']} MB vs {file_wav['size_mb']} MB, "
                      f"rasio {file_mp3['size_bytes']/file_wav['size_bytes']*100:.1f}%)\n")
    else:
        lines.append(f"- **Format lebih kecil**: WAV ({file_wav['size_mb']} MB vs {file_mp3['size_mb']} MB)\n")

    # Format lebih cepat
    mp3_avg_pipeline = np.mean([t["pipeline_total_s"] for t in mp3_timings])
    wav_avg_pipeline = np.mean([t["pipeline_total_s"] for t in wav_timings])
    if mp3_avg_pipeline < wav_avg_pipeline:
        lines.append(f"- **Format lebih cepat diproses**: MP3 ({mp3_avg_pipeline:.4f}s vs {wav_avg_pipeline:.4f}s)\n")
    else:
        lines.append(f"- **Format lebih cepat diproses**: WAV ({wav_avg_pipeline:.4f}s vs {mp3_avg_pipeline:.4f}s)\n")

    # Kemiripan preprocessing
    corr = spec_sim.get("pearson_correlation")
    if corr is not None:
        lines.append(f"- **Kemiripan preprocessing**: Pearson correlation spectrogram = {corr:.6f}. "
                      f"MAE = {spec_sim['mae']:.6f}, RMSE = {spec_sim['rmse']:.6f}\n")

    # Kemiripan beatmap
    mt50 = matching_results.get("time_only_50ms", {})
    if mt50.get("num_matched") is not None and len(mp3_runs[0]["notes"]) > 0:
        pct = mt50["num_matched"] / max(len(mp3_runs[0]["notes"]), len(wav_runs[0]["notes"])) * 100
        lines.append(f"- **Kemiripan beatmap akhir** (50 ms): {mt50['num_matched']} note cocok dari "
                      f"{len(mp3_runs[0]['notes'])} (MP3) dan {len(wav_runs[0]['notes'])} (WAV) = {pct:.1f}%\n")

    # Kompresi MP3
    lines.append(f"- **Pengaruh kompresi MP3**: Spectrogram RMSE = {spec_sim.get('rmse', 'N/A')}, "
                  f"menunjukkan {'perbedaan kecil' if spec_sim.get('rmse', 1) < 0.5 else 'perbedaan yang terukur'} "
                  f"pada level fitur akustik.\n")

    # Keduanya menghasilkan beatmap?
    lines.append(f"- **Kedua format menghasilkan beatmap**: MP3 = {mp3_r['beatmap_success']}, WAV = {wav_r['beatmap_success']}. "
                  f"Kedua format tetap menghasilkan beatmap yang dapat digunakan.\n")

    # Keterbatasan
    lines.append("\n### Keterbatasan Pengujian\n\n")
    lines.append("- Pengujian hanya menggunakan satu pasang lagu.\n")
    lines.append("- Tidak dilakukan uji statistik formal (t-test, dll) untuk jumlah pengulangan yang kecil.\n")
    lines.append("- Perbedaan waktu pemrosesan dapat dipengaruhi oleh beban sistem lain.\n")
    lines.append("- Encoding MP3 dapat bervariasi berdasarkan encoder dan bitrate yang digunakan.\n\n")

    # ── 11. Kesimpulan Faktual ──
    lines.append("## 11. Kesimpulan Faktual\n\n")
    lines.append("Berdasarkan hasil pengujian:\n\n")
    lines.append(f"1. File MP3 berukuran {file_mp3['size_mb']} MB, sedangkan WAV berukuran {file_wav['size_mb']} MB.\n")
    lines.append(f"2. Rata-rata waktu pipeline MP3 = {mp3_avg_pipeline:.4f}s, WAV = {wav_avg_pipeline:.4f}s.\n")
    lines.append(f"3. Spectrogram menunjukkan Pearson correlation = {spec_sim.get('pearson_correlation', 'N/A')}.\n")
    lines.append(f"4. MP3 menghasilkan {mp3_r['num_notes_after_pp']} note, WAV menghasilkan {wav_r['num_notes_after_pp']} note.\n")
    if mt50.get("num_matched") is not None:
        lines.append(f"5. Pada toleransi 50 ms, {mt50['num_matched']} note cocok berdasarkan waktu.\n")
    lines.append(f"6. Hasil deterministik antar-run: {'Ya' if not determinism.get('issues') else 'Tidak – ' + determinism.get('issues', '')}.\n")
    lines.append(f"7. Kedua format berhasil menghasilkan beatmap JSON yang valid.\n\n")

    # Warnings
    if warnings_list:
        lines.append("## Peringatan\n\n")
        for w in warnings_list:
            lines.append(f"- ⚠️ {w}\n")
        lines.append("\n")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Perbandingan MP3 vs WAV untuk skripsi BeatNusa")
    parser.add_argument("--mp3", required=True, help="Path file MP3")
    parser.add_argument("--wav", required=True, help="Path file WAV")
    parser.add_argument("--runs", type=int, default=3, help="Jumlah pengulangan (default: 3)")
    parser.add_argument("--output-dir", default="artifacts/mp3_wav_comparison", help="Direktori output")
    parser.add_argument("--config", default=None, help="Path konfigurasi (default: configs/local.yaml)")
    parser.add_argument("--checkpoint", default=None, help="Path checkpoint (default: checkpoints/best.pt)")
    args = parser.parse_args()

    # ── Resolve paths ──
    project_root = ROOT
    config_path = args.config or str(project_root / "configs" / "local.yaml")
    checkpoint_path = args.checkpoint or str(project_root / "checkpoints" / "best.pt")
    mp3_path = str(Path(args.mp3).resolve())
    wav_path = str(Path(args.wav).resolve())
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    num_runs = args.runs

    setup_logging(output_dir)
    log.info("=" * 70)
    log.info("Perbandingan MP3 vs WAV – BeatNusa")
    log.info("=" * 70)

    # ── Validate input files ──
    assert Path(mp3_path).exists(), f"MP3 tidak ditemukan: {mp3_path}"
    assert Path(wav_path).exists(), f"WAV tidak ditemukan: {wav_path}"
    assert Path(config_path).exists(), f"Config tidak ditemukan: {config_path}"
    assert Path(checkpoint_path).exists(), f"Checkpoint tidak ditemukan: {checkpoint_path}"
    log.info(f"MP3: {mp3_path}")
    log.info(f"WAV: {wav_path}")
    log.info(f"Config: {config_path}")
    log.info(f"Checkpoint: {checkpoint_path}")
    log.info(f"Output: {output_dir}")
    log.info(f"Runs: {num_runs}")

    warnings_list: List[str] = []

    # ── Load config & model ──
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info("Memuat model...")
    model = BeatmapModel(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Model berhasil dimuat.")

    # ── Environment ──
    env = gather_environment(checkpoint_path, config_path, device)
    env["num_runs"] = num_runs
    log.info(f"Environment: {json.dumps(env, indent=2, default=str)}")

    # ── File info ──
    log.info("Mengumpulkan informasi file...")
    file_mp3 = get_file_info(mp3_path)
    file_wav = get_file_info(wav_path)
    log.info(f"MP3 info: {json.dumps(file_mp3, indent=2, default=str)}")
    log.info(f"WAV info: {json.dumps(file_wav, indent=2, default=str)}")

    # Duration diff check
    dur_mp3 = file_mp3.get("duration_seconds", 0)
    dur_wav = file_wav.get("duration_seconds", 0)
    if isinstance(dur_mp3, (int, float)) and isinstance(dur_wav, (int, float)):
        dur_diff_ms = abs(dur_mp3 - dur_wav) * 1000
        if dur_diff_ms > 100:
            w = f"Selisih durasi {dur_diff_ms:.2f} ms > 100 ms. File mungkin tidak sejajar sempurna."
            warnings_list.append(w)
            log.warning(w)
    else:
        warnings_list.append("Tidak dapat membandingkan durasi.")

    # ── Backend simulation ──
    log.info("Simulasi penanganan file backend...")
    tmp_dir = tempfile.mkdtemp(prefix="beatnusa_cmp_")
    log.info(f"Direktori sementara: {tmp_dir}")

    # Copy originals to preserve them
    mp3_work = str(Path(tmp_dir) / Path(mp3_path).name)
    wav_work = str(Path(tmp_dir) / Path(wav_path).name)
    shutil.copy2(mp3_path, mp3_work)
    shutil.copy2(wav_path, wav_work)

    backend_mp3_validation = simulate_backend_validation(mp3_work)
    backend_wav_validation = simulate_backend_validation(wav_work)
    log.info(f"Backend validation MP3: {json.dumps(backend_mp3_validation, indent=2)}")
    log.info(f"Backend validation WAV: {json.dumps(backend_wav_validation, indent=2)}")
    if not backend_wav_validation["size_within_limit"]:
        w = (f"File WAV ({file_wav['size_mb']} MB) melebihi batas upload backend "
             f"({backend_wav_validation['max_upload_bytes'] / 1024 / 1024:.0f} MB). "
             f"Pengujian tetap dilanjutkan untuk perbandingan pipeline.")
        warnings_list.append(w)
        log.warning(w)
    if not backend_mp3_validation["size_within_limit"]:
        w = f"File MP3 ({file_mp3['size_mb']} MB) melebihi batas upload backend."
        warnings_list.append(w)
        log.warning(w)

    backend_mp3_save = simulate_backend_save(mp3_work, tmp_dir)
    backend_wav_save = simulate_backend_save(wav_work, tmp_dir)
    log.info(f"Backend save MP3: {json.dumps(backend_mp3_save, indent=2, default=str)}")
    log.info(f"Backend save WAV: {json.dumps(backend_wav_save, indent=2, default=str)}")

    # The audio path that goes into the model pipeline:
    # For MP3: backend transcodes to WAV first, but model pipeline (ai_service) receives
    # the file_path which is the stored WAV path. However, the model's extract_audio_features
    # will load it with librosa at 22050 mono regardless.
    # To faithfully test "what does the pipeline do with MP3 vs WAV",
    # we feed the ORIGINAL files to the model pipeline, because that's what matters
    # for the preprocessing comparison (librosa handles both).
    # BUT the backend always transcodes MP3→WAV before feeding to AI.
    # So we should also note this.

    # Path that goes to model: after backend save
    mp3_model_input = backend_mp3_save["transcoded_path"]  # This is the transcoded WAV
    wav_model_input = backend_wav_save["transcoded_path"]  # This is the copied WAV

    log.info(f"Model input MP3 (after backend): {mp3_model_input}")
    log.info(f"Model input WAV (after backend): {wav_model_input}")

    # ── Warm-up run ──
    log.info("Menjalankan warm-up run (tidak dihitung)...")
    set_seed(seed)
    _ = run_pipeline(wav_model_input, model, cfg, device, "warmup")
    log.info("Warm-up selesai.")

    # ── Benchmark runs ──
    mp3_runs: List[Dict] = []
    wav_runs: List[Dict] = []

    for run_idx in range(num_runs):
        log.info(f"=== Run {run_idx + 1}/{num_runs} ===")

        set_seed(seed)
        log.info(f"  MP3 pipeline...")
        mp3_result = run_pipeline(mp3_model_input, model, cfg, device, f"mp3_run{run_idx}")
        mp3_runs.append(mp3_result)
        log.info(f"  MP3 pipeline time: {mp3_result['timings']['pipeline_total_s']:.4f}s, "
                 f"notes: {mp3_result['results']['num_notes_after_pp']}")

        set_seed(seed)
        log.info(f"  WAV pipeline...")
        wav_result = run_pipeline(wav_model_input, model, cfg, device, f"wav_run{run_idx}")
        wav_runs.append(wav_result)
        log.info(f"  WAV pipeline time: {wav_result['timings']['pipeline_total_s']:.4f}s, "
                 f"notes: {wav_result['results']['num_notes_after_pp']}")

    # ── Determinism check ──
    log.info("Memeriksa determinisme...")
    determinism: Dict[str, Any] = {}
    issues = []

    mp3_note_counts = [r["results"]["num_notes_after_pp"] for r in mp3_runs]
    wav_note_counts = [r["results"]["num_notes_after_pp"] for r in wav_runs]
    determinism["mp3_note_count_consistent"] = len(set(mp3_note_counts)) == 1
    determinism["wav_note_count_consistent"] = len(set(wav_note_counts)) == 1
    if not determinism["mp3_note_count_consistent"]:
        issues.append(f"MP3 note count bervariasi: {mp3_note_counts}")
    if not determinism["wav_note_count_consistent"]:
        issues.append(f"WAV note count bervariasi: {wav_note_counts}")

    # Compare timestamps
    mp3_ts_sets = [tuple(n["time_ms"] for n in r["notes"]) for r in mp3_runs]
    wav_ts_sets = [tuple(n["time_ms"] for n in r["notes"]) for r in wav_runs]
    determinism["mp3_timestamps_consistent"] = len(set(mp3_ts_sets)) == 1
    determinism["wav_timestamps_consistent"] = len(set(wav_ts_sets)) == 1
    if not determinism["mp3_timestamps_consistent"]:
        issues.append("MP3 timestamps bervariasi antar-run")
    if not determinism["wav_timestamps_consistent"]:
        issues.append("WAV timestamps bervariasi antar-run")

    mp3_lane_sets = [tuple(n["lane"] for n in r["notes"]) for r in mp3_runs]
    wav_lane_sets = [tuple(n["lane"] for n in r["notes"]) for r in wav_runs]
    determinism["mp3_lanes_consistent"] = len(set(mp3_lane_sets)) == 1
    determinism["wav_lanes_consistent"] = len(set(wav_lane_sets)) == 1
    if not determinism["mp3_lanes_consistent"]:
        issues.append("MP3 lanes bervariasi antar-run")
    if not determinism["wav_lanes_consistent"]:
        issues.append("WAV lanes bervariasi antar-run")

    determinism["issues"] = "; ".join(issues) if issues else ""
    if issues:
        for iss in issues:
            warnings_list.append(f"Determinisme: {iss}")
    log.info(f"Determinisme: {json.dumps(determinism, indent=2)}")

    # ── Spectrogram similarity (use first run) ──
    log.info("Menghitung kemiripan spectrogram...")
    spec_sim = spectrogram_similarity(mp3_runs[0]["mel"], wav_runs[0]["mel"])
    log.info(f"Spectrogram similarity: {json.dumps(spec_sim, indent=2)}")

    # ── Note matching ──
    log.info("Mencocokkan note beatmap...")
    notes_mp3 = mp3_runs[0]["notes"]
    notes_wav = wav_runs[0]["notes"]
    matching_results: Dict[str, Any] = {}
    for tol in [25, 50, 100]:
        matching_results[f"time_only_{tol}ms"] = match_notes(notes_mp3, notes_wav, tol, False)
        matching_results[f"time_and_lane_{tol}ms"] = match_notes(notes_mp3, notes_wav, tol, True)
    log.info(f"Matching results: {json.dumps(matching_results, indent=2)}")

    # ── Save outputs ──
    log.info("Menyimpan output...")

    # Beatmap JSONs
    with open(output_dir / "beatmap_mp3.json", "w", encoding="utf-8") as f:
        json.dump(mp3_runs[0]["beatmap"], f, indent=2, ensure_ascii=False)
    with open(output_dir / "beatmap_wav.json", "w", encoding="utf-8") as f:
        json.dump(wav_runs[0]["beatmap"], f, indent=2, ensure_ascii=False)

    # Raw predictions
    np.savez(
        str(output_dir / "raw_predictions_mp3.npz"),
        event_prob=mp3_runs[0]["event_prob"],
        lane_prob=mp3_runs[0]["lane_prob"],
    )
    np.savez(
        str(output_dir / "raw_predictions_wav.npz"),
        event_prob=wav_runs[0]["event_prob"],
        lane_prob=wav_runs[0]["lane_prob"],
    )

    # Visualizations
    log.info("Membuat visualisasi...")
    plot_spectrogram(mp3_runs[0]["mel"], "Log-Mel Spectrogram – MP3", str(output_dir / "spectrogram_mp3.png"), cfg)
    plot_spectrogram(wav_runs[0]["mel"], "Log-Mel Spectrogram – WAV", str(output_dir / "spectrogram_wav.png"), cfg)
    plot_spectrogram_comparison(mp3_runs[0]["mel"], wav_runs[0]["mel"], str(output_dir / "spectrogram_comparison.png"), cfg)
    plot_spectrogram_difference(mp3_runs[0]["mel"], wav_runs[0]["mel"], str(output_dir / "spectrogram_difference.png"), cfg)
    plot_beatmap_timeline(notes_mp3, notes_wav, str(output_dir / "beatmap_timeline_comparison.png"))
    plot_lane_distribution(notes_mp3, notes_wav, str(output_dir / "lane_distribution_comparison.png"))
    plot_timing_histogram(notes_mp3, notes_wav, 100.0, str(output_dir / "timing_difference_histogram.png"))

    # ── comparison_metrics.json ──
    log.info("Menyusun comparison_metrics.json...")
    mp3_timings_list = [r["timings"] for r in mp3_runs]
    wav_timings_list = [r["timings"] for r in wav_runs]

    def timing_stats(timings_list, key):
        vals = [t[key] for t in timings_list if key in t]
        if not vals:
            return {}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(min(vals)), "max": float(max(vals))}

    timing_keys = [
        "audio_load_s", "mel_spectrogram_s", "rhythm_guides_s",
        "preprocessing_total_s", "inference_s", "postprocessing_s",
        "beatmap_json_build_s", "pipeline_total_s",
    ]

    metrics = {
        "environment": env,
        "input_files": {"mp3": file_mp3, "wav": file_wav},
        "validation": {"mp3": backend_mp3_validation, "wav": backend_wav_validation},
        "backend_handling": {"mp3": backend_mp3_save, "wav": backend_wav_save},
        "timing": {
            "mp3": {k: timing_stats(mp3_timings_list, k) for k in timing_keys},
            "wav": {k: timing_stats(wav_timings_list, k) for k in timing_keys},
        },
        "preprocessing": {
            "mp3": {k: mp3_runs[0]["results"].get(k) for k in [
                "waveform_sr", "waveform_channels", "waveform_samples", "waveform_duration_s",
                "mel_shape", "mel_time_frames", "mel_pre_norm_mean", "mel_pre_norm_std",
                "mel_min_pre_norm", "mel_max_pre_norm", "mel_post_norm_mean", "mel_post_norm_std",
                "mel_min_post_norm", "mel_max_post_norm", "num_onsets", "num_beats", "bpm",
            ]},
            "wav": {k: wav_runs[0]["results"].get(k) for k in [
                "waveform_sr", "waveform_channels", "waveform_samples", "waveform_duration_s",
                "mel_shape", "mel_time_frames", "mel_pre_norm_mean", "mel_pre_norm_std",
                "mel_min_pre_norm", "mel_max_pre_norm", "mel_post_norm_mean", "mel_post_norm_std",
                "mel_min_post_norm", "mel_max_post_norm", "num_onsets", "num_beats", "bpm",
            ]},
            "spectrogram_similarity": spec_sim,
        },
        "raw_predictions": {
            "mp3": {k: mp3_runs[0]["results"].get(k) for k in [
                "pred_frames", "event_prob_min", "event_prob_max", "event_prob_mean", "event_prob_std",
                "frames_above_eval_threshold", "frames_above_pp_threshold",
                "lane_distribution_pre_pp", "num_candidates_pre_pp",
            ]},
            "wav": {k: wav_runs[0]["results"].get(k) for k in [
                "pred_frames", "event_prob_min", "event_prob_max", "event_prob_mean", "event_prob_std",
                "frames_above_eval_threshold", "frames_above_pp_threshold",
                "lane_distribution_pre_pp", "num_candidates_pre_pp",
            ]},
        },
        "postprocessing": {
            "mp3": {k: mp3_runs[0]["results"].get(k) for k in [
                "pp_event_threshold", "pp_min_gap_ms", "pp_same_lane_min_gap_ms",
                "pp_onset_snap_tolerance_ms", "pp_beat_snap_tolerance_ms", "pp_max_density_nps",
                "num_candidates_pre_pp", "num_notes_after_pp", "notes_removed", "notes_removed_pct",
            ]},
            "wav": {k: wav_runs[0]["results"].get(k) for k in [
                "pp_event_threshold", "pp_min_gap_ms", "pp_same_lane_min_gap_ms",
                "pp_onset_snap_tolerance_ms", "pp_beat_snap_tolerance_ms", "pp_max_density_nps",
                "num_candidates_pre_pp", "num_notes_after_pp", "notes_removed", "notes_removed_pct",
            ]},
        },
        "beatmap": {
            "mp3": {k: mp3_runs[0]["results"].get(k) for k in [
                "beatmap_duration_ms", "note_density_per_s", "lane_counts",
                "beatmap_json_size_bytes", "beatmap_success",
            ]},
            "wav": {k: wav_runs[0]["results"].get(k) for k in [
                "beatmap_duration_ms", "note_density_per_s", "lane_counts",
                "beatmap_json_size_bytes", "beatmap_success",
            ]},
        },
        "beatmap_similarity": matching_results,
        "determinism": determinism,
        "warnings": warnings_list,
    }

    with open(output_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # ── comparison_summary.csv ──
    log.info("Menyusun comparison_summary.csv...")
    csv_rows = []
    csv_rows.append(["file_size", file_mp3["size_bytes"], file_wav["size_bytes"],
                      file_mp3["size_bytes"] - file_wav["size_bytes"],
                      round((file_mp3["size_bytes"] - file_wav["size_bytes"]) / max(file_wav["size_bytes"], 1) * 100, 2),
                      "bytes", ""])
    csv_rows.append(["duration_seconds",
                      f'{dur_mp3:.4f}' if isinstance(dur_mp3, float) else dur_mp3,
                      f'{dur_wav:.4f}' if isinstance(dur_wav, float) else dur_wav,
                      f'{dur_mp3-dur_wav:.6f}' if isinstance(dur_mp3, float) and isinstance(dur_wav, float) else "N/A",
                      "", "seconds", ""])
    # Pipeline time
    mp3_avg = float(np.mean([t["pipeline_total_s"] for t in mp3_timings_list]))
    wav_avg = float(np.mean([t["pipeline_total_s"] for t in wav_timings_list]))
    csv_rows.append(["pipeline_total_avg", f"{mp3_avg:.6f}", f"{wav_avg:.6f}",
                      f"{mp3_avg-wav_avg:.6f}",
                      f"{(mp3_avg-wav_avg)/max(wav_avg,1e-9)*100:.2f}",
                      "seconds", "rata-rata"])
    # Notes
    n_mp3 = mp3_runs[0]["results"]["num_notes_after_pp"]
    n_wav = wav_runs[0]["results"]["num_notes_after_pp"]
    csv_rows.append(["num_notes", n_mp3, n_wav, n_mp3 - n_wav,
                      round((n_mp3 - n_wav) / max(n_wav, 1) * 100, 2), "count", ""])
    # Spectrogram similarity
    csv_rows.append(["spec_pearson", spec_sim.get("pearson_correlation", ""), "", "", "", "", ""])
    csv_rows.append(["spec_mae", spec_sim.get("mae", ""), "", "", "", "", ""])
    csv_rows.append(["spec_rmse", spec_sim.get("rmse", ""), "", "", "", "", ""])
    csv_rows.append(["spec_cosine_sim", spec_sim.get("cosine_similarity", ""), "", "", "", "", ""])
    # Matching 50ms
    mt50 = matching_results.get("time_only_50ms", {})
    csv_rows.append(["matched_notes_50ms_time", mt50.get("num_matched", ""), "", "", "", "count", ""])
    ml50 = matching_results.get("time_and_lane_50ms", {})
    csv_rows.append(["matched_notes_50ms_time_lane", ml50.get("num_matched", ""), "", "", "", "count", ""])
    # BPM
    csv_rows.append(["bpm", mp3_runs[0]["results"]["bpm"], wav_runs[0]["results"]["bpm"],
                      mp3_runs[0]["results"]["bpm"] - wav_runs[0]["results"]["bpm"], "", "bpm", ""])

    with open(output_dir / "comparison_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mp3", "wav", "difference", "difference_percentage", "unit", "notes"])
        writer.writerows(csv_rows)

    # ── Generate report ──
    log.info("Menyusun comparison_report.md...")
    report = generate_report(
        env, file_mp3, file_wav,
        {**backend_mp3_validation, **backend_mp3_save},
        {**backend_wav_validation, **backend_wav_save},
        mp3_runs, wav_runs,
        spec_sim, matching_results,
        determinism, warnings_list,
        output_dir, num_runs,
    )
    with open(output_dir / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # ── Cleanup temp dir ──
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    # ── Terminal summary ──
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL PERBANDINGAN MP3 vs WAV")
    print("=" * 70)
    print(f"  Ukuran MP3      : {file_mp3['size_mb']} MB")
    print(f"  Ukuran WAV      : {file_wav['size_mb']} MB")
    print(f"  Pipeline avg MP3: {mp3_avg:.4f}s")
    print(f"  Pipeline avg WAV: {wav_avg:.4f}s")
    print(f"  Jumlah note MP3 : {n_mp3}")
    print(f"  Jumlah note WAV : {n_wav}")
    mt50_r = matching_results.get("time_only_50ms", {})
    ml50_r = matching_results.get("time_and_lane_50ms", {})
    print(f"  Note cocok 50ms (waktu)     : {mt50_r.get('num_matched', 'N/A')}")
    print(f"  Note cocok 50ms (waktu+lane): {ml50_r.get('num_matched', 'N/A')}")
    if mt50_r.get("num_matched", 0) > 0:
        lane_agr = ml50_r.get("num_matched", 0) / mt50_r["num_matched"] * 100
        print(f"  Kesesuaian lane (50ms)      : {lane_agr:.1f}%")
    if mt50_r.get("mean_dt_ms") is not None:
        print(f"  Rata-rata selisih timestamp : {mt50_r['mean_dt_ms']:.2f} ms")
    print(f"  Lokasi laporan  : {output_dir}")
    if warnings_list:
        print(f"\n  PERINGATAN:")
        for w in warnings_list:
            print(f"    [!] {w}")
    print("=" * 70)

    # List created files
    print("\nFile yang dibuat:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

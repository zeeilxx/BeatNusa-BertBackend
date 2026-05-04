from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np


@dataclass
class AudioFeatures:
    waveform: np.ndarray
    sr: int
    duration_ms: float
    mel: np.ndarray
    onset_times_ms: np.ndarray
    beat_times_ms: np.ndarray
    bpm: float


def load_audio(audio_path: str | Path, cfg: Dict) -> Tuple[np.ndarray, int]:
    audio_cfg = cfg['audio']
    y, sr = librosa.load(
        str(audio_path),
        sr=audio_cfg['sample_rate'],
        mono=audio_cfg.get('mono', True),
    )
    return y.astype(np.float32), sr


def compute_log_mel(y: np.ndarray, sr: int, cfg: Dict) -> np.ndarray:
    audio_cfg = cfg['audio']
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=audio_cfg['n_fft'],
        hop_length=audio_cfg['hop_length'],
        win_length=audio_cfg['win_length'],
        n_mels=audio_cfg['n_mels'],
        fmin=audio_cfg['fmin'],
        fmax=audio_cfg['fmax'],
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=audio_cfg.get('top_db', 80))
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


def compute_rhythm_guides(y: np.ndarray, sr: int, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    hop = cfg['audio']['hop_length']
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop) * 1000.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop) * 1000.0
    return onset_times.astype(np.float32), beat_times.astype(np.float32), float(np.asarray(tempo).reshape(-1)[0])


def augment_waveform(y: np.ndarray, sr: int, *, pitch_shift_steps: float = 0.0, time_stretch_rate: float = 1.0) -> np.ndarray:
    augmented = np.asarray(y, dtype=np.float32)
    if abs(float(pitch_shift_steps)) > 1e-8:
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=float(pitch_shift_steps))
    if abs(float(time_stretch_rate) - 1.0) > 1e-8:
        augmented = librosa.effects.time_stretch(augmented, rate=float(time_stretch_rate))
    return np.asarray(augmented, dtype=np.float32)


def extract_audio_features(audio_path: str | Path, cfg: Dict) -> AudioFeatures:
    y, sr = load_audio(audio_path, cfg)
    return extract_audio_features_from_waveform(y, sr, cfg)


def extract_audio_features_from_waveform(y: np.ndarray, sr: int, cfg: Dict) -> AudioFeatures:
    mel = compute_log_mel(y, sr, cfg)
    onset_ms, beat_ms, bpm = compute_rhythm_guides(y, sr, cfg)
    duration_ms = len(y) / sr * 1000.0
    return AudioFeatures(
        waveform=np.asarray(y, dtype=np.float32),
        sr=sr,
        duration_ms=duration_ms,
        mel=mel,
        onset_times_ms=onset_ms,
        beat_times_ms=beat_ms,
        bpm=bpm,
    )


def frame_times_ms(num_frames: int, cfg: Dict) -> np.ndarray:
    hop = cfg['audio']['hop_length']
    sr = cfg['audio']['sample_rate']
    return (np.arange(num_frames, dtype=np.float32) * hop / sr) * 1000.0

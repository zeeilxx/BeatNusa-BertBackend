"""
AI model loading and inference service.
Wraps the existing BeatmapBERT predictor module.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Ensure the 'src' folder is in the Python path so beatbert imports work
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beatbert.configs import load_config
from beatbert.models.beatmap_model import BeatmapModel
from beatbert.inference.predictor import predict_song
from beatbert.utils.audio import extract_audio_features, frame_times_ms
from beatbert.inference.predictor import _run_chunked_inference
from beatbert.inference.postprocess import postprocess_events


class AIService:
    """
    Singleton-style AI service that loads the model once at startup
    and provides inference methods.
    """

    def __init__(self):
        self.model: Optional[BeatmapModel] = None
        self.cfg: Optional[Dict] = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._loaded = False

    def load_model(self, config_path: str, checkpoint_path: str) -> None:
        """
        Load the BeatmapBERT model and config.
        Called once during application startup.
        """
        print(f"[AI Service] Memuat konfigurasi dari {config_path}...")
        self.cfg = load_config(config_path)

        print(f"[AI Service] Menginisialisasi model BeatmapBERT...")
        self.model = BeatmapModel(self.cfg).to(self.device)

        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"[AI Service] Model berhasil dimuat dari {checkpoint_path}")
        else:
            print(f"[AI Service] WARNING: Checkpoint tidak ditemukan di {checkpoint_path}")

        self.model.eval()
        self._loaded = True
        print(f"[AI Service] Model siap di device: {self.device}")

    def is_loaded(self) -> bool:
        return self._loaded

    def generate_beatmap(self, audio_path: str) -> Dict:
        """
        Generate a beatmap from an audio file using the loaded model.

        Returns a dictionary with the complete beatmap structure:
        {
            "bpm": float,
            "duration_ms": float,
            "offset_ms": 0,
            "lane_count": 4,
            "notes": [{"time_ms": int, "lane": int, "type": "tap", "length_ms": 0}, ...],
            "note_count": int
        }
        """
        if not self._loaded:
            raise RuntimeError("AI model belum dimuat. Panggil load_model() terlebih dahulu.")

        # Run inference directly using the already-loaded model
        # (avoid predict_song which reloads the checkpoint each time)
        return self._direct_inference(audio_path)

    def _direct_inference(self, audio_path: str) -> Dict:
        """
        Run inference directly using the already-loaded model
        (avoids reloading weights each time).
        """
        import numpy as np

        # Extract audio features
        feats = extract_audio_features(audio_path, self.cfg)
        mel = feats.mel
        f_ms = frame_times_ms(mel.shape[1], self.cfg)

        # Run chunked inference
        event_prob, lane_prob = _run_chunked_inference(
            self.model, mel, self.cfg, self.device
        )
        lane_pred = lane_prob.argmax(axis=-1)

        # Threshold filtering
        threshold = self.cfg["postprocess"]["event_threshold"]
        raw_events: List[Dict] = []
        idxs = np.where(event_prob >= threshold)[0]
        for idx in idxs:
            raw_events.append({
                "time_ms": float(f_ms[idx]),
                "lane": int(lane_pred[idx]),
                "confidence": float(event_prob[idx]),
            })

        # Post-processing (snap to beats/onsets, density limiting)
        final_events = postprocess_events(
            raw_events, feats.onset_times_ms, feats.beat_times_ms, self.cfg
        )

        # Build Unity-compatible notes
        notes = [
            {
                "time_ms": int(round(e["time_ms"])),
                "lane": int(e["lane"]),
                "type": "tap",
                "length_ms": 0,
            }
            for e in final_events
        ]

        lane_count = int(self.cfg["model"]["num_lanes"])

        return {
            "bpm": feats.bpm,
            "duration_ms": feats.duration_ms,
            "offset_ms": 0,
            "lane_count": lane_count,
            "notes": notes,
            "note_count": len(notes),
        }


# ── Module-level singleton ────────────────────────────────────
ai_service = AIService()

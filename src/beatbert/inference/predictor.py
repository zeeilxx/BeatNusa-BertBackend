from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from beatbert.models.beatmap_model import BeatmapModel
from beatbert.inference.postprocess import postprocess_events
from beatbert.utils.audio import extract_audio_features, frame_times_ms
from beatbert.utils.io import write_json


def load_checkpoint_model(checkpoint_path: str | Path, cfg: Dict, device: torch.device) -> BeatmapModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = BeatmapModel(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _run_chunked_inference(model: BeatmapModel, mel: np.ndarray, cfg: Dict, device: torch.device):
    total_frames = mel.shape[1]
    chunk = int(cfg['inference']['chunk_frames'])
    overlap = int(cfg['inference']['chunk_overlap'])
    step = max(1, chunk - overlap)

    event_accum = np.zeros(total_frames, dtype=np.float32)
    count_accum = np.zeros(total_frames, dtype=np.float32)
    lane_accum = np.zeros((total_frames, cfg['model']['num_lanes']), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, total_frames, step):
            end = min(total_frames, start + chunk)
            piece = mel[:, start:end]
            valid_len = piece.shape[1]
            if valid_len < chunk:
                piece = np.pad(piece, ((0, 0), (0, chunk - valid_len)))
                mask_np = np.concatenate([np.ones(valid_len, dtype=np.float32), np.zeros(chunk - valid_len, dtype=np.float32)])
            else:
                mask_np = np.ones(chunk, dtype=np.float32)

            x = torch.from_numpy(piece).unsqueeze(0).to(device)
            mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
            outputs = model(x, mask)
            event_prob = torch.sigmoid(outputs['event_logits']).squeeze(0).cpu().numpy()[:valid_len]
            lane_prob = torch.softmax(outputs['lane_logits'], dim=-1).squeeze(0).cpu().numpy()[:valid_len]

            event_accum[start:end] += event_prob
            count_accum[start:end] += 1
            lane_accum[start:end] += lane_prob
            if end == total_frames:
                break

    count_accum = np.clip(count_accum, 1e-6, None)
    event_prob = event_accum / count_accum
    lane_prob = lane_accum / count_accum[:, None]
    return event_prob, lane_prob


def predict_song(audio_path: str | Path, checkpoint_path: str | Path, cfg: Dict, output_json_path: str | Path | None = None) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['general']['device'] != 'cpu' else 'cpu')
    model = load_checkpoint_model(checkpoint_path, cfg, device)
    feats = extract_audio_features(audio_path, cfg)
    mel = feats.mel
    frame_ms = frame_times_ms(mel.shape[1], cfg)

    event_prob, lane_prob = _run_chunked_inference(model, mel, cfg, device)
    lane_pred = lane_prob.argmax(axis=-1)

    threshold = cfg['postprocess']['event_threshold']
    raw_events: List[Dict] = []
    idxs = np.where(event_prob >= threshold)[0]
    for idx in idxs:
        raw_events.append({
            'time_ms': float(frame_ms[idx]),
            'lane': int(lane_pred[idx]),
            'confidence': float(event_prob[idx]),
        })

    final_events = postprocess_events(raw_events, feats.onset_times_ms, feats.beat_times_ms, cfg)
    result = {
        'song': Path(audio_path).name,
        'bpm': feats.bpm,
        'offset_ms': 0,
        'num_events': len(final_events),
        'events': final_events,
        'notes': [{'time_ms': int(round(e['time_ms'])), 'lane': int(e['lane'])} for e in final_events],
    }
    if output_json_path is not None:
        write_json(output_json_path, result)
    return result

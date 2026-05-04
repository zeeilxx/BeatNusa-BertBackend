from __future__ import annotations

from typing import Dict, List

import numpy as np


def _snap(value: float, candidates: np.ndarray, tolerance_ms: float) -> float:
    if candidates.size == 0:
        return value
    idx = int(np.argmin(np.abs(candidates - value)))
    best = float(candidates[idx])
    return best if abs(best - value) <= tolerance_ms else value


def postprocess_events(events: List[Dict], onset_times_ms: np.ndarray, beat_times_ms: np.ndarray, cfg: Dict) -> List[Dict]:
    pcfg = cfg['postprocess']
    out = []
    last_global = -1e9
    last_by_lane = {lane: -1e9 for lane in range(cfg['model']['num_lanes'])}
    for e in sorted(events, key=lambda x: x['time_ms']):
        time_ms = float(e['time_ms'])
        lane = int(e['lane'])
        if pcfg.get('snap_to_onsets', True):
            time_ms = _snap(time_ms, onset_times_ms, pcfg['onset_snap_tolerance_ms'])
        if pcfg.get('snap_to_beats', True):
            time_ms = _snap(time_ms, beat_times_ms, pcfg['beat_snap_tolerance_ms'])
        if time_ms - last_global < pcfg['min_gap_ms']:
            continue
        if time_ms - last_by_lane[lane] < pcfg['same_lane_min_gap_ms']:
            continue
        e['time_ms'] = int(round(time_ms))
        out.append(e)
        last_global = time_ms
        last_by_lane[lane] = time_ms

    density_limit = float(pcfg.get('max_density_notes_per_second', 8))
    if density_limit <= 0:
        return out
    filtered = []
    window = []
    for e in out:
        t = e['time_ms']
        window = [x for x in window if t - x <= 1000]
        if len(window) < density_limit:
            filtered.append(e)
            window.append(t)
    return filtered

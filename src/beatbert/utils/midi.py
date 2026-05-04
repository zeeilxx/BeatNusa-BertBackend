from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import mido
import numpy as np


@dataclass
class MidiNote:
    pitch: int
    velocity: int
    start_ms: float
    end_ms: float
    lane: int


def _lane_from_pitch(pitch: int, cfg: Dict) -> int:
    midi_cfg = cfg['midi']
    strategy = midi_cfg['lane_strategy']
    num_lanes = int(midi_cfg['num_lanes'])
    if strategy == 'explicit':
        mapping = {int(k): int(v) for k, v in midi_cfg.get('explicit_pitch_map', {}).items()}
        if pitch not in mapping:
            raise KeyError(f'Pitch {pitch} not found in explicit_pitch_map')
        return mapping[pitch]
    if strategy == 'modulo':
        return int(pitch) % num_lanes
    if strategy == 'range':
        min_pitch = int(midi_cfg.get('min_pitch', 21))
        max_pitch = int(midi_cfg.get('max_pitch', 108))
        clipped = min(max(pitch, min_pitch), max_pitch)
        normalized = (clipped - min_pitch) / max(1, max_pitch - min_pitch + 1)
        return min(int(normalized * num_lanes), num_lanes - 1)
    raise ValueError(f'Unsupported lane strategy: {strategy}')


def parse_midi_notes(midi_path: str | Path, cfg: Dict) -> List[MidiNote]:
    midi = mido.MidiFile(str(midi_path))
    tempo = 500000
    current_ms = 0.0
    ticks_per_beat = midi.ticks_per_beat or cfg['midi'].get('ticks_per_beat_default', 480)
    active: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    notes: List[MidiNote] = []
    velocity_threshold = int(cfg['midi'].get('velocity_threshold', 1))

    merged = mido.merge_tracks(midi.tracks)
    for msg in merged:
        delta_ms = mido.tick2second(msg.time, ticks_per_beat, tempo) * 1000.0
        current_ms += delta_ms
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            continue
        if msg.type == 'note_on' and msg.velocity >= velocity_threshold:
            key = (msg.channel if hasattr(msg, 'channel') else 0, msg.note)
            active.setdefault(key, []).append((current_ms, msg.velocity))
            continue
        if msg.type in {'note_off', 'note_on'}:
            is_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
            if not is_off:
                continue
            key = (msg.channel if hasattr(msg, 'channel') else 0, msg.note)
            if key not in active or len(active[key]) == 0:
                continue
            start_ms, velocity = active[key].pop(0)
            end_ms = max(current_ms, start_ms)
            lane = _lane_from_pitch(msg.note, cfg)
            notes.append(MidiNote(
                pitch=msg.note,
                velocity=velocity,
                start_ms=start_ms,
                end_ms=end_ms,
                lane=lane,
            ))

    notes.sort(key=lambda n: n.start_ms)
    return notes


def scale_note_timestamps(notes: List[MidiNote], factor: float) -> List[MidiNote]:
    if factor <= 0:
        raise ValueError('scale factor must be positive')
    if abs(factor - 1.0) < 1e-8:
        return list(notes)
    scaled: List[MidiNote] = []
    for note in notes:
        scaled.append(replace(
            note,
            start_ms=float(note.start_ms) * factor,
            end_ms=float(note.end_ms) * factor,
        ))
    return scaled


def notes_to_frame_labels(notes: List[MidiNote], frame_times_ms: np.ndarray, cfg: Dict) -> Dict[str, np.ndarray]:
    num_frames = len(frame_times_ms)
    event = np.zeros(num_frames, dtype=np.float32)
    lane = np.full(num_frames, fill_value=-100, dtype=np.int64)

    for note in notes:
        idx = int(np.argmin(np.abs(frame_times_ms - note.start_ms)))
        event[idx] = 1.0
        lane[idx] = int(note.lane)

    return {
        'event': event,
        'lane': lane,
    }

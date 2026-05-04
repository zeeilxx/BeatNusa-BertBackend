"""
Pydantic schemas for Beatmap-related requests and responses.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class BeatmapNote(BaseModel):
    """Single note in a beatmap."""
    time_ms: int
    lane: int
    type: str = "tap"
    length_ms: int = 0


class BeatmapJSON(BaseModel):
    """
    The beatmap payload returned to Unity.
    This is the exact structure Unity expects.
    """
    lane_count: int = 4
    offset_ms: int = 0
    notes: List[BeatmapNote]


class BeatmapResponse(BaseModel):
    """Full beatmap response with metadata."""
    id: int
    song_id: int
    song_code: str
    model_name: str
    model_version: str
    difficulty_name: str
    lane_count: int
    offset_ms: int
    note_count: int
    generation_status: str
    generated_at: datetime
    beatmap: BeatmapJSON

    model_config = {"from_attributes": True}


class BeatmapRegenerateResponse(BaseModel):
    """Response after regenerating a beatmap."""
    status: str
    song_code: str
    note_count: int
    message: str

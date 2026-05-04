"""
Pydantic schemas for GameResult-related requests and responses.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class GameResultCreate(BaseModel):
    """Submitted by Unity after gameplay."""
    song_code: str
    beatmap_id: int
    player_name: str
    score: int
    accuracy: float
    max_combo: int
    hit_count: int = 0
    miss_count: int = 0
    good_count: int = 0
    perfect_count: int = 0
    bad_count: int = 0
    mean_offset_ms: Optional[float] = None


class GameResultResponse(BaseModel):
    """Returned after storing a game result."""
    id: int
    song_code: str
    beatmap_id: int
    player_name: str
    score: int
    accuracy: float
    max_combo: int
    hit_count: int
    miss_count: int
    good_count: int
    perfect_count: int
    bad_count: int
    mean_offset_ms: Optional[float] = None
    played_at: datetime

    model_config = {"from_attributes": True}


class LeaderboardEntry(BaseModel):
    """Single row in a leaderboard."""
    rank: int
    player_name: str
    score: int
    accuracy: float
    max_combo: int
    played_at: datetime

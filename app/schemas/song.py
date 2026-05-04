"""
Pydantic schemas for Song-related requests and responses.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SongUploadResponse(BaseModel):
    """Returned immediately after a successful upload + generation."""
    status: str
    song_code: str
    title: str
    process_status: str
    message: str

    model_config = {"from_attributes": True}


class SongListItem(BaseModel):
    """Compact song item for listing playable songs."""
    id: int
    song_code: str
    title: str
    artist: Optional[str] = None
    genre: Optional[str] = None
    file_format: str
    duration_seconds: Optional[float] = None
    bpm: Optional[float] = None
    process_status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SongDetail(BaseModel):
    """Full song metadata."""
    id: int
    song_code: str
    title: str
    artist: Optional[str] = None
    genre: Optional[str] = None
    original_filename: str
    stored_filename: str
    file_path: str
    file_format: str
    duration_seconds: Optional[float] = None
    bpm: Optional[float] = None
    upload_source: str
    process_status: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SongStatusResponse(BaseModel):
    """Quick status check response."""
    song_code: str
    process_status: str
    title: str

    model_config = {"from_attributes": True}

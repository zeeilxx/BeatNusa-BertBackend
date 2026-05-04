"""
Beatmap-related API endpoints.
Handles retrieval and regeneration of beatmaps.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.beatmap import Beatmap
from app.models.song import Song
from app.schemas.beatmap import (
    BeatmapJSON,
    BeatmapNote,
    BeatmapRegenerateResponse,
    BeatmapResponse,
)
from app.services.beatmap_service import get_beatmap_for_song, regenerate_beatmap

router = APIRouter(prefix="/api/beatmaps", tags=["Beatmaps"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /api/beatmaps/{song_code}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.get("/{song_code}", response_model=BeatmapResponse)
async def get_beatmap(
    song_code: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the beatmap JSON for a song.
    Only returns beatmaps for songs with process_status = 'done'.
    This is the primary endpoint Unity calls to load gameplay data.
    """
    beatmap = await get_beatmap_for_song(song_code, db)

    if not beatmap:
        raise HTTPException(
            status_code=404,
            detail=f"Beatmap untuk song {song_code} tidak ditemukan "
            f"atau song belum selesai diproses.",
        )

    # Parse the stored JSON text back into structured data
    try:
        raw_json = json.loads(beatmap.beatmap_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Beatmap JSON rusak di database.",
        )

    # Build structured response
    notes = [
        BeatmapNote(
            time_ms=n["time_ms"],
            lane=n["lane"],
            type=n.get("type", "tap"),
            length_ms=n.get("length_ms", 0),
        )
        for n in raw_json.get("notes", [])
    ]

    beatmap_payload = BeatmapJSON(
        lane_count=raw_json.get("lane_count", beatmap.lane_count),
        offset_ms=raw_json.get("offset_ms", beatmap.offset_ms),
        notes=notes,
    )

    return BeatmapResponse(
        id=beatmap.id,
        song_id=beatmap.song_id,
        song_code=song_code,
        model_name=beatmap.model_name,
        model_version=beatmap.model_version,
        difficulty_name=beatmap.difficulty_name,
        lane_count=beatmap.lane_count,
        offset_ms=beatmap.offset_ms,
        note_count=beatmap.note_count,
        generation_status=beatmap.generation_status,
        generated_at=beatmap.generated_at,
        beatmap=beatmap_payload,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /api/beatmaps/{song_code}/regenerate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.post("/{song_code}/regenerate", response_model=BeatmapRegenerateResponse)
async def regenerate_beatmap_endpoint(
    song_code: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Force-regenerate beatmap for a song.
    Creates a new beatmap record (old one is kept for history).
    """
    try:
        beatmap = await regenerate_beatmap(song_code, db)
        return BeatmapRegenerateResponse(
            status="success",
            song_code=song_code,
            note_count=beatmap.note_count,
            message="Beatmap berhasil diregenerasi.",
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

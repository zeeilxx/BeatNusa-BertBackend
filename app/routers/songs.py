"""
Song-related API endpoints.
Handles upload, listing, and status checking.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.song import Song
from app.schemas.song import (
    SongDetail,
    SongListItem,
    SongStatusResponse,
    SongUploadResponse,
)
from app.services.audio_service import AudioValidationError
from app.services.beatmap_service import upload_and_process

router = APIRouter(prefix="/api/songs", tags=["Songs"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /api/songs/upload
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.post("/upload", response_model=SongUploadResponse, status_code=201)
async def upload_song(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    artist: Optional[str] = Form(default=None),
    genre: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload an audio file (MP3/WAV/OGG) and trigger AI beatmap generation.

    The full pipeline runs synchronously:
    1. Validate file
    2. Save to storage
    3. Extract metadata
    4. AI generates beatmap
    5. Store everything in database

    Returns the song_code for future reference.
    """
    try:
        song = await upload_and_process(
            file=file,
            db=db,
            title=title,
            artist=artist,
            genre=genre,
        )
        return SongUploadResponse(
            status="success",
            song_code=song.song_code,
            title=song.title,
            process_status=song.process_status,
            message="Audio berhasil diupload dan beatmap telah digenerate.",
        )

    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal: {str(e)}",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /api/songs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.get("", response_model=List[SongListItem])
async def list_songs(
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List all available songs.
    By default, only returns songs where process_status = 'done'.
    Pass `?status=all` to see all songs.
    """
    stmt = select(Song).where(Song.is_active == True)

    if status != "all":
        stmt = stmt.where(Song.process_status == "done")

    stmt = stmt.order_by(Song.created_at.desc())
    result = await db.execute(stmt)
    songs = result.scalars().all()

    return [SongListItem.model_validate(s) for s in songs]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /api/songs/{song_code}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.get("/{song_code}", response_model=SongDetail)
async def get_song(
    song_code: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full song details by song_code."""
    stmt = select(Song).where(Song.song_code == song_code)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise HTTPException(status_code=404, detail=f"Song {song_code} tidak ditemukan.")

    return SongDetail.model_validate(song)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /api/songs/{song_code}/status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.get("/{song_code}/status", response_model=SongStatusResponse)
async def get_song_status(
    song_code: str,
    db: AsyncSession = Depends(get_db),
):
    """Quick check on the processing status of a song."""
    stmt = select(Song).where(Song.song_code == song_code)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise HTTPException(status_code=404, detail=f"Song {song_code} tidak ditemukan.")

    return SongStatusResponse(
        song_code=song.song_code,
        process_status=song.process_status,
        title=song.title,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /api/songs/{song_code}/audio
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from fastapi.responses import FileResponse
import os

@router.get("/{song_code}/audio{ext}")
async def get_song_audio(
    song_code: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the actual audio file for the song."""
    stmt = select(Song).where(Song.song_code == song_code)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise HTTPException(status_code=404, detail=f"Song {song_code} tidak ditemukan.")

    if not song.file_path or not os.path.exists(song.file_path):
        raise HTTPException(status_code=404, detail="File audio tidak ditemukan di server.")

    # Deteksi mime type berdasarkan ekstensi
    extension = os.path.splitext(song.file_path)[1].lower()
    if extension == ".mp3":
        media_type = "audio/mpeg"
    elif extension == ".wav":
        media_type = "audio/wav"
    elif extension == ".ogg":
        media_type = "audio/ogg"
    else:
        media_type = "application/octet-stream"

    return FileResponse(path=song.file_path, media_type=media_type, filename=song.original_filename)

"""
Beatmap orchestration service.
Handles the end-to-end flow: upload → validate → store → AI → DB.
Supports background processing in separate threads to avoid freezing the server.
"""

import json
import uuid
import asyncio
from typing import Optional

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.song import Song
from app.models.beatmap import Beatmap
from app.services.audio_service import (
    AudioValidationError,
    validate_upload,
    save_audio_file,
    extract_audio_metadata,
    validate_audio_duration,
    delete_audio_file,
)
from app.services.ai_service import ai_service
from app.database import AsyncSessionLocal


async def upload_initial(
    file: UploadFile,
    db: AsyncSession,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    genre: Optional[str] = None,
) -> Song:
    """
    Step 1 of the pipeline (Synchronous/Fast):
    1. Validate file format & size
    2. Save to storage
    3. Extract metadata (duration, BPM)
    4. Validate duration
    5. Insert song record (status = 'uploaded')
    """
    ext = await validate_upload(file)
    stored_filename, file_path, file_format = await save_audio_file(file, ext)

    try:
        duration_seconds, bpm = extract_audio_metadata(file_path)
    except Exception as e:
        delete_audio_file(file_path)
        raise AudioValidationError(f"Gagal membaca metadata audio: {str(e)}")

    try:
        validate_audio_duration(duration_seconds)
    except AudioValidationError:
        delete_audio_file(file_path)
        raise

    song_code = f"SONG-{uuid.uuid4().hex[:8].upper()}"
    song_title = title or (file.filename.rsplit(".", 1)[0] if file.filename else "Untitled")

    song = Song(
        song_code=song_code,
        title=song_title,
        artist=artist,
        genre=genre,
        original_filename=file.filename or "unknown",
        stored_filename=stored_filename,
        file_path=file_path,
        file_format=file_format,
        duration_seconds=duration_seconds,
        bpm=bpm,
        upload_source="user_upload",
        process_status="uploaded",
        is_active=True,
    )
    db.add(song)
    await db.commit()
    await db.refresh(song)
    
    return song


async def process_ai_background(song_id: int):
    """
    Step 2 of the pipeline (Background Task):
    Runs in a separate thread to prevent blocking the event loop.
    """
    async with AsyncSessionLocal() as db:
        stmt = select(Song).where(Song.id == song_id)
        result = await db.execute(stmt)
        song = result.scalar_one_or_none()
        
        if not song:
            return

        song.process_status = "processing"
        await db.commit()

        try:
            # IMPORTANT: Run the CPU-heavy AI task in a thread pool!
            # This prevents the '502 Bad Gateway' on polling requests.
            loop = asyncio.get_event_loop()
            ai_result = await loop.run_in_executor(
                None, 
                ai_service.generate_beatmap, 
                song.file_path
            )
            
            beatmap_payload = _build_beatmap_json(ai_result)
            beatmap = Beatmap(
                song_id=song.id,
                model_name="BeatmapBERT",
                model_version="1.0",
                difficulty_name="normal",
                lane_count=ai_result["lane_count"],
                offset_ms=ai_result["offset_ms"],
                beatmap_json=json.dumps(beatmap_payload),
                note_count=ai_result["note_count"],
                generation_status="generated",
            )
            db.add(beatmap)
            song.process_status = "done"
            await db.commit()
            print(f"[Background] Sukses: {song.song_code}")

        except Exception as e:
            print(f"[Background] ERROR: {str(e)}")
            song.process_status = "failed"
            await db.commit()


async def get_beatmap_for_song(song_code: str, db: AsyncSession) -> Optional[Beatmap]:
    stmt = (
        select(Beatmap)
        .join(Song, Beatmap.song_id == Song.id)
        .where(Song.song_code == song_code)
        .where(Song.process_status == "done")
        .where(Beatmap.generation_status == "generated")
        .order_by(Beatmap.generated_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def regenerate_beatmap(song_code: str, db: AsyncSession) -> Beatmap:
    stmt = select(Song).where(Song.song_code == song_code).where(Song.is_active == True)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise ValueError(f"Song {song_code} not found.")

    song.process_status = "processing"
    await db.commit()

    try:
        loop = asyncio.get_event_loop()
        ai_result = await loop.run_in_executor(None, ai_service.generate_beatmap, song.file_path)
        
        beatmap_payload = _build_beatmap_json(ai_result)
        beatmap = Beatmap(
            song_id=song.id,
            model_name="BeatmapBERT",
            model_version="1.0",
            difficulty_name="normal",
            lane_count=ai_result["lane_count"],
            offset_ms=ai_result["offset_ms"],
            beatmap_json=json.dumps(beatmap_payload),
            note_count=ai_result["note_count"],
            generation_status="generated",
        )
        db.add(beatmap)
        song.process_status = "done"
        await db.commit()
        await db.refresh(beatmap)
        return beatmap
    except Exception as e:
        song.process_status = "failed"
        await db.commit()
        raise RuntimeError(str(e))


def _build_beatmap_json(ai_result: dict) -> dict:
    return {
        "lane_count": ai_result["lane_count"],
        "offset_ms": ai_result["offset_ms"],
        "notes": ai_result["notes"],
    }

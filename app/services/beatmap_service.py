"""
Beatmap orchestration service.
Handles the end-to-end flow: upload → validate → store → AI → DB.
"""

import json
import uuid
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


async def upload_and_process(
    file: UploadFile,
    db: AsyncSession,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    genre: Optional[str] = None,
) -> Song:
    """
    Full upload + AI generation pipeline:
    1. Validate file format & size
    2. Save to storage
    3. Extract metadata (duration, BPM)
    4. Validate duration
    5. Insert song record (status = 'uploaded')
    6. Update status → 'processing'
    7. Run AI inference
    8. Insert beatmap record
    9. Update status → 'done'

    On any error: set status → 'failed', cleanup file.

    Returns the Song ORM object.
    """
    # ── Step 1: Validate ──────────────────────────────────────
    ext = await validate_upload(file)

    # ── Step 2: Save to disk ──────────────────────────────────
    stored_filename, file_path, file_format = await save_audio_file(file, ext)

    # ── Step 3: Extract metadata ──────────────────────────────
    try:
        duration_seconds, bpm = extract_audio_metadata(file_path)
    except Exception as e:
        delete_audio_file(file_path)
        raise AudioValidationError(f"Gagal membaca metadata audio: {str(e)}")

    # ── Step 4: Validate duration ─────────────────────────────
    try:
        validate_audio_duration(duration_seconds)
    except AudioValidationError:
        delete_audio_file(file_path)
        raise

    # ── Step 5: Insert song record ────────────────────────────
    song_code = f"SONG-{uuid.uuid4().hex[:8].upper()}"
    song_title = title or file.filename.rsplit(".", 1)[0] if file.filename else "Untitled"

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

    # ── Step 6: Update status → processing ────────────────────
    song.process_status = "processing"
    await db.commit()

    # ── Step 7–9: AI generation ───────────────────────────────
    try:
        ai_result = ai_service.generate_beatmap(file_path)
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
        await db.refresh(song)

    except Exception as e:
        song.process_status = "failed"
        await db.commit()
        raise RuntimeError(f"AI generation gagal: {str(e)}")

    return song


async def get_beatmap_for_song(song_code: str, db: AsyncSession) -> Optional[Beatmap]:
    """
    Get the beatmap for a song by song_code.
    Only returns beatmaps for songs with process_status = 'done'.
    """
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
    """
    Force-regenerate beatmap for an existing song.
    """
    # Find the song
    stmt = select(Song).where(Song.song_code == song_code).where(Song.is_active == True)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise ValueError(f"Song dengan kode {song_code} tidak ditemukan.")

    # Update status → processing
    song.process_status = "processing"
    await db.commit()

    try:
        ai_result = ai_service.generate_beatmap(song.file_path)
        beatmap_payload = _build_beatmap_json(ai_result)

        # Create new beatmap record (keep old for history)
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
        raise RuntimeError(f"Regenerasi beatmap gagal: {str(e)}")


def _build_beatmap_json(ai_result: dict) -> dict:
    """
    Build the beatmap JSON payload in the format expected by Unity.
    """
    return {
        "lane_count": ai_result["lane_count"],
        "offset_ms": ai_result["offset_ms"],
        "notes": ai_result["notes"],
    }

"""
Audio file validation, storage, and metadata extraction.
"""

import os
import uuid
from pathlib import Path
from typing import Tuple

import aiofiles
from fastapi import UploadFile

from app.config import settings


# Allowed audio formats
ALLOWED_FORMATS = {".mp3", ".wav", ".ogg"}


class AudioValidationError(Exception):
    """Raised when audio validation fails."""
    pass


async def validate_upload(file: UploadFile) -> str:
    """
    Validate the uploaded audio file.
    Returns the file extension (e.g. '.mp3').
    Raises AudioValidationError on failure.
    """
    if not file.filename:
        raise AudioValidationError("Nama file tidak boleh kosong.")

    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_FORMATS:
        raise AudioValidationError(
            f"Format file tidak didukung: {ext}. "
            f"Format yang diperbolehkan: {', '.join(ALLOWED_FORMATS)}"
        )

    # Check file size by reading content length hint
    # We'll do definitive size check after saving
    if file.size and file.size > settings.MAX_UPLOAD_SIZE_BYTES:
        raise AudioValidationError(
            f"Ukuran file melebihi batas maksimum {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    return ext


async def save_audio_file(file: UploadFile, ext: str) -> Tuple[str, str, str]:
    """
    Save the uploaded audio to storage directory.

    Returns:
        (stored_filename, file_path, file_format)
        - stored_filename: UUID-based filename (e.g. 'abc123.mp3')
        - file_path: relative path from project root (e.g. 'storage/audio/abc123.mp3')
        - file_format: extension without dot (e.g. 'mp3')
    """
    # Ensure storage directory exists
    storage_path = settings.STORAGE_PATH
    storage_path.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = storage_path / stored_filename

    # Save file to disk
    content = await file.read()

    # Definitive size check after reading
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise AudioValidationError(
            f"Ukuran file melebihi batas maksimum {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    file_format = ext.lstrip(".")
    return stored_filename, str(file_path), file_format


def extract_audio_metadata(file_path: str) -> Tuple[float, float]:
    """
    Extract duration and BPM from an audio file using librosa.

    Returns:
        (duration_seconds, bpm)
    """
    import librosa

    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration_seconds = float(len(y) / sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)

    return duration_seconds, bpm


def validate_audio_duration(duration_seconds: float) -> None:
    """
    Validate that audio duration is within allowed limits.
    Raises AudioValidationError if too long.
    """
    if duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
        max_minutes = settings.MAX_AUDIO_DURATION_SECONDS / 60
        raise AudioValidationError(
            f"Durasi audio ({duration_seconds:.0f}s) melebihi batas "
            f"maksimum {max_minutes:.0f} menit."
        )


def delete_audio_file(file_path: str) -> None:
    """Remove an audio file from storage (cleanup on failure)."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass  # Best effort cleanup

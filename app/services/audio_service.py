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
    if file.size and file.size > settings.MAX_UPLOAD_SIZE_BYTES:
        raise AudioValidationError(
            f"Ukuran file melebihi batas maksimum {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    return ext


async def save_audio_file(file: UploadFile, ext: str) -> Tuple[str, str, str]:
    """
    Save the uploaded audio to storage directory.
    If the file is not a WAV, transcode it to WAV to prevent
    parsing/FMOD compatibility errors on both backend and frontend.
    """
    # Ensure storage directory exists
    storage_path = settings.STORAGE_PATH
    storage_path.mkdir(parents=True, exist_ok=True)

    # Temporary unique filename for saving raw upload
    temp_filename = f"temp_{uuid.uuid4().hex}{ext}"
    temp_file_path = storage_path / temp_filename

    # Save raw upload to disk
    content = await file.read()

    # Definitive size check after reading
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise AudioValidationError(
            f"Ukuran file melebihi batas maksimum {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    async with aiofiles.open(temp_file_path, "wb") as f:
        await f.write(content)

    file_format = ext.lstrip(".").lower()

    if file_format != "wav":
        # Transcode to WAV
        target_filename = f"{uuid.uuid4().hex}.wav"
        target_file_path = storage_path / target_filename
        try:
            import librosa
            import soundfile as sf
            # Load with native samplerate and channels
            y, sr = librosa.load(str(temp_file_path), sr=None, mono=False)
            # Write to WAV
            sf.write(str(target_file_path), y.T, sr, subtype='PCM_16')

            # Delete temp raw file
            if temp_file_path.exists():
                temp_file_path.unlink()

            stored_filename = target_filename
            file_path = target_file_path
            file_format = "wav"
            print(f"[AudioService] Transcoded upload {file.filename} ({ext}) to clean WAV: {stored_filename}")
        except Exception as e:
            # Fallback to saving original if transcoding fails
            if target_file_path.exists():
                target_file_path.unlink()
            # Rename temp to original target
            stored_filename = f"{uuid.uuid4().hex}{ext}"
            file_path = storage_path / stored_filename
            temp_file_path.rename(file_path)
            file_format = ext.lstrip(".")
            print(f"[AudioService] Warning: Failed to transcode, fallback to raw: {e}")
    else:
        # Already WAV, just rename temp to original target
        stored_filename = f"{uuid.uuid4().hex}.wav"
        file_path = storage_path / stored_filename
        temp_file_path.rename(file_path)
        file_format = "wav"

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

"""
SQLAlchemy ORM model for the `songs` table.
Stores metadata about uploaded audio files.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Song(Base):
    __tablename__ = "songs"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    song_code = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=True)
    genre = Column(String(100), nullable=True)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_format = Column(
        Enum("mp3", "wav", name="file_format_enum"),
        nullable=False,
    )
    duration_seconds = Column(Float, nullable=True)
    bpm = Column(Float, nullable=True)
    cover_image_path = Column(String(500), nullable=True)
    upload_source = Column(
        Enum("user_upload", "seeded", "admin", name="upload_source_enum"),
        nullable=False,
        default="user_upload",
    )

    process_status = Column(
        Enum("uploaded", "processing", "done", "failed", name="process_status_enum"),
        nullable=False,
        default="uploaded",
    )

    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ─────────────────────────────────────────
    beatmaps = relationship("Beatmap", back_populates="song", lazy="selectin")
    game_results = relationship("GameResult", back_populates="song", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Song id={self.id} code={self.song_code} status={self.process_status}>"

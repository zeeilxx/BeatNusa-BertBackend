"""
SQLAlchemy ORM model for the `beatmaps` table.
Stores generated beatmaps from the AI model.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Beatmap(Base):
    __tablename__ = "beatmaps"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    song_id = Column(BigInteger, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False, index=True)

    model_name = Column(String(100), nullable=False, default="BeatmapBERT")
    model_version = Column(String(100), nullable=False, default="1.0")
    difficulty_name = Column(String(100), nullable=False, default="normal")
    lane_count = Column(Integer, nullable=False, default=4)
    offset_ms = Column(Integer, nullable=False, default=0)

    # Stored as JSON text — the full beatmap payload
    beatmap_json = Column(Text, nullable=False)

    note_count = Column(Integer, nullable=False, default=0)

    generation_status = Column(
        Enum("generated", "validated", "failed", name="generation_status_enum"),
        nullable=False,
        default="generated",
    )

    validation_notes = Column(Text, nullable=True)

    generated_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ─────────────────────────────────────────
    song = relationship("Song", back_populates="beatmaps")
    game_results = relationship("GameResult", back_populates="beatmap", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Beatmap id={self.id} song_id={self.song_id} notes={self.note_count}>"

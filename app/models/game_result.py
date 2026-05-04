"""
SQLAlchemy ORM model for the `game_results` table.
Stores gameplay results submitted from Unity.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base


class GameResult(Base):
    __tablename__ = "game_results"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    song_id = Column(BigInteger, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False, index=True)
    beatmap_id = Column(BigInteger, ForeignKey("beatmaps.id", ondelete="CASCADE"), nullable=False, index=True)

    player_name = Column(String(100), nullable=False)
    score = Column(Integer, nullable=False, default=0)
    accuracy = Column(Float, nullable=False, default=0.0)
    max_combo = Column(Integer, nullable=False, default=0)
    hit_count = Column(Integer, nullable=False, default=0)
    miss_count = Column(Integer, nullable=False, default=0)
    good_count = Column(Integer, nullable=False, default=0)
    perfect_count = Column(Integer, nullable=False, default=0)
    bad_count = Column(Integer, nullable=False, default=0)
    mean_offset_ms = Column(Float, nullable=True)

    played_at = Column(DateTime, nullable=False, server_default=func.now())

    # ── Relationships ─────────────────────────────────────────
    song = relationship("Song", back_populates="game_results")
    beatmap = relationship("Beatmap", back_populates="game_results")

    def __repr__(self) -> str:
        return f"<GameResult id={self.id} player={self.player_name} score={self.score}>"

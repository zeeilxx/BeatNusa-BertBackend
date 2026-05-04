"""
Game result API endpoints.
Handles submission and retrieval of gameplay results.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.song import Song
from app.models.beatmap import Beatmap
from app.models.game_result import GameResult
from app.schemas.game_result import (
    GameResultCreate,
    GameResultResponse,
    LeaderboardEntry,
)

router = APIRouter(prefix="/api/game-results", tags=["Game Results"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST /api/game-results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.post("", response_model=GameResultResponse, status_code=201)
async def submit_game_result(
    payload: GameResultCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit gameplay results from Unity after a session ends.
    Unity sends: song_code, beatmap_id, player stats.
    """
    # Resolve song_code → song_id
    stmt = select(Song).where(Song.song_code == payload.song_code)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise HTTPException(
            status_code=404,
            detail=f"Song {payload.song_code} tidak ditemukan.",
        )

    # Verify beatmap exists
    stmt = select(Beatmap).where(Beatmap.id == payload.beatmap_id)
    result = await db.execute(stmt)
    beatmap = result.scalar_one_or_none()

    if not beatmap:
        raise HTTPException(
            status_code=404,
            detail=f"Beatmap ID {payload.beatmap_id} tidak ditemukan.",
        )

    # Verify beatmap belongs to this song
    if beatmap.song_id != song.id:
        raise HTTPException(
            status_code=400,
            detail=f"Beatmap ID {payload.beatmap_id} bukan milik song {payload.song_code}.",
        )

    # Create game result record
    game_result = GameResult(
        song_id=song.id,
        beatmap_id=payload.beatmap_id,
        player_name=payload.player_name,
        score=payload.score,
        accuracy=payload.accuracy,
        max_combo=payload.max_combo,
        hit_count=payload.hit_count,
        miss_count=payload.miss_count,
        good_count=payload.good_count,
        perfect_count=payload.perfect_count,
        bad_count=payload.bad_count,
        mean_offset_ms=payload.mean_offset_ms,
    )
    db.add(game_result)
    await db.commit()
    await db.refresh(game_result)

    return GameResultResponse(
        id=game_result.id,
        song_code=payload.song_code,
        beatmap_id=game_result.beatmap_id,
        player_name=game_result.player_name,
        score=game_result.score,
        accuracy=game_result.accuracy,
        max_combo=game_result.max_combo,
        hit_count=game_result.hit_count,
        miss_count=game_result.miss_count,
        good_count=game_result.good_count,
        perfect_count=game_result.perfect_count,
        bad_count=game_result.bad_count,
        mean_offset_ms=game_result.mean_offset_ms,
        played_at=game_result.played_at,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /api/game-results/{song_code}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.get("/{song_code}", response_model=List[GameResultResponse])
async def get_game_results(
    song_code: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all gameplay results for a specific song."""
    # Find the song
    stmt = select(Song).where(Song.song_code == song_code)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise HTTPException(
            status_code=404,
            detail=f"Song {song_code} tidak ditemukan.",
        )

    # Fetch results
    stmt = (
        select(GameResult)
        .where(GameResult.song_id == song.id)
        .order_by(desc(GameResult.played_at))
    )
    result = await db.execute(stmt)
    results = result.scalars().all()

    return [
        GameResultResponse(
            id=r.id,
            song_code=song_code,
            beatmap_id=r.beatmap_id,
            player_name=r.player_name,
            score=r.score,
            accuracy=r.accuracy,
            max_combo=r.max_combo,
            hit_count=r.hit_count,
            miss_count=r.miss_count,
            good_count=r.good_count,
            perfect_count=r.perfect_count,
            bad_count=r.bad_count,
            mean_offset_ms=r.mean_offset_ms,
            played_at=r.played_at,
        )
        for r in results
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET /api/game-results/{song_code}/leaderboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@router.get("/{song_code}/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    song_code: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the top scores leaderboard for a specific song.
    Returns top N entries sorted by score (descending).
    """
    # Find the song
    stmt = select(Song).where(Song.song_code == song_code)
    result = await db.execute(stmt)
    song = result.scalar_one_or_none()

    if not song:
        raise HTTPException(
            status_code=404,
            detail=f"Song {song_code} tidak ditemukan.",
        )

    # Fetch top results
    stmt = (
        select(GameResult)
        .where(GameResult.song_id == song.id)
        .order_by(desc(GameResult.score))
        .limit(limit)
    )
    result = await db.execute(stmt)
    results = result.scalars().all()

    return [
        LeaderboardEntry(
            rank=idx + 1,
            player_name=r.player_name,
            score=r.score,
            accuracy=r.accuracy,
            max_combo=r.max_combo,
            played_at=r.played_at,
        )
        for idx, r in enumerate(results)
    ]

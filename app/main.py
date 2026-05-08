"""
FastAPI application factory — main entry point.
"""

import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables

# Ensure 'src' is on the path so beatbert module imports work
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIFESPAN — startup / shutdown logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup:
    1. Create DB tables (idempotent)
    2. Load AI model into memory
    3. Ensure storage directory exists
    """
    print("=" * 60)
    print("  BeatmapBERT Backend — Starting Up")
    print("=" * 60)

    # 1. Create database tables
    # Import models so Base.metadata knows about them
    import app.models  # noqa: F401
    await create_tables()
    print("[Startup] Database tables sudah siap.")

    # 2. Load AI model
    from app.services.ai_service import ai_service
    ai_service.load_model(
        config_path=settings.MODEL_CONFIG,
        checkpoint_path=settings.MODEL_CHECKPOINT,
    )

    # 3. Ensure storage directory exists
    settings.STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[Startup] Storage directory: {settings.STORAGE_PATH.resolve()}")

    print("=" * 60)
    print("  Server siap menerima request!")
    print("=" * 60)

    yield  # ← server runs here

    # Shutdown
    print("[Shutdown] BeatmapBERT Backend stopped.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APP CREATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="BeatmapBERT AI Backend",
    description="Backend API untuk rhythm game WebGL — upload audio, generate beatmap, simpan gameplay result.",
    version="1.0.0",
    lifespan=lifespan,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORS — allow Unity WebGL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.add_middleware(
    CORSMiddleware,
    # Untuk produksi, ganti "*" dengan domain Unity WebGL Anda
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from app.routers import songs, beatmaps, game_results

app.include_router(songs.router)
app.include_router(beatmaps.router)
app.include_router(game_results.router)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.get("/", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    from app.services.ai_service import ai_service
    import torch
    return {
        "status": "AI Server is Alive!",
        "model_loaded": ai_service.is_loaded(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "database": settings.DB_NAME,
    }

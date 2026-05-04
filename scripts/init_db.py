"""
Database initialization script.
Creates the database tables in MySQL.

Usage:
    python scripts/init_db.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main():
    print("=" * 50)
    print("  BeatmapBERT — Database Initialization")
    print("=" * 50)

    from app.config import settings
    from app.database import create_tables, engine

    # Import models so Base.metadata knows about them
    import app.models  # noqa: F401

    print(f"Database: {settings.DB_NAME}")
    print(f"Host:     {settings.DB_HOST}:{settings.DB_PORT}")
    print(f"User:     {settings.DB_USER}")
    print()

    try:
        await create_tables()
        print("[OK] Semua tabel berhasil dibuat!")
        print()
        print("Tabel yang dibuat:")
        print("  - songs")
        print("  - beatmaps")
        print("  - game_results")
    except Exception as e:
        print(f"[FAIL] Gagal membuat tabel: {e}")
        print()
        print("Pastikan:")
        print(f"  1. MySQL berjalan di {settings.DB_HOST}:{settings.DB_PORT}")
        print(f"  2. Database '{settings.DB_NAME}' sudah dibuat")
        print(f"  3. User '{settings.DB_USER}' punya akses")
        sys.exit(1)
    finally:
        await engine.dispose()

    print()
    print("Database siap digunakan!")


if __name__ == "__main__":
    asyncio.run(main())

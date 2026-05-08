"""
Application settings loaded from environment variables / .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized configuration using pydantic-settings.
    Values are loaded from .env file at the project root.
    """

    # ── Database ──────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_NAME: str = "beatmap_game_db"
    DB_SSL: bool = False

    # ── Storage ───────────────────────────────────────────────
    STORAGE_DIR: str = "storage/audio"

    # ── AI Model ──────────────────────────────────────────────
    MODEL_CHECKPOINT: str = "checkpoints/best.pt"
    MODEL_CONFIG: str = "configs/local.yaml"

    # ── Upload Limits ─────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_AUDIO_DURATION_SECONDS: int = 600  # 10 minutes

    @property
    def DATABASE_URL(self) -> str:
        """Async MySQL connection string for SQLAlchemy."""
        password_part = f":{self.DB_PASSWORD}" if self.DB_PASSWORD else ""
        return (
            f"mysql+aiomysql://{self.DB_USER}{password_part}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Sync MySQL connection string (for scripts / migrations)."""
        password_part = f":{self.DB_PASSWORD}" if self.DB_PASSWORD else ""
        url = (
            f"mysql+pymysql://{self.DB_USER}{password_part}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
        if self.DB_SSL:
            url += "?ssl_verify_cert=true&ssl_verify_identity=true"
        return url

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def STORAGE_PATH(self) -> Path:
        return Path(self.STORAGE_DIR)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings instance
settings = Settings()

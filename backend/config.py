from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # JWT
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # ML Model
    MODEL_PATH: str = str(BASE_DIR / "models" / "trained_model.pkl")
    SCALER_PATH: str = str(BASE_DIR / "models" / "scaler.pkl")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR.parent / ".env"), extra="ignore")


settings = Settings()
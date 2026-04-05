from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Axiom-I Image Forensics"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    DEVICE: str = "cpu"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOW_MODEL_DOWNLOAD: bool = False
    VIT_MODEL_NAME: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"

    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    MAX_IMAGE_PIXELS: int = 20_000_000

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AXIOM_", extra="ignore")


settings = Settings()

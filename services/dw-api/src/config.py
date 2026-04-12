import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    ENVIRONMENT: str = "development"  # Fallback if not set in os.environ

    model_config = SettingsConfigDict(
        env_file=f".env.{os.environ.get('ENVIRONMENT', 'development').lower()}",
        env_file_encoding="utf-8",
        extra="ignore",
    )

def get_settings():
    return Settings()

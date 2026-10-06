from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    qdrant_url: str
    qdrant_collection: str = "documents"
    redis_url: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()  
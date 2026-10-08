from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    qdrant_url: str
    qdrant_collection: str = "documents"
    redis_url: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_upload_mb: int = 10

    # API 2 (conversational RAG)
    groq_api_key: SecretStr
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    chat_history_limit: int = 20
    chat_ttl_seconds: int = 3600

    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.20


@lru_cache
def get_settings() -> Settings:
    return Settings()
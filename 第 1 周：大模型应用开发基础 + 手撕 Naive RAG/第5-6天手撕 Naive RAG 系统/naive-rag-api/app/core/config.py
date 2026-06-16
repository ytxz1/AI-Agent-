from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Naive RAG API"
    app_env: str = "dev"
    api_prefix: str = "/api/v1"

    upload_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")
    metadata_dir: Path = Path("data/metadata")
    chroma_collection: str = "naive_rag"

    max_file_size_mb: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 120
    default_top_k: int = 4

    embedding_provider: str = "hash"
    chat_provider: str = "deepseek"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def registry_path(self) -> Path:
        return self.metadata_dir / "documents.json"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    loaded_settings = Settings()
    loaded_settings.ensure_dirs()
    return loaded_settings


settings = get_settings()

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_title: str = "Elite Research Assistant"
    google_api_key: SecretStr | None = None
    google_cse_id: str | None = None
    app_api_key: SecretStr | None = None
    database_path: Path = Path("data/research.db")
    max_sources: int = Field(default=10, ge=3, le=20)
    request_timeout_seconds: float = Field(default=20, ge=5, le=60)
    model_timeout_seconds: float = Field(default=240, ge=30, le=600)
    max_query_length: int = Field(default=500, ge=50, le=2000)
    max_document_chars: int = Field(default=8_000, ge=1_000, le=25_000)
    web_search_enabled: bool = True
    web_search_engine: str = "exa"
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    web_search_cache_minutes: int = Field(default=30, ge=1, le=1440)
    web_fetch_fallback_enabled: bool = True
    web_fetch_engine: str = "openrouter"
    web_fetch_max_fallbacks: int = Field(default=2, ge=0, le=10)

    @property
    def google_search_enabled(self) -> bool:
        return bool(self.google_api_key and self.google_cse_id)

    @property
    def model_enabled(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def live_web_enabled(self) -> bool:
        return bool(self.openrouter_api_key and self.web_search_enabled)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

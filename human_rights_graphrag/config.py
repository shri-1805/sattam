"""Application settings loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for FalkorDB, embeddings, and language models."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    falkordb_host: str = "localhost"
    falkordb_port: int = Field(default=6379, ge=1, le=65535)
    falkordb_password: str | None = None
    falkordb_graph: str = "human_rights"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token: str | None = None
    ai_provider: Literal["groq", "xai"] = "groq"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"
    xai_model: str = "grok-4.7"
    llm_ca_bundle: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LLM_CA_BUNDLE", "GROQ_CA_BUNDLE", "XAI_CA_BUNDLE"
        ),
    )
    llm_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
        le=300,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "XAI_TIMEOUT_SECONDS"),
    )
    llm_verify_tls: bool = True
    retrieval_limit: int = Field(default=8, ge=1, le=30)

    @property
    def llm_api_key(self) -> str | None:
        """Return the API key for the selected provider."""
        return self.groq_api_key if self.ai_provider == "groq" else self.xai_api_key

    @property
    def llm_base_url(self) -> str:
        """Return the base URL for the selected provider."""
        return self.groq_base_url if self.ai_provider == "groq" else self.xai_base_url

    @property
    def llm_model(self) -> str:
        """Return the model for the selected provider."""
        return self.groq_model if self.ai_provider == "groq" else self.xai_model

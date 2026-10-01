from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação, lidas de variáveis de ambiente / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Vazio por padrão para que os testes de guardrails do SQL (app/db.py)
    # possam rodar sem exigir uma chave da OpenRouter. A ausência da chave
    # só é validada quando o agente de fato é construído (ver app/agent.py).
    openrouter_api_key: str = ""
    openrouter_model: str = "poolside/laguna-s-2.1:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    db_path: Path = Path("cinerocket.db")

    # Guardrails de execução de SQL
    sql_max_rows: int = 200
    sql_timeout_seconds: int = 15


settings = Settings()

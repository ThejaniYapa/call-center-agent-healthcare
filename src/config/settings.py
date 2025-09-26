from __future__ import annotations

import os
from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "openai")  # openai | ollama

    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "callcenter")

    mcp_mongo_cmd: str = os.getenv("MCP_MONGO_CMD", "python -m src.mcp.mongo_server")

    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "data/faiss")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


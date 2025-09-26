from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_community.embeddings import OllamaEmbeddings
except Exception:  # pragma: no cover
    OllamaEmbeddings = None  # type: ignore

from src.config.settings import get_settings


def _get_embeddings():
    settings = get_settings()
    if settings.embeddings_provider == "openai":
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAI embeddings not available. Install langchain-openai.")
        return OpenAIEmbeddings(model=settings.openai_embed_model)
    elif settings.embeddings_provider == "ollama":
        if OllamaEmbeddings is None:
            raise RuntimeError("Ollama embeddings not available. Install langchain-community.")
        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {settings.embeddings_provider}")


def build_faiss_index(docs_dir: str | Path, out_dir: str | Path) -> None:
    docs_path = Path(docs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(str(docs_path), glob="**/*", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    vs = FAISS.from_documents(chunks, embedding=embeddings)
    vs.save_local(str(out_path))


def load_faiss_retriever(index_dir: str | Path, k: int = 4):
    embeddings = _get_embeddings()
    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})


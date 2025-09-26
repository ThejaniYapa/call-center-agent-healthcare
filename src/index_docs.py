from __future__ import annotations

import typer
from pathlib import Path
from rich import print
from src.tools.retriever import build_faiss_index


app = typer.Typer(add_completion=False)


@app.command()
def main(docs: str = typer.Option("src/data/docs", help="Docs directory"),
         out: str = typer.Option("data/faiss", help="FAISS index output directory")):
    docs_p = Path(docs)
    out_p = Path(out)
    print(f"[bold cyan]Building FAISS index[/] from {docs_p} -> {out_p}")
    build_faiss_index(docs_p, out_p)
    print("[green]Done.[/]")


if __name__ == "__main__":
    app()


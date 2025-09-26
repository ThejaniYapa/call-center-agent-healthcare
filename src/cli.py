from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from src.agent.graph import build_graph
from src.agent.memory import get_checkpointer
from src.config.settings import get_settings
from src.index_docs import main as index_main
from src.tools.human import human_confirm


app = typer.Typer(add_completion=False)


@app.command()
def serve_mcp():
    """Run the MCP Mongo server (stdio)."""
    from src.mcp.mongo_server import amain
    import anyio

    print("[bold cyan]Starting MCP Mongo server (stdio)...[/]")
    anyio.run(amain)


@app.command()
def seed_mongo():
    """Seed MongoDB with example data for the demo."""
    from src.mcp.seed_mongo import seed
    seed()


@app.command()
def index(docs: str = typer.Option("src/data/docs", help="Docs directory"),
          out: str = typer.Option("data/faiss", help="FAISS index output directory")):
    index_main.callback = None  # silence Typer re-entry issues
    index_main(docs=docs, out=out)


@app.command()
def chat(caller_profile: Optional[Path] = typer.Option(None, exists=True),
         session_id: str = typer.Option("default-session"),
         query: Optional[str] = typer.Option(None, help="Single-turn query; if omitted, enters interactive mode.")):
    """Chat with the agent. Uses FAISS + MCP + prompts + memory."""
    settings = get_settings()
    graph = build_graph()
    checkpointer = get_checkpointer()

    caller = {}
    if caller_profile:
        caller = json.loads(Path(caller_profile).read_text(encoding="utf-8"))

    def run_once(user_text: str):
        state = {
            "messages": [{"type": "human", "content": user_text}],
            "caller_profile": caller,
        }
        result = graph.invoke(state, config={"configurable": {"thread_id": session_id}, "checkpointer": checkpointer})
        print("\n[bold green]Agent:[/]\n" + result.get("answer", "(no answer)"))

    if query:
        print(f"[bold cyan]Caller Profile:[/] {caller}")
        print(f"[bold cyan]User:[/] {query}")
        run_once(query)
        raise typer.Exit(0)

    print("[bold cyan]Interactive chat. Type 'exit' to quit.[/]")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        run_once(user)


@app.command()
def eval():
    from src.eval.run_eval import run_eval
    print("[bold cyan]Running evaluation scenarios...[/]")
    run_eval()


if __name__ == "__main__":
    app()

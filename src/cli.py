from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

import typer
from langchain_core.runnables import RunnableConfig
from rich import print

from src.agent.graph import AgentState, build_graph
from src.agent.memory import get_checkpointer
from src.index_docs import run_index


app = typer.Typer(add_completion=False)


@app.command()
def serve_mcp():
    """Run the MCP Mongo server (stdio)."""
    from src.mcp.mongo_server import mcp

    print("[bold cyan]Starting MCP Mongo server (stdio)...[/]")
    mcp.run()


@app.command()
def seed_mongo():
    """Seed MongoDB with example data for the demo."""
    from src.mcp.seed_mongo import seed
    seed()


@app.command()
def index(docs: str = typer.Option("src/data/docs", help="Docs directory"),
          out: str = typer.Option("data/faiss", help="FAISS index output directory")):
    docs_path = Path(docs)
    out_path = Path(out)
    run_index(docs_path, out_path)


@app.command()
def chat(caller_profile: Optional[Path] = typer.Option(None, exists=True),
         session_id: str = typer.Option("default-session"),
         query: Optional[str] = typer.Option(None, help="Single-turn query; if omitted, enters interactive mode.")):
    """Chat with the agent. Uses FAISS + MCP + prompts + memory."""
    graph = build_graph()
    checkpointer = get_checkpointer()

    caller: Dict[str, Any] = {}
    if caller_profile:
        caller = json.loads(Path(caller_profile).read_text(encoding="utf-8"))

    def run_once(user_text: str) -> None:
        state: AgentState = {
            "messages": [{"type": "human", "content": user_text}],
            "caller_profile": caller,
        }
        graph_config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        if checkpointer is not None:
            extended_config: Dict[str, Any] = dict(graph_config)
            extended_config["checkpointer"] = checkpointer
            graph_config = cast(RunnableConfig, extended_config)
        result = cast(AgentState, graph.invoke(state, config=graph_config))
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

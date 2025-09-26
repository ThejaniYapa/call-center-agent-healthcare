# Agentic AI Call Center Demo

An end-to-end demo to showcase Agentic AI concepts for a real-time call center use case:

- Context engineering and prompt design
- Tool calling (retrieval, MCP tools, human-in-the-loop)
- MCP (Model Context Protocol) server for MongoDB
- Instructions and prompt optimization patterns
- Session management and memory with LangGraph
- Reasoning and multi-step actions (planner → tools → synthesis)
- Evaluation harness for scenarios
- Clean code and folder structure

## Tech

- LangGraph + LangChain (agent + tools + memory)
- FAISS local vector store (FAQ, policies, insurance info)
- MCP Python server exposing MongoDB lookups
- Typer CLI (index, chat, eval, serve-mcp)

## Structure

```
.
├─ src/
│  ├─ agent/
│  │  ├─ graph.py            # LangGraph agent orchestration
│  │  └─ memory.py           # Session management (checkpointer)
│  ├─ config/
│  │  └─ settings.py         # Env settings via Pydantic
│  ├─ mcp/
│  │  └─ mongo_server.py     # MCP server exposing MongoDB tools
│  ├─ tools/
│  │  ├─ retriever.py        # FAISS loader and retrieval tool
│  │  ├─ mcp_client.py       # MCP client wrapper tools
│  │  └─ human.py            # Human-in-the-loop tool
│  ├─ prompts/
│  │  ├─ system.txt          # System/role instructions
│  │  ├─ planner.txt         # Planner prompt
│  │  └─ rewriter.txt        # Query rewrite prompt
│  ├─ eval/
│  │  ├─ scenarios.yaml      # Example eval scenarios
│  │  └─ run_eval.py         # Simple evaluation harness
│  ├─ data/
│  │  └─ docs/               # Example FAQ/policy docs (seed)
│  ├─ cli.py                 # Typer CLI entry
│  └─ index_docs.py          # Build FAISS index from docs
├─ requirements.txt
└─ .env.example
```

## Setup

1) Python 3.10+ recommended.

2) Install deps:

```
pip install -r requirements.txt
```

3) Configure environment variables. Copy `.env.example` to `.env` and set values:

```
cp .env.example .env
```

Required (choose an embedding provider):
- `OPENAI_API_KEY` for OpenAI embeddings / chat
or
- Set `EMBEDDINGS_PROVIDER=ollama` and ensure an embedding model is available locally (e.g., `nomic-embed-text` via Ollama).

MongoDB / MCP:
- `MONGODB_URI` and `MONGODB_DB`

4) Seed FAISS index from docs:

```
python -m src.index_docs --docs src/data/docs --out data/faiss
```

5) Run MCP Mongo server (separate terminal):

```
python -m src.cli serve-mcp
```

6) Chat demo (agent uses FAISS + MCP + human tool):

```
python -m src.cli chat --caller-profile examples/caller_profile.json --session-id demo1
```

During the chat, when a sensitive action is planned, the agent will ask for human confirmation.

7) Evaluation run (toy scoring):

```
python -m src.cli eval
```

## Notes

- This demo is structured for clarity and teaching; it favors explicit steps and prompts.
- Network access is required for OpenAI unless using a fully local stack (e.g., Ollama for LLM and embeddings).
- MCP server is a simple stdio server exposing a few MongoDB operations. The agent calls it via an MCP client tool.
- Session memory persists across turns via a SQLite checkpointer (under `.checkpoints/`).

## Teaching Map

- Context engineering: `src/prompts/*.txt`, retrieved context from FAISS, caller profile, and MCP results are woven into the model input.
- Prompt engineering: planner vs. worker prompts, rewrite prompt, and instruction shaping.
- Tool calling: vector retrieval, MCP tools, human confirmation tool.
- MCP: `src/mcp/mongo_server.py` and `src/tools/mcp_client.py` use MCP over stdio.
- Instruction optimization: separate prompt files and reduced-shot structure.
- Session management: `src/agent/memory.py` via LangGraph checkpointer and CLI `--session-id`.
- Reasoning: planner node produces a plan; the graph executes tools and synthesizes a final reply.
- Multi-step actions: planner → retrieve → MCP lookup → synthesis (+ optional confirmation).
- Human interactions: `HumanConfirmTool` prompts user for approval/inputs mid-execution.
- Evaluation: `src/eval/run_eval.py` runs scenarios and computes simple metrics or LLM-as-judge if configured.

## Streamlit UI

To demo the agent with live tracing, install the dependencies and run:

```
streamlit run streamlit_app.py
```

The app streams the planner output, tool calls (retrieval + MCP), and final synthesis so you can follow each step in the browser. Use the sidebar to load a caller profile JSON and reset the session.

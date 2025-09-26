from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml

from src.agent.graph import build_graph
from src.agent.memory import get_checkpointer


def keyword_score(text: str, keywords: List[str]) -> float:
    t = text.lower()
    hits = sum(1 for k in keywords if k.lower() in t)
    return hits / max(1, len(keywords))


def run_eval():
    scenarios = yaml.safe_load(Path("src/eval/scenarios.yaml").read_text(encoding="utf-8"))
    graph = build_graph()
    checkpointer = get_checkpointer(".checkpoints/eval.db")

    results: List[Dict] = []
    for s in scenarios:
        name = s.get("name")
        caller = s.get("caller_profile", {})
        query = s.get("query", "")
        expected = s.get("expect_keywords", [])
        state = {
            "messages": [{"type": "human", "content": query}],
            "caller_profile": caller,
        }
        out = graph.invoke(state, config={"configurable": {"thread_id": f"eval-{name}"}, "checkpointer": checkpointer})
        answer = out.get("answer", "")
        score = keyword_score(answer, expected)
        results.append({"name": name, "score": score, "answer": answer})

    for r in results:
        print(f"- {r['name']}: score={r['score']:.2f}\n{r['answer'][:400]}\n")


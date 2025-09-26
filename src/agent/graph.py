from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.config.settings import get_settings
from src.tools.retriever import load_faiss_retriever
from src.tools.mcp_client import mcp_get_customer_by_phone, mcp_get_policy_by_number


class AgentState(TypedDict, total=False):
    messages: List[dict]
    caller_profile: Dict[str, Any]
    plan: Dict[str, Any]
    retrieved: List[Dict[str, Any]]
    mcp: Dict[str, Any]
    confirmed: bool
    answer: str


def _llm():
    settings = get_settings()
    return ChatOpenAI(model=settings.openai_model, temperature=0)


def node_plan(state: AgentState) -> AgentState:
    settings = get_settings()
    planner = _llm()
    with open("src/prompts/planner.txt", "r", encoding="utf-8") as f:
        planner_text = f.read()
    prompt = ChatPromptTemplate.from_messages([
        ("system", planner_text),
        ("human", "Caller profile: {caller}\n\nConversation so far: {history}\n\nUser query: {query}\n\nRespond with JSON only."),
    ])
    last_user = ""
    for m in reversed(state.get("messages", [])):
        if m.get("type") == "human":
            last_user = m.get("content", "")
            break
    history = "\n".join([f"{m.get('type')}: {m.get('content')}" for m in state.get("messages", [])])
    chain = prompt | planner | StrOutputParser()
    raw = chain.invoke({"caller": json.dumps(state.get("caller_profile", {})), "history": history, "query": last_user})
    plan = {}
    try:
        plan = json.loads(raw)
    except Exception:
        plan = {"steps": ["RETRIEVE_KNOWLEDGE", "DRAFT_ANSWER"]}
    return {**state, "plan": plan}


def node_retrieve(state: AgentState) -> AgentState:
    retriever = load_faiss_retriever(get_settings().faiss_index_dir)
    # Optionally rewrite query here with another LLM step; keep simple.
    last_user = ""
    for m in reversed(state.get("messages", [])):
        if m.get("type") == "human":
            last_user = m.get("content", "")
            break
    docs = retriever.get_relevant_documents(last_user)
    simple_docs = [{"content": d.page_content, "source": d.metadata.get("source")} for d in docs]
    return {**state, "retrieved": simple_docs}


def node_mongo_mcp(state: AgentState) -> AgentState:
    caller = state.get("caller_profile", {})
    result: Dict[str, Any] = {}
    steps = (state.get("plan") or {}).get("steps", [])
    # Very simple: if phone is present, do a lookup
    if any((isinstance(s, dict) and s.get("action") == "MCP_LOOKUP") or s == "MCP_LOOKUP" for s in steps):
        if "phone" in caller:
            try:
                result["customer"] = mcp_get_customer_by_phone(caller.get("phone"))
            except Exception as e:
                result["error"] = f"MCP get_customer_by_phone failed: {e}"
        if "policy_number" in caller:
            try:
                result["policy"] = mcp_get_policy_by_number(caller.get("policy_number"))
            except Exception as e:
                result.setdefault("error", str(e))
    return {**state, "mcp": result}


def node_summarize(state: AgentState) -> AgentState:
    llm = _llm()
    with open("src/prompts/system.txt", "r", encoding="utf-8") as f:
        system_txt = f.read()
    system = SystemMessage(content=system_txt)
    last_user = ""
    for m in reversed(state.get("messages", [])):
        if m.get("type") == "human":
            last_user = m.get("content", "")
            break
    context_parts: List[str] = []
    if state.get("retrieved"):
        context_parts.append("Knowledge base excerpts:\n" + "\n---\n".join([d["content"][:1000] for d in state["retrieved"]]))
    if state.get("mcp"):
        context_parts.append("Caller data (MCP):\n" + json.dumps(state["mcp"], indent=2))
    context = "\n\n".join(context_parts) or "(no extra context)"

    prompt = ChatPromptTemplate.from_messages([
        system,
        ("human", "Context to consider:\n{context}"),
        ("human", "User: {query}"),
        ("human", "Write a concise, accurate response. Cite policy text when relevant.")
    ])
    chain = prompt | llm | StrOutputParser()
    final = chain.invoke({"context": context, "query": last_user})
    return {**state, "answer": final}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("retrieve", node_retrieve)
    g.add_node("mcp", node_mongo_mcp)
    g.add_node("synthesize", node_summarize)
    
    # Optional human confirm node inline to keep it simple
    def node_human(state: AgentState) -> AgentState:
        from src.tools.human import human_confirm
        needs = any((isinstance(s, dict) and s.get("action") == "HUMAN_CONFIRM") or s == "HUMAN_CONFIRM" for s in (state.get("plan") or {}).get("steps", []))
        confirmed = True
        if needs:
            confirmed = human_confirm("Proceed with the planned sensitive action?", default=False)
        return {**state, "confirmed": confirmed}
    g.add_node("human", node_human)

    g.set_entry_point("plan")

    def needs_retrieval(state: AgentState) -> str:
        steps = (state.get("plan") or {}).get("steps", [])
        wants_retrieval = any((isinstance(s, dict) and s.get("action") == "RETRIEVE_KNOWLEDGE") or s == "RETRIEVE_KNOWLEDGE" for s in steps)
        return "retrieve" if wants_retrieval else "mcp"

    def after_retrieval(state: AgentState) -> str:
        return "mcp"

    def after_mcp(state: AgentState) -> str:
        return "synthesize"

    g.add_conditional_edges("plan", needs_retrieval, {"retrieve": "retrieve", "mcp": "mcp"})
    g.add_edge("retrieve", "mcp")
    g.add_edge("mcp", "human")
    g.add_edge("human", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()

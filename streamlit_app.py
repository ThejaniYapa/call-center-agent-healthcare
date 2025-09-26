from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import streamlit as st
from langchain_core.runnables import RunnableConfig

from src.agent.graph import AgentState, build_graph
from src.agent.memory import get_checkpointer
from src.config.settings import get_settings

st.set_page_config(page_title="Call Center Agent Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def load_agent() -> Tuple[Any, Any]:
    graph = build_graph()
    checkpointer = get_checkpointer()
    return graph, checkpointer


graph, checkpointer = load_agent()
settings = get_settings()


def build_config(session_id: str) -> RunnableConfig:
    config: Dict[str, Any] = {"configurable": {"thread_id": session_id}}
    if checkpointer is not None:
        config["checkpointer"] = checkpointer
    return cast(RunnableConfig, config)


def stream_agent(state: AgentState, session_id: str) -> Iterator[Dict[str, Any]]:
    config = build_config(session_id)
    try:
        for update in graph.stream(state, config=config, stream_mode="updates"):
            yield update
    except AttributeError:
        result = graph.invoke(state, config=config)
        yield result


def describe_documents(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "_No documents retrieved._"
    lines: List[str] = []
    for idx, doc in enumerate(docs[:3], 1):
        source = doc.get("source") or "unknown"
        content = doc.get("content") or ""
        snippet = content.replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280].rstrip() + "..."
        lines.append(f"{idx}. {source}: {snippet}")
    if len(docs) > 3:
        lines.append(f"...and {len(docs) - 3} more")
    return "\n".join(lines)


def describe_mcp(result: Dict[str, Any]) -> str:
    if not result:
        return "_No MCP calls executed._"
    try:
        return "```json\n" + json.dumps(result, indent=2) + "\n```"
    except (TypeError, ValueError):
        return str(result)


def reset_conversation() -> None:
    st.session_state.chat_messages = []
    st.session_state.agent_messages = []


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "caller_profile" not in st.session_state:
    st.session_state.caller_profile = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"
if "caller_profile_raw" not in st.session_state:
    st.session_state.caller_profile_raw = ""


with st.sidebar:
    st.header("Controls")
    st.button("Reset conversation", on_click=reset_conversation, use_container_width=True)
    session_value = st.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_value.strip() or "streamlit-session"
    if not settings.openai_api_key:
        st.warning("OPENAI_API_KEY is not set. Set it in your environment before running the agent.")

    st.subheader("Caller Profile")
    uploaded = st.file_uploader("Upload JSON profile", type=["json"])
    if uploaded is not None:
        st.session_state.caller_profile_raw = uploaded.read().decode("utf-8")
    profile_text = st.text_area(
        "Caller profile JSON",
        value=st.session_state.caller_profile_raw,
        height=200,
    )
    st.session_state.caller_profile_raw = profile_text
    if st.button("Apply profile", use_container_width=True):
        try:
            profile = json.loads(profile_text) if profile_text.strip() else {}
            st.session_state.caller_profile = profile
            st.success("Profile updated.")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
    if st.session_state.caller_profile:
        st.caption("Active profile")
        st.json(st.session_state.caller_profile)


st.title("Customer Support Agent Demo")
st.caption("Streamlit UI for the LangGraph-powered agent with live trace of planning and tool calls.")


for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask the agent a question")

if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.agent_messages + [{"type": "human", "content": prompt}]
    caller_profile = st.session_state.caller_profile

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        with st.expander("Agent trace", expanded=True):
            plan_placeholder = st.empty()
            retrieval_placeholder = st.empty()
            mcp_placeholder = st.empty()
            log_placeholder = st.empty()

        status_placeholder.info("Planning next steps...")
        plan_placeholder.markdown("_Waiting for plan..._")
        retrieval_placeholder.markdown("_Waiting for retrieval results..._")
        mcp_placeholder.markdown("_Waiting for MCP calls..._")
        log_lines: List[str] = ["Agent run started."]
        log_placeholder.markdown("\n".join(f"- {line}" for line in log_lines))

        final_answer: Optional[str] = None
        final_state: Optional[Dict[str, Any]] = None
        try:
            for update in stream_agent({"messages": history, "caller_profile": caller_profile}, st.session_state.session_id):
                if not update:
                    continue
                final_state = update

                if "plan" in update and update["plan"]:
                    plan_placeholder.json(update["plan"])
                    log_lines.append("Plan generated.")
                    status_placeholder.info("Plan ready. Executing tools...")

                if "retrieved" in update:
                    docs = update.get("retrieved") or []
                    retrieval_placeholder.markdown(describe_documents(docs))
                    log_lines.append(f"Retrieved {len(docs)} documents.")

                if "mcp" in update:
                    mcp_placeholder.markdown(describe_mcp(update.get("mcp") or {}))
                    log_lines.append("MCP lookup complete.")

                if "confirmed" in update:
                    confirmed = bool(update.get("confirmed"))
                    log_lines.append("Human confirmation granted." if confirmed else "Human confirmation denied.")

                if "answer" in update and update["answer"]:
                    final_answer = str(update["answer"])
                    answer_placeholder.markdown(final_answer)
                    status_placeholder.success("Response ready.")
                    log_lines.append("Assistant response prepared.")

                log_placeholder.markdown("\n".join(f"- {line}" for line in log_lines))
        except Exception as err:
            status_placeholder.error("Agent run failed. Check logs for details.")
            log_lines.append(f"Error: {err}")
            log_placeholder.markdown("\n".join(f"- {line}" for line in log_lines))
            st.session_state.agent_messages = history
        else:
            if final_answer is None and final_state:
                answer = final_state.get("answer")
                if answer:
                    final_answer = str(answer)
                    answer_placeholder.markdown(final_answer)
                    status_placeholder.success("Response ready.")
                    log_lines.append("Assistant response prepared.")
                    log_placeholder.markdown("\n".join(f"- {line}" for line in log_lines))

            if final_answer:
                st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})
                history.append({"type": "ai", "content": final_answer})
            st.session_state.agent_messages = history

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.config.settings import get_settings


try:
    from mcp.client.session import Session
    from mcp.client.stdio import StdioServerParameters, connect_stdio
except Exception as e:  # pragma: no cover
    Session = None  # type: ignore
    connect_stdio = None  # type: ignore
    StdioServerParameters = None  # type: ignore


@dataclass
class MCPConfig:
    command: str

    @property
    def argv(self) -> list[str]:
        return shlex.split(self.command)


async def _call_mcp_tool(tool: str, args: Dict[str, Any]) -> Any:
    settings = get_settings()
    if Session is None or connect_stdio is None or StdioServerParameters is None:
        raise RuntimeError("mcp client interfaces not available. Install 'mcp'.")

    params = StdioServerParameters(command=getattr(settings, "mcp_mongo_cmd", "python -m src.mcp.mongo_server").split())
    async with connect_stdio(params) as (read, write):
        async with Session(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            # result is expected to be a list of contents; normalize to string
            # Many MCP servers return a JSON-string payload as first text item.
            if isinstance(result, list) and result:
                item = result[0]
                # Accept dict content or object with `.text`
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    return text_attr
            return result


def _ensure_json(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if isinstance(text, str) and text.strip():
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}
    return {}


def mcp_get_customer_by_phone(phone: str) -> Dict[str, Any]:
    text = asyncio.run(_call_mcp_tool("get_customer_by_phone", {"phone": phone}))
    return _ensure_json(text)


def mcp_get_policy_by_number(policy_number: str) -> Dict[str, Any]:
    text = asyncio.run(_call_mcp_tool("get_policy_by_number", {"policy_number": policy_number}))
    return _ensure_json(text)

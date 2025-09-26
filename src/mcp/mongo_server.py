from __future__ import annotations

import json
from typing import Any, Dict

from pymongo import MongoClient

from src.config.settings import get_settings
from typing import Any, cast

# MCP server (stdio) exposing basic MongoDB tools.
# This implementation uses the base mcp server API and stdio transport.

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent
except Exception as e:  # pragma: no cover
    raise RuntimeError("mcp package is required to run the MCP server") from e


settings = get_settings()
client = MongoClient(settings.mongodb_uri)
db = client[settings.mongodb_db]

# Pylance typing workaround: Server has a dynamic `.tool` decorator not declared in stubs.
# Casting to Any avoids reportAttributeAccessIssue while preserving runtime behavior.
server = cast(Any, Server("mongo-mcp"))


@server.tool()
async def get_customer_by_phone(phone: str) -> TextContent:
    """Lookup a customer by phone number. Returns JSON as a text content."""
    customer = db.customers.find_one({"phone": phone}, {"_id": 0})
    return TextContent(type="text", text=json.dumps(customer or {}))


@server.tool()
async def get_policy_by_number(policy_number: str) -> TextContent:
    """Lookup a policy by policy number. Returns JSON as a text content."""
    policy = db.policies.find_one({"policy_number": policy_number}, {"_id": 0})
    return TextContent(type="text", text=json.dumps(policy or {}))


async def amain() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write)


if __name__ == "__main__":
    import anyio

    anyio.run(amain)

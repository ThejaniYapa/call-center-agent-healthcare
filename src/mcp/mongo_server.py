from __future__ import annotations

import json
from typing import Any, Dict

from pymongo import MongoClient

from src.config.settings import get_settings
from typing import Any, cast
from mcp.types import TextContent
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("mongo-mcp")

settings = get_settings()
client = MongoClient(settings.mongodb_uri)
db = client[settings.mongodb_db]



@mcp.tool()
async def get_customer_by_phone(phone: str) -> TextContent:
    """Lookup a customer by phone number. Returns JSON as a text content."""
    customer = db.customers.find_one({"phone": phone}, {"_id": 0})
    return TextContent(type="text", text=json.dumps(customer or {}))


@mcp.tool()
async def get_policy_by_number(policy_number: str) -> TextContent:
    """Lookup a policy by policy number. Returns JSON as a text content."""
    policy = db.policies.find_one({"policy_number": policy_number}, {"_id": 0})
    return TextContent(type="text", text=json.dumps(policy or {}))


# if __name__ == "__main__":
#     mcp.run()
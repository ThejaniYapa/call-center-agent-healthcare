from __future__ import annotations

from pymongo import MongoClient

from src.config.settings import get_settings


def seed():
    s = get_settings()
    client = MongoClient(s.mongodb_uri)
    db = client[s.mongodb_db]

    db.customers.create_index("phone", unique=True)
    db.policies.create_index("policy_number", unique=True)

    db.customers.update_one(
        {"phone": "+1-555-555-1212"},
        {"$set": {"name": "Alex Parker", "phone": "+1-555-555-1212", "policy_number": "PC-123456"}},
        upsert=True,
    )
    db.policies.update_one(
        {"policy_number": "PC-123456"},
        {"$set": {"policy_number": "PC-123456", "type": "auto", "deductible_comprehensive": 500}},
        upsert=True,
    )

    db.customers.update_one(
        {"phone": "+1-555-555-3434"},
        {"$set": {"name": "Jamie Rivera", "phone": "+1-555-555-3434", "policy_number": "PC-654321"}},
        upsert=True,
    )
    db.policies.update_one(
        {"policy_number": "PC-654321"},
        {"$set": {"policy_number": "PC-654321", "type": "auto", "accident_claim_window_days": 30}},
        upsert=True,
    )

    print("Seeded MongoDB with example customers and policies.")


if __name__ == "__main__":
    seed()


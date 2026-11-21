"""Saves one record per LLM request to MongoDB."""

import json
import os
import time
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError

FALLBACK_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "usage_fallback.jsonl"
)


class UsageLogger:

    def __init__(
        self,
        mongo_uri=None,
        db_name="llm_monitoring",
        collection="api_usage"
    ):
        mongo_uri = mongo_uri or os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017"
        )

        self.client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000
        )

        self.collection = self.client[db_name][collection]

        try:
            self.client.admin.command("ping")
            print("Connected to MongoDB.")

        except PyMongoError as error:
            print(
                f"Warning: MongoDB not reachable "
                f"({error.__class__.__name__}). "
                f"Records will be saved to {FALLBACK_FILE}"
            )

    def log_call(self, record):

        record["timestamp"] = datetime.now(timezone.utc)

        for attempt in range(3):

            try:
                self.collection.insert_one(dict(record))
                print("Usage details saved to MongoDB.")
                return True

            except PyMongoError as error:

                print(
                    f"MongoDB write attempt {attempt + 1} failed: "
                    f"{error.__class__.__name__}"
                )

                if attempt < 2:
                    time.sleep(2 ** attempt)

        with open(FALLBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        print("Warning: saved to local fallback file instead.")
        return False

    def close(self):
        self.client.close()
"""Checks Ollama, model and MongoDB setup."""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

ok = True

url = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

model = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

# Check Ollama and model

try:

    names = [
        m["name"]
        for m in requests.get(
            url + "/api/tags",
            timeout=5
        ).json().get("models", [])
    ]

    print(f"[OK] Ollama is running at {url}")

    if any(n == model or n.startswith(model + ":") for n in names):

        print(f"[OK] Model '{model}' is downloaded")

    else:

        ok = False
        print(
            f"[FAIL] Model '{model}' not found. "
            f"Run: ollama pull {model}"
        )

except requests.exceptions.RequestException:

    ok = False
    print(
        f"[FAIL] Cannot reach Ollama at {url}. "
        "Start the Ollama app, then retry."
    )

# Check MongoDB

from pymongo import MongoClient

try:

    MongoClient(
        os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017"
        ),
        serverSelectionTimeoutMS=4000
    ).admin.command("ping")

    print("[OK] MongoDB is reachable")

except Exception as e:

    ok = False

    print(
        f"[FAIL] MongoDB not reachable "
        f"({type(e).__name__}). Check MONGODB_URI in .env"
    )

sys.exit(0 if ok else 1)
import json
import os


def load(db_path: str = "metadata.json") -> dict:
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(database: dict, db_path: str = "database.json") -> None:
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    print(f"Database saved: {db_path}")


def add_metadata(database: dict, category, metadata: dict) -> dict:
    database.setdefault("metadata", {})
    database["metadata"].setdefault(category, {})
    database["metadata"][category][metadata["audio_file"]] = metadata
    return database

def get_existing_windows(database: dict, category: str) -> list:
    database.setdefault("data", {})
    database["data"].setdefault(category, {})
    existing = database["data"][category]
    return [e["windows"] for e in existing.values()]
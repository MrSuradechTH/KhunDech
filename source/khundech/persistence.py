# Persistence helpers for bot memory, history, and JSON data files.
#
# This file is part of KhunDech.
# KhunDech is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json
import os
from datetime import datetime

from khundech.config import DATA_DIR


os.makedirs(DATA_DIR, exist_ok=True)


def _path(filename: str) -> str:
    # Build absolute path under the runtime data directory.
    return os.path.join(DATA_DIR, filename)


def load_json(filename: str, default):
    # Load JSON file from data directory, returning `default` when missing.
    file_path = _path(filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return default


def save_json(filename: str, data) -> None:
    # Persist JSON data to disk using UTF-8 and pretty formatting.
    with open(_path(filename), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_memory() -> dict:
    # Load the global memory state used by the assistant.
    return load_json(
        "memory.json",
        {
            "learned_facts": [],
            "total_messages": 0,
            "first_seen": datetime.now().isoformat(),
        },
    )


def save_memory(memory: dict) -> None:
    # Save global memory state.
    save_json("memory.json", memory)


def load_history(user_id: int) -> list:
    # Load per-user conversation history.
    return load_json(f"history_{user_id}.json", [])


def save_history(user_id: int, history: list) -> None:
    # Save per-user conversation history with a bounded size.
    if len(history) > 200:
        history = history[-200:]
    save_json(f"history_{user_id}.json", history)
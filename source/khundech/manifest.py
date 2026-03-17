# Skills manifest utilities.
# The manifest provides a compact map of project files and capabilities that
# the assistant can include in prompts for grounded behavior.
#
# This file is part of KhunDech.
# KhunDech is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json
from pathlib import Path

from khundech.config import SOURCE_DIR


MANIFEST_PATH = Path(SOURCE_DIR) / "skills_manifest.json"

DEFAULT_SKILLS_MANIFEST = {
    "project": "KhunDech",
    "files": [
        {"path": "bot.py", "role": "Discord entrypoint and command wiring."},
        {"path": "khundech/config.py", "role": "Environment configuration and runtime paths."},
        {"path": "khundech/persistence.py", "role": "Persistent memory and chat history storage."},
        {"path": "khundech/set_finance.py", "role": "SET stock financial data fetching and formatting."},
        {"path": "khundech/runtime_info.py", "role": "Runtime/environment detection and Docker state facts."},
        {"path": "khundech/docker_tools.py", "role": "Docker CLI command wrappers used by Discord commands."},
        {"path": "khundech/self_improve.py", "role": "Self-upgrade engine for create/update/delete file operations."},
        {"path": "khundech/manifest.py", "role": "Loads and maintains the bot skill/path manifest."},
        {"path": "skills_manifest.json", "role": "Project path and skill manifest that the bot can read."},
        {"path": "requirements.txt", "role": "Python runtime dependencies."},
        {"path": "Dockerfile", "role": "Container build definition."}
    ],
    "skills": [
        {"name": "conversation", "description": "Respond to Discord messages using Gemini with memory context."},
        {"name": "memory", "description": "Persist learned facts and per-user chat history under data/."},
        {"name": "daily_reports", "description": "Generate and post a scheduled daily report to the configured Discord channel."},
        {"name": "set_financial_data", "description": "Fetch SET stock quote and financial statement data with !setfinancial or !setbalance."},
        {"name": "runtime_grounding", "description": "Report runtime and Docker facts from physical files and container state only."},
        {"name": "docker_control", "description": "Run docker and docker compose commands via Discord for allowed user."},
        {"name": "path_verification", "description": "Check and verify workspace file paths before and after self-upgrade operations."},
        {"name": "self_improvement", "description": "Modify the workspace with !improve or !upgrade by creating, updating, or deleting text/code files safely."}
    ]
}


def ensure_skills_manifest() -> None:
    # Create default skills manifest when it does not exist.
    if MANIFEST_PATH.exists():
        return
    MANIFEST_PATH.write_text(
        json.dumps(DEFAULT_SKILLS_MANIFEST, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_skills_manifest() -> dict:
    # Load manifest from disk and auto-heal corrupt content.
    ensure_skills_manifest()
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        MANIFEST_PATH.write_text(
            json.dumps(DEFAULT_SKILLS_MANIFEST, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return DEFAULT_SKILLS_MANIFEST


def build_skills_context(manifest: dict) -> str:
    # Render manifest into plain-text context for AI prompts.
    skills = manifest.get("skills", [])
    files = manifest.get("files", [])
    lines = ["Skills:"]
    for skill in skills:
        lines.append(f"- {skill['name']}: {skill['description']}")
    lines.append("Files:")
    for file_info in files:
        lines.append(f"- {file_info['path']}: {file_info['role']}")
    return "\n".join(lines)
# Scheduled task parsing and persistence helpers.

import re
from datetime import datetime

from khundech.persistence import load_json, save_json


# Schema:
# {
#   "tasks": [
#     {
#       "id": 1,
#       "name": "stock seeking",
#       "prompt": "...",
#       "interval": 6,
#       "unit": "hours",
#       "created_at": "...",
#       "last_run": null,
#       "active": true
#     }
#   ]
# }


UNIT_ALIASES = {
    "minute": "minutes",
    "minutes": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "นาที": "minutes",
    "hour": "hours",
    "hours": "hours",
    "hr": "hours",
    "hrs": "hours",
    "ชั่วโมง": "hours",
    "day": "days",
    "days": "days",
    "วัน": "days",
    "month": "months",
    "months": "months",
    "เดือน": "months",
    "year": "years",
    "years": "years",
    "ปี": "years",
}


def load_scheduled_tasks() -> dict:
    # Load scheduled task store and normalize root structure.
    data = load_json("tasks.json", {"tasks": []})
    if not isinstance(data, dict):
        return {"tasks": []}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return {"tasks": []}
    return data


def save_scheduled_tasks(data: dict) -> None:
    # Persist scheduled tasks to disk.
    save_json("tasks.json", data)


def _normalize_unit(unit_text: str | None) -> str | None:
    # Map user-provided interval unit to canonical unit name.
    if not unit_text:
        return None
    return UNIT_ALIASES.get(unit_text.strip().lower())


def _extract_interval(text: str) -> tuple[int, str] | None:
    # Extract schedule interval from natural language text.
    interval_match = re.search(
        r"(?:every|ervery|run\s+every|run\s+ervery|ทุก)\s*(\d+)\s*(minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี)",
        text,
        re.I,
    )
    if not interval_match:
        return None
    interval = int(interval_match.group(1))
    unit = _normalize_unit(interval_match.group(2))
    if interval <= 0 or not unit:
        return None
    return interval, unit


def _extract_quoted_value(text: str, labels: list[str]) -> str | None:
    # Extract a quoted value after any supported label.
    for label in labels:
        pattern = rf"{label}\s*[\"']([^\"']+)[\"']"
        match = re.search(pattern, text, re.I)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def parse_task_add_intent(text: str) -> dict | None:
    # Parse natural language into structured `add task` payload.
    lower = (text or "").lower()
    if not lower:
        return None

    task_markers = [
        "add task",
        "create task",
        "new task",
        "task name",
        "add schedule",
        "schedule task",
        "เพิ่ม task",
        "ตั้ง task",
        "เพิ่มงาน",
        "ตั้งงาน",
    ]
    if not any(marker in lower for marker in task_markers):
        return None

    task_name = _extract_quoted_value(text, ["task\\s*name", "name", "ชื่อ\\s*task", "ชื่อ"])
    prompt = _extract_quoted_value(text, ["prompt", "pormt", "promt", "task\\s*prompt"])

    if not task_name:
        name_match = re.search(
            r"(?:task\s*name|name)\s*[:=]?\s*([a-zA-Z0-9ก-๙ _-]{2,80}?)(?:\s+(?:and\s+this\s+is\s+)?(?:prompt|pormt|promt)\b|\s+every\b|\s+ervery\b|$)",
            text,
            re.I,
        )
        if name_match:
            task_name = name_match.group(1).strip()

    if not prompt:
        prompt_match = re.search(
            r"(?:prompt|pormt|promt)\s*(?:for\s*the\s*task)?\s*[:=]?\s*([\s\S]+?)(?:\s+(?:every|ervery|run\s+every|run\s+ervery|ทุก)\s*\d+\s*(?:minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี)|$)",
            text,
            re.I,
        )
        if prompt_match:
            prompt = prompt_match.group(1).strip().strip('"').strip("'")
    interval_info = _extract_interval(text)

    if not task_name or not prompt or not interval_info:
        return None

    prompt = re.sub(r"\s+and\s+the\s+task\s+shoulde?\s+be\s*$", "", prompt, flags=re.I).strip()

    interval, unit = interval_info
    return {
        "name": task_name,
        "prompt": prompt,
        "interval": interval,
        "unit": unit,
    }


def add_scheduled_task(name: str, prompt: str, interval: int, unit: str) -> dict:
    # Create and persist one scheduled task entry.
    data = load_scheduled_tasks()
    tasks_list = data.setdefault("tasks", [])

    next_id = 1
    if tasks_list:
        existing_ids = [item.get("id", 0) for item in tasks_list if isinstance(item, dict)]
        next_id = max([value for value in existing_ids if isinstance(value, int)] + [0]) + 1

    task_entry = {
        "id": next_id,
        "name": name.strip(),
        "prompt": prompt.strip(),
        "interval": int(interval),
        "unit": unit,
        "created_at": datetime.now().isoformat(),
        "last_run": None,
        "active": True,
    }
    tasks_list.append(task_entry)
    save_scheduled_tasks(data)
    return task_entry


async def run_task(bot, task_def):
    # Placeholder async runner (task execution is handled by bot loop).
    return None

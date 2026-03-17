# Auto-learning job management.
# This module stores and parses user-managed learning jobs that drive periodic
# knowledge generation cycles.

import re
from datetime import datetime

from khundech.persistence import load_json, save_json


AUTO_LEARN_FILE = "auto_learning_jobs.json"
DEFAULT_JOB_NAME = "set_financial"
DEFAULT_SEED_QUESTION = "what knowledge i shoulde be know about book value of set stock"
DEFAULT_PURPOSE = "SET stock financial analysis"


AUTO_LEARN_KEYWORDS = [
    "auto learn",
    "auto lern",
    "autolearn",
    "autolern",
]


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
}


def _slugify(value: str) -> str:
    # Convert text to a filesystem-safe slug.
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return slug or "auto_learn"


def _normalize_job(job: dict) -> dict:
    # Normalize one job record and apply defaults for missing fields.
    normalized = dict(job or {})
    normalized["name"] = str(normalized.get("name", "")).strip()
    normalized["purpose"] = str(normalized.get("purpose") or normalized.get("topic") or DEFAULT_PURPOSE).strip()
    normalized["seed_question"] = str(normalized.get("seed_question") or DEFAULT_SEED_QUESTION).strip()
    normalized["interval"] = int(normalized.get("interval") or 10)
    normalized["unit"] = _normalize_unit(str(normalized.get("unit") or "minutes")) or "minutes"
    normalized["active"] = bool(normalized.get("active", True))
    normalized["last_run"] = normalized.get("last_run")
    normalized["created_at"] = str(normalized.get("created_at") or datetime.now().isoformat())
    return normalized


def load_auto_learning_jobs() -> dict:
    # Load all auto-learning jobs from persistent storage.
    data = load_json(AUTO_LEARN_FILE, {"jobs": []})
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        return {"jobs": []}
    data["jobs"] = [_normalize_job(job) for job in data.get("jobs", []) if isinstance(job, dict)]
    return data


def save_auto_learning_jobs(data: dict) -> None:
    # Save all auto-learning jobs.
    save_json(AUTO_LEARN_FILE, data)


def ensure_default_auto_learning_job() -> dict:
    # Ensure the default financial learner exists and return it.
    data = load_auto_learning_jobs()
    jobs = data.setdefault("jobs", [])
    for job in jobs:
        if isinstance(job, dict) and str(job.get("name", "")).strip().lower() == DEFAULT_JOB_NAME:
            return job

    job = {
        "id": 1,
        "name": DEFAULT_JOB_NAME,
        "purpose": DEFAULT_PURPOSE,
        "interval": 10,
        "unit": "minutes",
        "seed_question": DEFAULT_SEED_QUESTION,
        "active": True,
        "last_run": None,
        "created_at": datetime.now().isoformat(),
    }
    jobs.append(job)
    save_auto_learning_jobs(data)
    return job


def _normalize_unit(unit_text: str | None) -> str | None:
    # Map free-form unit text to canonical interval units.
    if not unit_text:
        return None
    return UNIT_ALIASES.get(unit_text.strip().lower())


def _extract_interval(text: str):
    # Extract interval payload from text, supporting `every` and labeled forms.
    match = re.search(r"(?:every|run\s+every|ทุก)\s*(\d+)\s*(minutes?|mins?|min|hours?|hrs?|hr|days?|นาที|ชั่วโมง|วัน)", text, re.I)
    if not match:
        match = re.search(r"(?:interval|frequency|schedule)\s*[:=]\s*(\d+)\s*(minutes?|mins?|min|hours?|hrs?|hr|days?|นาที|ชั่วโมง|วัน)", text, re.I)
    if not match:
        return None
    interval = int(match.group(1))
    unit = _normalize_unit(match.group(2))
    if interval <= 0 or not unit:
        return None
    return interval, unit


def _extract_quoted_value(text: str, labels: list[str]) -> str | None:
    # Extract quoted value after one of provided labels.
    for label in labels:
        match = re.search(rf"{label}\s*[\"']([^\"']+)[\"']", text, re.I)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _extract_labeled_value(text: str, labels: list[str]) -> str | None:
    # Extract value from line-based `label: value` syntax.
    lines = (text or "").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        for label in labels:
            pattern = rf"^(?:[-*]\s*)?(?:{label})\s*[:=]\s*(.+)$"
            match = re.match(pattern, line, re.I)
            if match:
                value = match.group(1).strip().strip('"\'')
                if value:
                    return value
    return None


def _extract_active_state(text: str) -> bool | None:
    # Infer enabled/disabled state from natural-language commands.
    lower = (text or "").lower()
    if any(token in lower for token in [" disable", " disabled", "disable it", "turn off", "pause", "inactive", "ปิด"]):
        return False
    if any(token in lower for token in [" enable", " enabled", "enable it", "turn on", "resume", "active", "เปิด"]):
        return True
    return None


def parse_auto_learning_add_intent(text: str) -> dict | None:
    # Parse add/create auto-learning command into structured payload.
    lower = (text or "").lower()
    if not lower:
        return None
    markers = [
        "add auto learn",
        "add auto lern",
        "add more auto learn",
        "add more auto lern",
        "add more autolearn",
        "add more autolern",
        "create auto learn",
        "create auto lern",
        "new auto learn",
        "new auto lern",
        "เพิ่ม auto learn",
        "สร้าง auto learn",
    ]
    if not any(marker in lower for marker in markers):
        return None

    name = _extract_quoted_value(text, [r"name", r"auto\s*learn\s*name"]) or _extract_labeled_value(text, [r"name", r"auto\s*learn\s*name"])
    purpose = _extract_quoted_value(text, [r"purpose", r"topic", r"about", r"for"]) or _extract_labeled_value(text, [r"purpose", r"topic"])
    seed_question = _extract_quoted_value(text, [r"init\s*prompt", r"prompt", r"question", r"seed", r"first\s*question"]) or _extract_labeled_value(text, [r"init\s*question", r"init\s*prompt", r"first\s*question", r"question", r"prompt", r"seed"])
    interval_info = _extract_interval(text)
    if not interval_info:
        return None
    interval, unit = interval_info
    final_name = (name or _slugify(purpose or "auto_learn")).strip()
    return {
        "name": final_name,
        "purpose": purpose or DEFAULT_PURPOSE,
        "seed_question": seed_question or f"What should I learn first about {purpose or final_name}?",
        "interval": interval,
        "unit": unit,
        "active": _extract_active_state(text) if _extract_active_state(text) is not None else True,
    }


def parse_auto_learning_update_intent(text: str) -> dict | None:
    # Parse update command for interval, prompt, purpose, and active state.
    lower = (text or "").lower()
    if not lower or not any(keyword in lower for keyword in AUTO_LEARN_KEYWORDS) or not any(token in lower for token in ["update", "change", "edit", "enable", "disable", "turn on", "turn off", "pause", "resume", "active", "inactive", "เปิด", "ปิด"]):
        return None
    name = _extract_quoted_value(text, [r"name", r"auto\s*learn\s*name"]) or _extract_labeled_value(text, [r"name", r"auto\s*learn\s*name"])
    interval_info = _extract_interval(text)
    seed_question = _extract_quoted_value(text, [r"init\s*prompt", r"prompt", r"question", r"seed", r"next\s*question"]) or _extract_labeled_value(text, [r"init\s*question", r"init\s*prompt", r"next\s*question", r"question", r"prompt", r"seed"])
    purpose = _extract_quoted_value(text, [r"purpose", r"topic", r"about", r"for"]) or _extract_labeled_value(text, [r"purpose", r"topic"])
    active = _extract_active_state(text)
    if not name:
        return None
    payload = {"name": name}
    if interval_info:
        payload["interval"], payload["unit"] = interval_info
    if seed_question:
        payload["seed_question"] = seed_question
    if purpose:
        payload["purpose"] = purpose
    if active is not None:
        payload["active"] = active
    if len(payload) == 1:
        return None
    return payload


def parse_auto_learning_delete_intent(text: str) -> str | None:
    # Parse delete/remove command and return target job name.
    lower = (text or "").lower()
    if not lower:
        return None
    markers = ["delete auto learn", "delete auto lern", "remove auto learn", "remove auto lern", "ลบ auto learn"]
    if not any(marker in lower for marker in markers):
        return None
    return _extract_quoted_value(text, [r"name", r"auto\s*learn\s*name"]) or _extract_labeled_value(text, [r"name", r"auto\s*learn\s*name"])


def is_auto_learning_list_intent(text: str) -> bool:
    # Check whether input asks to list or count auto-learning jobs.
    lower = (text or "").lower()
    if not any(keyword in lower for keyword in AUTO_LEARN_KEYWORDS):
        return False
    list_markers = [
        "show auto learn",
        "list auto learn",
        "show autolearn",
        "list autolearn",
        "how many auto",
        "auto learn count",
        "auto lern count",
        "กี่ auto",
        "auto learn มีอะไร",
        "auto lern มีอะไร",
    ]
    return any(marker in lower for marker in list_markers)


def add_auto_learning_job(name: str, purpose: str, seed_question: str, interval: int, unit: str, active: bool = True) -> dict:
    # Create and persist a new auto-learning job.
    data = load_auto_learning_jobs()
    jobs = data.setdefault("jobs", [])
    for job in jobs:
        if isinstance(job, dict) and str(job.get("name", "")).strip().lower() == name.strip().lower():
            raise ValueError(f"Auto-learning job already exists: {name}")
    next_id = max([job.get("id", 0) for job in jobs if isinstance(job, dict) and isinstance(job.get("id"), int)] + [0]) + 1
    entry = {
        "id": next_id,
        "name": name.strip(),
        "purpose": purpose.strip(),
        "seed_question": seed_question.strip(),
        "interval": int(interval),
        "unit": _normalize_unit(unit) or "minutes",
        "active": bool(active),
        "last_run": None,
        "created_at": datetime.now().isoformat(),
    }
    jobs.append(entry)
    save_auto_learning_jobs(data)
    return entry


def update_auto_learning_job(
    name: str,
    interval: int | None = None,
    unit: str | None = None,
    seed_question: str | None = None,
    purpose: str | None = None,
    active: bool | None = None,
) -> dict | None:
    # Update one existing auto-learning job by name.
    data = load_auto_learning_jobs()
    jobs = data.get("jobs", [])
    target = None
    for job in jobs:
        if isinstance(job, dict) and str(job.get("name", "")).strip().lower() == name.strip().lower():
            target = job
            break
    if target is None:
        return None
    if interval is not None:
        target["interval"] = int(interval)
    if unit is not None:
        target["unit"] = _normalize_unit(unit) or target.get("unit") or "minutes"
    if seed_question:
        target["seed_question"] = seed_question.strip()
    if purpose:
        target["purpose"] = purpose.strip()
    if active is not None:
        target["active"] = bool(active)
    save_auto_learning_jobs(data)
    return target


def delete_auto_learning_job(name: str) -> bool:
    # Delete one auto-learning job by name.
    data = load_auto_learning_jobs()
    jobs = data.get("jobs", [])
    filtered = [job for job in jobs if not (isinstance(job, dict) and str(job.get("name", "")).strip().lower() == name.strip().lower())]
    if len(filtered) == len(jobs):
        return False
    data["jobs"] = filtered
    save_auto_learning_jobs(data)
    return True


def list_auto_learning_jobs() -> list[dict]:
    # Return normalized list of auto-learning jobs.
    data = load_auto_learning_jobs()
    return [job for job in data.get("jobs", []) if isinstance(job, dict)]

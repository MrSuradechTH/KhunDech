import json
import py_compile
from datetime import datetime
from pathlib import Path

from khundech.config import DATA_DIR, SOURCE_DIR


BACKUPS_DIR = Path(DATA_DIR) / "backups"
SELF_IMPROVE_LOG_PATH = Path(DATA_DIR) / "self_improve_log.json"
EDITABLE_FILE_SUFFIXES = {".py", ".txt", ".md", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg"}
EDITABLE_SPECIAL_FILENAMES = {"Dockerfile", "requirements.txt"}
IGNORED_PATH_PARTS = {"data", "backups", "__pycache__", ".git"}


def _workspace_root() -> Path:
    return Path(SOURCE_DIR)


def normalize_relative_path(relative_path: str) -> str:
    return relative_path.replace("\\", "/").strip().lstrip("/")


def workspace_file_exists(relative_path: str) -> bool:
    cleaned = normalize_relative_path(relative_path)
    if not cleaned:
        return False
    resolved = (_workspace_root() / cleaned).resolve()
    try:
        resolved.relative_to(_workspace_root())
    except ValueError:
        return False
    return resolved.exists()


def get_path_state(relative_path: str) -> dict:
    cleaned = normalize_relative_path(relative_path)
    if not cleaned:
        return {
            "path": cleaned,
            "allowed": False,
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "resolved": None,
        }

    resolved = (_workspace_root() / cleaned).resolve()
    try:
        resolved.relative_to(_workspace_root())
        in_workspace = True
    except ValueError:
        in_workspace = False

    allowed = in_workspace and _is_allowed_workspace_path(cleaned)
    exists = in_workspace and resolved.exists()
    return {
        "path": cleaned,
        "allowed": allowed,
        "exists": exists,
        "is_file": exists and resolved.is_file(),
        "is_dir": exists and resolved.is_dir(),
        "resolved": str(resolved) if in_workspace else None,
    }


def _is_allowed_workspace_path(relative_path: str) -> bool:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return False
    if any(part in IGNORED_PATH_PARTS for part in candidate.parts):
        return False
    if candidate.name in EDITABLE_SPECIAL_FILENAMES:
        return True
    return candidate.suffix in EDITABLE_FILE_SUFFIXES


def _workspace_path(relative_path: str) -> Path:
    cleaned = normalize_relative_path(relative_path)
    if not cleaned:
        raise ValueError("Path is required.")
    if not _is_allowed_workspace_path(cleaned):
        raise ValueError(f"Path '{cleaned}' is not editable.")

    resolved = (_workspace_root() / cleaned).resolve()
    try:
        resolved.relative_to(_workspace_root())
    except ValueError as exc:
        raise ValueError(f"Path '{cleaned}' escapes the workspace.") from exc
    return resolved


def list_workspace_files() -> list[str]:
    files = []
    for path in sorted(_workspace_root().rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(_workspace_root()).as_posix()
        if not _is_allowed_workspace_path(relative):
            continue
        files.append(relative)
    return files


def list_actual_files(max_files: int = 500) -> list[str]:
    root = _workspace_root()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files.append(relative)
        if len(files) >= max_files:
            break
    return files


def read_workspace_file(relative_path: str) -> str:
    return _workspace_path(relative_path).read_text(encoding="utf-8")


def backup_workspace_file(relative_path: str) -> str:
    source_path = _workspace_path(relative_path)
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = relative_path.replace("/", "__")
    backup_path = BACKUPS_DIR / f"{timestamp}_{backup_name}"
    backup_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(backup_path)


def capture_workspace_snapshot(max_chars: int = 50000) -> str:
    parts = []
    total = 0
    for relative in list_workspace_files():
        text = read_workspace_file(relative)
        block = f"FILE: {relative}\n```\n{text}\n```\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model did not return a JSON object.")

    return json.loads(cleaned[start:end + 1])


def _load_self_improve_log() -> list:
    if not SELF_IMPROVE_LOG_PATH.exists():
        return []
    try:
        return json.loads(SELF_IMPROVE_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def record_self_improvement(
    summary: str,
    changed_files: list[str],
    backup_paths: list[str],
    post_apply: str,
    verification: list[dict] | None = None,
) -> None:
    history = _load_self_improve_log()
    history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "changed_files": changed_files,
            "backup_paths": backup_paths,
            "post_apply": post_apply,
            "verification": verification or [],
        }
    )
    history = history[-50:]
    SELF_IMPROVE_LOG_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def load_last_self_improvement() -> dict | None:
    history = _load_self_improve_log()
    if not history:
        return None
    return history[-1]


def validate_python_files() -> None:
    for relative in list_workspace_files():
        if relative.endswith(".py"):
            py_compile.compile(str(_workspace_path(relative)), doraise=True)


def rollback_self_improvement(applied_changes: list[dict]) -> None:
    for change in reversed(applied_changes):
        target = _workspace_path(change["path"])
        backup_path = change.get("backup_path")
        if change["action"] == "create":
            if target.exists():
                target.unlink()
            continue
        if backup_path:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(Path(backup_path).read_text(encoding="utf-8"), encoding="utf-8")


def _verify_operation(action: str, before_state: dict, after_state: dict) -> bool:
    if action in {"create", "update"}:
        return after_state["exists"] and after_state["is_file"] and after_state["allowed"]
    if action == "delete":
        return not after_state["exists"] and after_state["allowed"]
    return False


def apply_self_improvement(payload: dict) -> tuple[list[str], list[str], str, list[dict]]:
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("Improvement payload must contain a non-empty 'operations' list.")

    applied_changes = []
    backup_paths = []
    changed_files = []
    verification = []
    post_apply = payload.get("post_apply", "restart")

    try:
        for entry in operations:
            if not isinstance(entry, dict):
                raise ValueError("Each operation must be an object.")

            action = entry.get("action")
            relative_path = entry.get("path")
            content = entry.get("content")

            if action not in {"create", "update", "delete"}:
                raise ValueError(f"Unsupported action: {action}")
            if not isinstance(relative_path, str):
                raise ValueError("Operation path must be a string.")

            before_state = get_path_state(relative_path)
            if not before_state["allowed"]:
                raise ValueError(f"Path '{relative_path}' is not allowed for self-improvement.")

            target = _workspace_path(relative_path)
            exists = target.exists()

            if action == "create":
                if exists:
                    raise ValueError(f"Cannot create existing file: {relative_path}")
                if not isinstance(content, str):
                    raise ValueError(f"Create operation requires content for {relative_path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                after_state = get_path_state(relative_path)
                if not _verify_operation(action, before_state, after_state):
                    raise ValueError(f"Path verification failed for create: {relative_path}")
                applied_changes.append({"action": action, "path": relative_path})
                changed_files.append(relative_path)
                verification.append({"action": action, "before": before_state, "after": after_state})
                continue

            if not exists:
                raise ValueError(f"File does not exist: {relative_path}")

            backup_path = backup_workspace_file(relative_path)
            backup_paths.append(backup_path)

            if action == "update":
                if not isinstance(content, str):
                    raise ValueError(f"Update operation requires content for {relative_path}")
                current = target.read_text(encoding="utf-8")
                if current == content:
                    continue
                target.write_text(content, encoding="utf-8")
            else:
                target.unlink()

            after_state = get_path_state(relative_path)
            if not _verify_operation(action, before_state, after_state):
                raise ValueError(f"Path verification failed for {action}: {relative_path}")

            applied_changes.append({"action": action, "path": relative_path, "backup_path": backup_path})
            changed_files.append(relative_path)
            verification.append({"action": action, "before": before_state, "after": after_state})

        validate_python_files()
    except Exception:
        rollback_self_improvement(applied_changes)
        raise

    return list(dict.fromkeys(changed_files)), backup_paths, post_apply, verification


async def propose_self_improvement(genai_client, ai_model: str, goal: str) -> dict:
    source_snapshot = capture_workspace_snapshot()
    prompt = (
        "You are improving a Discord bot codebase. "
        "Return JSON only with this exact shape: "
        '{"summary": "short summary", "post_apply": "restart|rebuild|none", "operations": [{"action": "create|update|delete", "path": "relative/path.py", "content": "full file content when needed"}]}. '
        "You may create, update, or delete text files inside the project workspace, but never under data/ or backups/. "
        "Use relative paths only. "
        "When adding or removing modules, also update skills_manifest.json so the bot knows its current file map and skills. "
        "Keep the bot grounded: runtime claims must be based on physical files or direct command output only. "
        "When creating a new module, also update imports in existing files if needed. "
        "Keep working features intact, make minimal changes, and ensure all Python files remain valid. "
        "Set post_apply to rebuild if requirements.txt or Dockerfile changes; otherwise use restart for Python/module changes. "
        f"Goal: {goal}\n\nCurrent project files:\n{source_snapshot}"
    )
    response = await __import__("asyncio").to_thread(
        genai_client.models.generate_content,
        model=ai_model,
        contents=prompt,
    )
    return extract_json_object(response.text or "")
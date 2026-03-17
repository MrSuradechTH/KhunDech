# KhunDech Discord bot entrypoint.
#
# This file is part of KhunDech.
# KhunDech is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KhunDech is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KhunDech.  If not, see <https://www.gnu.org/licenses/>.

import asyncio
import importlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord
from discord.ext import commands, tasks
from google import genai

from khundech.config import AI_MODEL, ALLOWED_USER_ID, CRON_CHANNEL_ID, DATA_DIR, DISCORD_TOKEN, GEMINI_API_KEY, LEARNING_CHANNEL_ID, MAIN_CHANNEL_ID, SOURCE_DIR
from khundech.auto_learning import (
    add_auto_learning_job,
    delete_auto_learning_job,
    ensure_default_auto_learning_job,
    is_auto_learning_list_intent,
    list_auto_learning_jobs,
    load_auto_learning_jobs,
    parse_auto_learning_add_intent,
    parse_auto_learning_delete_intent,
    parse_auto_learning_update_intent,
    save_auto_learning_jobs,
    update_auto_learning_job,
)
from khundech.docker_tools import format_host_path_report, restart_self, run_docker_command, run_sync_probe
from khundech.financial_knowledge import build_knowledge_research_context, run_financial_knowledge_learning_cycle
from khundech.manifest import build_skills_context, ensure_skills_manifest, load_skills_manifest
from khundech.persistence import load_history, load_memory, save_history, save_memory
from khundech.runtime_info import format_runtime_facts, get_runtime_facts
from khundech.task_scheduler import add_scheduled_task, load_scheduled_tasks, parse_task_add_intent, save_scheduled_tasks
from khundech.self_improve import (
    apply_self_improvement,
    get_path_state,
    list_actual_files,
    load_last_self_improvement,
    propose_self_improvement,
    record_self_improvement,
    workspace_file_exists,
)
from khundech.set_finance import fetch_set_financial_data, format_set_financial_summary, normalize_stock_symbol

try:
    from finance_module import get_stock_data as _finance_get_stock_data
except Exception:
    _finance_get_stock_data = None


genai_client = genai.Client(api_key=GEMINI_API_KEY)
ensure_skills_manifest()

BASE_SYSTEM_INSTRUCTION = (
    "You are KhunDech, an intelligent and self-improving AI assistant embedded in Discord.\n"
    "You serve only your master.\n"
    "Be helpful, direct, and concise.\n"
    "You must only claim facts that are grounded in provided data, physical files, command output, or explicit user messages.\n"
    "Never claim that you created/updated/deleted files unless those file paths are physically verified to exist (or not exist for delete) on disk.\n"
    "Use maximum reasoning depth and check your answer for consistency before replying.\n"
    "If you are unsure, say you do not know instead of inventing details.\n"
    "When the user needs SET stock financial data, tell them to use !setfinancial <SYMBOL> or !setbalance <SYMBOL>.\n"
    "When the user wants you to change your own codebase, tell them to use !improve <goal> or !upgrade <goal>.\n"
    "When the user asks about runtime or Docker state, use !runtime, !docker, or !compose."
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


WEB_SCRAPING_MODULE_RELATIVE_PATH = "khundech/web_scraping.py"
TASK_RUNNER_INTERVAL_SECONDS = 10
TASK_DUE_GRACE_SECONDS = 2
KNOWLEDGE_LEARNING_INTERVAL_MINUTES = 10
AUTO_LEARNING_RUNNER_INTERVAL_SECONDS = 60
UTC_PLUS_7 = timezone(timedelta(hours=7))
WEB_SCRAPING_MODULE_TEMPLATE = r'''import ast
import html
import re
import urllib.request


SIAMCHART_STOCK_URL = "http://siamchart.com/stock/"
SIAMCHART_JDATA_URL = "http://siamchart.com/jdata.jsp"
SIAMCHART_TABLE_HEADERS = [
    "Name", "No.", "Links", "Sign", "Last", "Chg%", "Volume", "Value (k)", "MCap (M)",
    "P/E", "P/BV", "D/E", "DPS", "EPS", "ROA%", "ROE%", "NPM%", "Yield%", "FFloat%", "MG%",
    "Magic1", "Magic2", "PEG", "CG",
]


def _base_n_word(value: int, base: int) -> str:
    if value < base:
        if value < 10:
            return str(value)
        if value < 36:
            return chr(87 + value)
        return chr(29 + value)
    return _base_n_word(value // base, base) + _base_n_word(value % base, base)


def _js_unescape(text: str) -> str:
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return text


def _unpack_packer_payload(payload: str) -> str:
    match = re.search(
        r"eval\(function\(p,a,c,k,e,d\)\{.*?\}\('(?P<p>.*)',(?P<a>\d+),(?P<c>\d+),'(?P<k>.*)'\.split\('\|'\),0,\{\}\)\)",
        payload,
        re.S,
    )
    if not match:
        return payload

    packed = _js_unescape(match.group("p"))
    base = int(match.group("a"))
    count = int(match.group("c"))
    key_raw = _js_unescape(match.group("k"))
    keys = key_raw.split("|")

    unpacked = packed
    for idx in range(count - 1, -1, -1):
        if idx >= len(keys):
            continue
        replacement = keys[idx]
        if replacement == "":
            continue
        token = _base_n_word(idx, base)
        unpacked = re.sub(rf"\b{re.escape(token)}\b", replacement, unpacked)
    return unpacked


def _build_request(url: str, referer: str | None = None) -> urllib.request.Request:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    if referer:
        headers["Referer"] = referer
    return urllib.request.Request(url, headers=headers)


def _is_siamchart_stock_url(url: str) -> bool:
    normalized = (url or "").strip().lower()
    return normalized.startswith("http://siamchart.com/stock/") or normalized.startswith("https://siamchart.com/stock/")


def _decode_response(raw_bytes: bytes, content_type: str = "") -> tuple[str, str]:
    candidates = []

    header_match = re.search(r"charset=([A-Za-z0-9_-]+)", content_type or "", re.I)
    if header_match:
        candidates.append(header_match.group(1))

    head_ascii = raw_bytes[:4096].decode("ascii", errors="ignore")
    meta_match = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9_-]+)", head_ascii, re.I)
    if meta_match:
        candidates.append(meta_match.group(1))

    candidates.extend(["tis-620", "cp874", "utf-8"])

    seen = set()
    for charset in candidates:
        normalized = (charset or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            return raw_bytes.decode(normalized), normalized
        except Exception:
            continue

    return raw_bytes.decode("utf-8", errors="ignore"), "utf-8"


def get_siamchart_data(timeout: int = 15) -> str:
    request = _build_request(SIAMCHART_JDATA_URL, referer=SIAMCHART_STOCK_URL)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_bytes = response.read()
        content_type = response.headers.get("Content-Type", "")
    text, _charset = _decode_response(raw_bytes, content_type)
    return text.strip()


def _extract_balanced_array_literal(source: str, start_index: int) -> str | None:
    depth = 0
    in_string = False
    string_char = ""
    escaped = False

    for index in range(start_index, len(source)):
        ch = source[index]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_char:
                in_string = False
            continue

        if ch in {"'", '"'}:
            in_string = True
            string_char = ch
            continue

        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return source[start_index:index + 1]
            continue

    return None


def _find_js_array_assignments(source: str) -> list[str]:
    arrays = []
    for match in re.finditer(r"(?:var|1d)\s+[A-Za-z0-9_]+\s*=\s*\[\[", source):
        start = source.find("[[", match.start())
        if start < 0:
            continue
        raw_array = _extract_balanced_array_literal(source, start)
        if raw_array:
            arrays.append(raw_array)
    return arrays


def _parse_js_array_rows(raw_array: str) -> list[list[str]]:
    parsed = None
    attempts = [raw_array]
    attempts.append(raw_array.replace("\\'", "'"))
    try:
        attempts.append(bytes(raw_array, "utf-8").decode("unicode_escape").replace("\\'", "'"))
    except Exception:
        pass

    for candidate in attempts:
        try:
            parsed = ast.literal_eval(candidate)
            break
        except Exception:
            continue
    if parsed is None or not isinstance(parsed, list):
        return []

    rows = []
    for item in parsed:
        if isinstance(item, list):
            rows.append([str(cell) if cell is not None else "" for cell in item])
    return rows


def _extract_siamchart_rows(payload: str) -> list[list[str]]:
    unpacked = _unpack_packer_payload(payload)
    candidate_sources = [unpacked, payload]
    best_rows: list[list[str]] = []
    best_score = (-1, -1)

    for source in candidate_sources:
        for raw_array in _find_js_array_assignments(source):
            rows = _parse_js_array_rows(raw_array)
            if not rows:
                continue

            max_width = max((len(row) for row in rows), default=0)
            score = (len(rows), max_width)
            if max_width >= 20 and len(rows) >= 100:
                return rows
            if score > best_score:
                best_rows = rows
                best_score = score

    return best_rows


def get_siamchart_stock_rows(limit: int | None = 200) -> list[dict[str, str]]:
    payload = get_siamchart_data()
    rows = _extract_siamchart_rows(payload)
    if limit is not None:
        rows = rows[: max(0, limit)]

    mapped = []
    for index, row in enumerate(rows, start=1):
        if not row:
            continue

        symbol = html.unescape(row[0]).strip()
        links = html.unescape(row[3]).replace("|", " ").strip() if len(row) > 3 else ""
        sign = html.unescape(row[4]).strip() if len(row) > 4 else ""
        last = html.unescape(row[5]).strip() if len(row) > 5 else ""
        chg = html.unescape(row[6]).strip() if len(row) > 6 else ""
        volume = html.unescape(row[7]).strip() if len(row) > 7 else ""
        value_k = html.unescape(row[8]).strip() if len(row) > 8 else ""
        mcap_m = html.unescape(row[9]).strip() if len(row) > 9 else ""
        pe = html.unescape(row[10]).strip() if len(row) > 10 else ""
        pbv = html.unescape(row[11]).strip() if len(row) > 11 else ""
        de = html.unescape(row[12]).strip() if len(row) > 12 else ""
        dps = html.unescape(row[13]).strip() if len(row) > 13 else ""
        eps = html.unescape(row[14]).strip() if len(row) > 14 else ""
        roa = html.unescape(row[15]).strip() if len(row) > 15 else ""
        roe = html.unescape(row[16]).strip() if len(row) > 16 else ""
        npm = html.unescape(row[17]).strip() if len(row) > 17 else ""
        yield_pct = html.unescape(row[18]).strip() if len(row) > 18 else ""
        ffloat = html.unescape(row[19]).strip() if len(row) > 19 else ""
        mg = html.unescape(row[20]).strip() if len(row) > 20 else ""
        magic1 = html.unescape(row[21]).strip() if len(row) > 21 else ""
        magic2 = html.unescape(row[22]).strip() if len(row) > 22 else ""
        peg = html.unescape(row[23]).strip() if len(row) > 23 else ""
        cg = html.unescape(row[24]).strip() if len(row) > 24 else ""

        mapped.append(
            {
                "Name": symbol,
                "No.": str(index),
                "Links": links,
                "Sign": sign,
                "Last": last,
                "Chg%": chg,
                "Volume": volume,
                "Value (k)": value_k,
                "MCap (M)": mcap_m,
                "P/E": pe,
                "P/BV": pbv,
                "D/E": de,
                "DPS": dps,
                "EPS": eps,
                "ROA%": roa,
                "ROE%": roe,
                "NPM%": npm,
                "Yield%": yield_pct,
                "FFloat%": ffloat,
                "MG%": mg,
                "Magic1": magic1,
                "Magic2": magic2,
                "PEG": peg,
                "CG": cg,
            }
        )
    return mapped


def get_siamchart_stock_table_text(limit: int = 120) -> str:
    rows = get_siamchart_stock_rows(limit=limit)
    if not rows:
        return "No Siamchart stock rows parsed from jdata.jsp"

    lines = ["\\t".join(SIAMCHART_TABLE_HEADERS)]
    for row in rows:
        lines.append("\\t".join(row.get(header, "") for header in SIAMCHART_TABLE_HEADERS))
    return "\\n".join(lines)


def fetch_web_raw_text(url: str, timeout: int = 20) -> str:
    if _is_siamchart_stock_url(url):
        return get_siamchart_data(timeout=timeout)

    referer = SIAMCHART_STOCK_URL if "siamchart.com/jdata.jsp" in (url or "").lower() else None
    request = _build_request(url, referer=referer)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_bytes = response.read()
        content_type = response.headers.get("Content-Type", "")
    text, _charset = _decode_response(raw_bytes, content_type)
    return text


def fetch_web_text(url: str, timeout: int = 20) -> str:
    if _is_siamchart_stock_url(url):
        return get_siamchart_stock_table_text(limit=120)

    content = fetch_web_raw_text(url, timeout=timeout)
    cleaned = re.sub(r"\\s+", " ", content).strip()
    return cleaned
'''


def _load_manifest_for_update() -> tuple[Path, dict]:
    manifest_path = Path(SOURCE_DIR) / "skills_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = load_skills_manifest()
    else:
        manifest = load_skills_manifest()
    if not isinstance(manifest.get("files"), list):
        manifest["files"] = []
    if not isinstance(manifest.get("skills"), list):
        manifest["skills"] = []
    return manifest_path, manifest


def _upsert_manifest_file_entry(manifest: dict, path: str, role: str) -> None:
    files = manifest.get("files", [])
    for item in files:
        if isinstance(item, dict) and item.get("path") == path:
            item["role"] = role
            return
    files.append({"path": path, "role": role})


def _upsert_manifest_skill_entry(manifest: dict, name: str, description: str) -> None:
    skills = manifest.get("skills", [])
    for item in skills:
        if isinstance(item, dict) and str(item.get("name", "")).lower() == name.lower():
            item["name"] = name
            item["description"] = description
            return
    skills.append({"name": name, "description": description})


def _extract_skill_add_intent(text: str) -> dict | None:
    lower = (text or "").lower()
    if not lower:
        return None

    add_markers = ["add skill", "create skill", "เพิ่ม skill", "เพิ่มสกิล", "skill name"]
    if not any(marker in lower for marker in add_markers):
        return None

    skill_name = None
    name_match = re.search(r"skill\s*name\s*[(:]?\s*[\"']?([a-zA-Z0-9_-]+)", text, re.I)
    if name_match:
        skill_name = name_match.group(1).strip().lower()

    if not skill_name and ("web" in lower and ("scrap" in lower or "scrape" in lower)):
        skill_name = "web_scraping"

    if not skill_name:
        return None

    normalized_name = skill_name.replace("-", "_")
    if normalized_name in {"webscraping", "web_scrping", "webscrping", "web_scrape"}:
        normalized_name = "web_scraping"

    required_parameter = "url" if "url" in lower else None
    return {"skill_name": normalized_name, "required_parameter": required_parameter}


def _run_add_skill_intent(intent: dict) -> str:
    skill_name = intent.get("skill_name")
    required_parameter = intent.get("required_parameter") or "url"
    if skill_name != "web_scraping":
        return "I currently support deterministic add-skill flow for web_scraping only."

    module_relative = WEB_SCRAPING_MODULE_RELATIVE_PATH
    module_path = Path(SOURCE_DIR) / module_relative
    created_module = False
    if not module_path.exists():
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(WEB_SCRAPING_MODULE_TEMPLATE, encoding="utf-8")
        created_module = True

    manifest_path, manifest = _load_manifest_for_update()
    _upsert_manifest_file_entry(
        manifest,
        module_relative,
        "Website scraping helper module for automation tasks requiring URL input.",
    )
    _upsert_manifest_skill_entry(
        manifest,
        "web_scraping",
        f"Scrape website content for automation using required parameter `{required_parameter}`.",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"✅ Skill `web_scraping` registered."]
    lines.append(f"- Module: /app/{module_relative} ({'created' if created_module else 'already exists'})")
    lines.append("- Manifest: /app/skills_manifest.json (updated)")
    lines.append("- Required parameter: url")
    lines.append(f"- Verify module exists: {workspace_file_exists(module_relative)}")
    lines.append(f"- Verify manifest exists: {workspace_file_exists('skills_manifest.json')}")
    return "\n".join(lines)


def _extract_skill_status_intent(text: str) -> bool:
    lower = (text or "").lower().strip()
    if not lower:
        return False

    direct_markers = [
        "how many skill",
        "how many skills",
        "list skill",
        "list skills",
        "show skill",
        "show skills",
        "what skill",
        "what skills",
        "skill you have",
        "skills you have",
        "มี skill",
        "มีสกิล",
        "สกิลอะไรบ้าง",
        "มีอะไรบ้าง",
    ]
    if any(marker in lower for marker in direct_markers):
        return True

    if "skill" in lower and any(token in lower for token in ["how many", "count", "list", "show", "what", "now"]):
        return True

    return False


def _run_skill_status_intent() -> str:
    manifest = load_skills_manifest()
    skills = manifest.get("skills", []) if isinstance(manifest, dict) else []
    files = manifest.get("files", []) if isinstance(manifest, dict) else []

    # Normalise: accept both {"name":..., "description":...} dicts and plain strings
    valid_skills = []
    for s in skills:
        if isinstance(s, dict) and s.get("name"):
            valid_skills.append(s)
        elif isinstance(s, str) and s.strip():
            valid_skills.append({"name": s.strip(), "description": ""})

    lines = [f"🧭 KhunDech has {len(valid_skills)} skill(s) right now."]

    if valid_skills:
        lines.append("Skills:")
        for skill in valid_skills:
            name = str(skill.get("name", "")).strip()
            desc = str(skill.get("description", "")).strip()
            if desc:
                lines.append(f"- {name}: {desc}")
            else:
                lines.append(f"- {name}")
    else:
        lines.append("No skills listed in manifest yet.")

    lines.append("")
    lines.append(f"Known file/path entries: {len([f for f in files if isinstance(f, dict)])}")
    lines.append("Tip: use !skills for full report.")
    return "\n".join(lines)


def _extract_skill_usage_intent(text: str) -> bool:
    lower = (text or "").lower().strip()
    if not lower:
        return False
    usage_markers = [
        "how can i use the skill",
        "how can i use skill",
        "how to use the skill",
        "how to use skill",
        "how can i use the stock skill",
        "how to use stock analysis",
        "use the skill",
        "วิธีใช้สกิล",
        "ใช้สกิลยังไง",
    ]
    return any(marker in lower for marker in usage_markers)


def _extract_pretty_table_output_intent(text: str) -> bool:
    lower = (text or "").lower().strip()
    if not lower:
        return False
    pretty_markers = [
        "pretty table",
        "pretty output",
        "improve pretty table",
        "table output",
        "discord table",
        "format table",
        "ตารางสวย",
        "จัดตาราง",
    ]
    stock_markers = ["stock", "stock seeking", "task", "table", "หุ้น", "ตาราง"]
    return any(marker in lower for marker in pretty_markers) and any(marker in lower for marker in stock_markers)


def _run_list_auto_learning_intent() -> str:
    # Render a Discord-friendly summary of all auto-learning jobs.
    jobs = list_auto_learning_jobs()
    if not jobs:
        return "🧠 No auto-learning jobs found."
    lines = [f"🧠 **Auto-learning jobs: {len(jobs)}**"]
    for index, job in enumerate(jobs, start=1):
        name = str(job.get("name") or f"job_{index}")
        status = "enabled" if job.get("active", True) else "disabled"
        interval = f"every {job.get('interval')} {job.get('unit')}"
        purpose = str(job.get("purpose") or "-")
        first_prompt = str(job.get("seed_question") or "-")

        last_run_text = _format_time_utc7(job.get("last_run"))

        lines.append(f"\n**{index}) {name}**")
        lines.append(f"• Interval: {interval}")
        lines.append(f"• Status: {status}")
        lines.append(f"• Last run: {last_run_text}")
        lines.append(f"• Purpose: {purpose}")
        lines.append(f"• First prompt: {first_prompt}")
    return "\n".join(lines)


def _run_add_auto_learning_intent(intent: dict) -> str:
    try:
        job = add_auto_learning_job(
            intent["name"],
            intent["purpose"],
            intent["seed_question"],
            intent["interval"],
            intent["unit"],
            intent.get("active", True),
        )
    except ValueError as exc:
        return f"⚠️ {exc}"
    return (
        f"✅ Auto-learning job added: {job.get('name')}\n"
        f"- Purpose: {job.get('purpose')}\n"
        f"- Init prompt: {job.get('seed_question')}\n"
        f"- Interval: every {job.get('interval')} {job.get('unit')}\n"
        f"- Active: {job.get('active', True)}"
    )


def _run_update_auto_learning_intent(intent: dict) -> str:
    job = update_auto_learning_job(
        intent["name"],
        interval=intent.get("interval"),
        unit=intent.get("unit"),
        seed_question=intent.get("seed_question"),
        purpose=intent.get("purpose"),
        active=intent.get("active"),
    )
    if not job:
        return f"Auto-learning job not found: {intent['name']}"
    return (
        f"✅ Auto-learning job updated: {job.get('name')}\n"
        f"- Interval: every {job.get('interval')} {job.get('unit')}\n"
        f"- Purpose: {job.get('purpose')}\n"
        f"- Init prompt: {job.get('seed_question')}\n"
        f"- Active: {job.get('active', True)}"
    )


def _run_delete_auto_learning_intent(name: str) -> str:
    if not delete_auto_learning_job(name):
        return f"Auto-learning job not found: {name}"
    return f"✅ Auto-learning job deleted: {name}"


def _run_pretty_table_output_intent() -> str:
    return (
        "✅ Stock Seeking output now uses Discord-friendly pipe table formatting (with 1Y H/L, Debt/Equit, Rate Score).\n"
        "Run this to check latest output: run task stock_seeking now"
    )


def _run_skill_usage_intent() -> str:
    lines = ["🧭 Skill usage guide"]
    lines.append("- SET financial summary + quick analysis: !setfinancial <SYMBOL>")
    lines.append("- Price snapshot (SET source): !price <SYMBOL>")
    lines.append("- Price compare (SET vs Siamchart): !compareprice <SYMBOL>")
    lines.append("- Run stock screening task now: run task stock_seeking now")
    if _finance_get_stock_data is not None:
        lines.append("- Optional yfinance module: !stock <TICKER>")
    lines.append("Example: !setfinancial AOT")
    return "\n".join(lines)


def _extract_move_web_scraping_intent(text: str) -> bool:
    lower = (text or "").lower()
    move_markers = ["move", "ย้าย"]
    has_webscraping = ("web_scraping" in lower) or ("webscraping" in lower) or ("web scraping" in lower)
    has_khundech = ("/app/khundech" in lower) or ("khundech folder" in lower) or ("khundech/" in lower)
    return any(marker in lower for marker in move_markers) and has_webscraping and has_khundech


def _run_move_web_scraping_to_khundech() -> str:
    target_relative = WEB_SCRAPING_MODULE_RELATIVE_PATH
    target_path = Path(SOURCE_DIR) / target_relative
    candidate_sources = [
        Path(SOURCE_DIR) / "web_scraping.py",
        Path(SOURCE_DIR) / "webscraping.py",
        Path(SOURCE_DIR) / "khundech/webscraping.py",
        target_path,
    ]

    source_path = None
    for candidate in candidate_sources:
        if candidate.exists() and candidate.is_file():
            source_path = candidate
            break

    if source_path is None:
        return (
            "⚠️ web_scraping module was not found on disk.\n"
            f"- Expected target: /app/{target_relative}\n"
            "Use 'add skill name web_scraping with required url' first."
        )

    moved = False
    if source_path.resolve() != target_path.resolve():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)
        moved = True

    manifest_path, manifest = _load_manifest_for_update()
    _upsert_manifest_file_entry(
        manifest,
        target_relative,
        "Website scraping helper module for automation tasks requiring URL input.",
    )
    _upsert_manifest_skill_entry(
        manifest,
        "web_scraping",
        "Scrape website content for automation using required parameter `url`.",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return (
        "✅ web_scraping module is now in /app/khundech/.\n"
        f"- Target exists: {target_path.exists()}\n"
        f"- Action: {'moved file' if moved else 'already in target path'}\n"
        "- Manifest updated: /app/skills_manifest.json"
    )


def _extract_web_scraping_where_intent(text: str) -> bool:
    lower = (text or "").lower()
    has_webscraping = ("web_scraping" in lower) or ("webscraping" in lower) or ("web scraping" in lower)
    where_markers = ["where", "folder", "path", "อยู่ไหน", "ที่ไหน"]
    return has_webscraping and any(marker in lower for marker in where_markers)


def _extract_web_scraping_use_intent(text: str) -> dict | None:
    lower = (text or "").lower()
    if not lower:
        return None

    url_match = re.search(r"https?://[^\s\"']+", text, re.I)

    # Handle: "!scrape <url>" or "do !scrape <url>" or "scrape <url>"
    if url_match and re.search(r"!?scrape\b", lower):
        wants_raw = "raw" in lower or "ดิบ" in lower
        return {"url": url_match.group(0).strip(), "raw": wants_raw}

    scraping_markers = [
        "web scraping",
        "web_scraping",
        "webscraping",
        "websraping",
        "web sraping",
        "web scrping",
        "webscrping",
        "scraping skill",
        "web scraping skill",
        "webscraping skill",
        "scrap data",
    ]
    use_markers = ["use", "using", "try to use", "get data", "raw data", "ดึงข้อมูล", "with this", "try with this"]
    has_scraping_marker = any(marker in lower for marker in scraping_markers)
    has_use_marker = any(marker in lower for marker in use_markers)

    if not has_scraping_marker:
        if not (url_match and any(marker in lower for marker in ["try with this", "use with this", "with this page", "with this"])):
            return None

    if not has_use_marker and not (url_match and "with this" in lower):
        return None

    if not url_match:
        return None

    wants_raw = "raw" in lower or "ดิบ" in lower
    return {"url": url_match.group(0).strip(), "raw": wants_raw}


def _resolve_followup_web_scraping_intent(text: str, history: list) -> dict | None:
    current = (text or "").strip()
    if not current:
        return None

    lower = current.lower()
    wants_raw = ("raw" in lower or "ดิบ" in lower)
    raw_followup_markers = ["show raw", "show me raw", "raw data", "raw please", "ดู raw", "ขอดิบ"]

    url_match = re.search(r"https?://[^\s\"']+", current, re.I)
    if not url_match:
        if wants_raw or any(marker in lower for marker in raw_followup_markers):
            for turn in reversed(history[-10:]):
                for candidate_text in [turn.get("user") or "", turn.get("assistant") or ""]:
                    candidate_match = re.search(r"https?://[^\s\"']+", candidate_text, re.I)
                    if candidate_match:
                        return {"url": candidate_match.group(0).strip(), "raw": True}
        return None

    explicit_page_markers = ["this page", "that page", "หน้าเว็บนี้", "เว็บนี้", "use with"]
    has_page_reference = any(marker in lower for marker in explicit_page_markers)

    if not has_page_reference:
        return None

    for turn in reversed(history[-10:]):
        user_text = (turn.get("user") or "").lower()
        assistant_text = (turn.get("assistant") or "").lower()
        if any(token in user_text for token in ["web scraping", "web sraping", "web_scraping", "scraping skill", "scrap data"]):
            wants_raw = (wants_raw or "raw" in user_text)
            return {"url": url_match.group(0).strip(), "raw": wants_raw}
        if "web scraping completed" in assistant_text:
            wants_raw = wants_raw or "raw data:" in assistant_text
            return {"url": url_match.group(0).strip(), "raw": wants_raw}

    return None


def _run_web_scraping_where() -> str:
    relative = WEB_SCRAPING_MODULE_RELATIVE_PATH
    state = get_path_state(relative)
    if state["exists"]:
        return f"✅ web_scraping path: /app/{relative}\n{format_path_state(relative)}"
    return (
        f"⚠️ web_scraping module not found at /app/{relative}.\n"
        f"{format_path_state(relative)}"
    )


def _run_web_scraping_fetch_intent(intent: dict) -> str:
    url = intent.get("url", "").strip()
    wants_raw = bool(intent.get("raw"))
    if not url:
        return "Please provide a URL, for example: use web scraping skill on https://example.com and give raw data"

    module_relative = WEB_SCRAPING_MODULE_RELATIVE_PATH
    if not workspace_file_exists(module_relative):
        return (
            f"⚠️ web_scraping module not found at /app/{module_relative}.\n"
            "Use: add skill name web_scraping with required parameter url"
        )

    try:
        fetch_fn = _load_web_scraping_fetcher(wants_raw=wants_raw)
        content = fetch_fn(url)
    except Exception as exc:
        return f"⚠️ Web scraping failed: {exc}"

    content = (content or "").strip()
    if not content:
        return "Web scraping completed but no text content was returned."

    clipped = content[:3500]
    clipped_note = "\n(Trimmed to first 3500 chars)" if len(content) > 3500 else ""

    if wants_raw:
        return (
            f"✅ Web scraping completed for {url}\n"
            f"- Source module: /app/{module_relative}\n"
            f"- Characters: {len(content)}{clipped_note}\n"
            f"Raw data:\n{clipped}"
        )

    return (
        f"✅ Web scraping completed for {url}\n"
        f"- Source module: /app/{module_relative}\n"
        f"- Characters: {len(content)}{clipped_note}\n"
        f"Preview:\n{clipped}"
    )


def _load_web_scraping_fetcher(wants_raw: bool = False):
    module_relative = WEB_SCRAPING_MODULE_RELATIVE_PATH
    if not workspace_file_exists(module_relative):
        raise RuntimeError(
            f"web_scraping module not found at /app/{module_relative}. "
            "Use: add skill name web_scraping with required parameter url"
        )

    web_module = importlib.import_module("khundech.web_scraping")
    fetch_name = "fetch_web_raw_text" if wants_raw else "fetch_web_text"
    fetch_fn = getattr(web_module, fetch_name, None)
    if wants_raw and not callable(fetch_fn):
        fetch_fn = getattr(web_module, "fetch_web_text", None)
    if not callable(fetch_fn):
        raise RuntimeError(f"web_scraping module exists but {fetch_name}(url) is missing.")
    return fetch_fn


def _looks_like_operational_request(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    if re.search(r"https?://[^\s\"']+", lower) and any(marker in lower for marker in ["use", "page", "เว็บ", "scrap", "scrape"]):
        return True
    operational_markers = [
        "add task",
        "task name",
        "schedule",
        "create",
        "update",
        "delete",
        "remove",
        "move",
        "skill",
        "upgrade",
        "improve",
        "checkpath",
        "actual path",
        "full path",
        "host path",
        "เพิ่ม",
        "ลบ",
        "ย้าย",
        "อัพเกรด",
        "อัปเกรด",
    ]
    return any(marker in lower for marker in operational_markers)


def _extract_stock_symbol(text: str) -> str | None:
    if not text:
        return None

    explicit_patterns = [
        r"(?:symbol|ticker|หุ้น|ราคา|price\s+of|set\s+financial)\s*[:=]?\s*([A-Za-z0-9._-]{2,10})",
        r"\b([A-Za-z0-9._-]{2,10})\s+(?:price|หุ้น|stock)\b",
        r"(?:stock|หุ้น)\s+([A-Za-z0-9._-]{2,10})",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            candidate = re.sub(r"[^A-Za-z0-9._-]", "", match.group(1)).upper()
            if candidate:
                return candidate

    stopwords = {
        "GIVE", "ME", "PRICE", "STOCK", "SET", "FINANCIAL", "PLEASE", "FOR", "OF", "THE", "AND", "WITH",
        "USING", "SHOW", "TELL", "GET", "DATA", "INFO", "QUOTE", "BALANCE", "SHEET", "COMPANY", "HIGHLIGHTS",
    }
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9._-]{1,9}\b", text)
    for token in tokens:
        upper = token.upper()
        if upper in stopwords:
            continue
        if len(upper) < 2 or len(upper) > 10:
            continue
        return upper
    return None


def _extract_set_price_intent(text: str) -> str | None:
    raw = text or ""
    lower = raw.lower()
    if not lower:
        return None

    # Guard: ignore pasted multiline transcripts unless this is a direct price command/request line
    direct_like = bool(re.match(r"^\s*!?price\b", lower))
    if "\n" in raw and not direct_like:
        return None

    # Don't extract price when user is modifying a task with prompt/change/update keywords
    task_action_markers = ["change prompt", "update prompt", "prompt of", "create a task", "add a task", "make a task", "remove task", "delete task"]
    if any(marker in lower for marker in task_action_markers):
        return None

    price_markers = ["price", "ราคา", "quote", "หุ้น"]
    request_markers = ["give me", "show", "what is", "what's", "check", "get", "ขอ", "ดู", "เช็ค"]

    if not any(marker in lower for marker in price_markers):
        return None

    symbol = _extract_stock_symbol(text)
    if not symbol:
        return None

    if any(marker in lower for marker in request_markers) or re.search(rf"\b{re.escape(symbol.lower())}\s+price\b", lower):
        return symbol
    return None


def _format_set_price_snapshot(data: dict, symbol: str | None = None) -> str:
    resolved_symbol = (symbol or data.get("symbol") or "").upper() or "UNKNOWN"
    if data.get("price") is None:
        return f"I found {resolved_symbol}, but price is unavailable right now."

    parts = [f"📈 {resolved_symbol} price: {data.get('price')} THB"]
    if data.get("change") is not None:
        parts.append(f"Change: {data.get('change')}")
    if data.get("percent_change") is not None:
        try:
            parts.append(f"Percent: {float(data.get('percent_change')):.2f}%")
        except Exception:
            parts.append(f"Percent: {data.get('percent_change')}%")
    return " | ".join(parts)


def _to_float_or_none(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "n/a"}:
        return None
    text = re.sub(r"[^0-9.+-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _compare_set_vs_siamchart_price(symbol: str) -> str:
    normalized = normalize_stock_symbol(symbol)
    set_data = fetch_set_financial_data(normalized)
    set_price = _to_float_or_none(set_data.get("price"))

    siam_last = None
    try:
        from khundech.web_scraping import get_siamchart_stock_rows as _scrape_rows

        for row in _scrape_rows(limit=1000):
            if normalize_stock_symbol(row.get("Name", "")) == normalized:
                siam_last = _to_float_or_none(row.get("Last"))
                break
    except Exception:
        siam_last = None

    lines = [
        f"🔎 Price compare for {normalized}",
        f"- SET (!setfinancial source): {set_data.get('price') if set_data.get('price') is not None else 'N/A'}",
        f"- Siamchart Last: {siam_last if siam_last is not None else 'N/A'}",
    ]

    if set_price is not None and siam_last is not None:
        diff = round(set_price - siam_last, 4)
        lines.append(f"- Difference (SET - Siamchart): {diff:+g}")
    lines.append("✅ Canonical price used by bot: SET (!setfinancial).")
    return "\n".join(lines)


def _get_balance_metric_value(data: dict, label: str) -> float | None:
    metrics = data.get("balance_metrics")
    if not isinstance(metrics, list):
        return None
    for row in metrics:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        row_label = str(row[0]).strip()
        if row_label == label:
            return _to_float_or_none(row[1])
    return None


def _extract_balance_ratios(data: dict) -> tuple[float | None, float | None, float | None]:
    assets = _get_balance_metric_value(data, "รวมสินทรัพย์")
    liabilities = _get_balance_metric_value(data, "รวมหนี้สิน")
    equity = _get_balance_metric_value(data, "รวมส่วนของผู้ถือหุ้น")
    cash = _get_balance_metric_value(data, "เงินสด")

    debt_to_equity = None
    equity_ratio = None
    cash_to_liabilities = None

    if liabilities is not None and equity not in (None, 0):
        debt_to_equity = liabilities / equity
    if equity is not None and assets not in (None, 0):
        equity_ratio = equity / assets
    if cash is not None and liabilities not in (None, 0):
        cash_to_liabilities = cash / liabilities

    return debt_to_equity, equity_ratio, cash_to_liabilities


def _get_siamchart_metrics_for_symbol(symbol: str) -> dict:
    normalized = normalize_stock_symbol(symbol)
    try:
        from khundech.web_scraping import get_siamchart_stock_rows as _scrape_rows

        for row in _scrape_rows(limit=1000):
            if normalize_stock_symbol(row.get("Name", "")) != normalized:
                continue
            return {
                "roa": _to_float_or_none(row.get("ROA%")),
                "roe": _to_float_or_none(row.get("ROE%")),
                "yield_pct": _to_float_or_none(row.get("Yield%")),
                "pe": _to_float_or_none(row.get("P/E")),
                "pbv": _to_float_or_none(row.get("P/BV")),
            }
    except Exception:
        pass
    return {"roa": None, "roe": None, "yield_pct": None, "pe": None, "pbv": None}


def _compute_stock_health_score(
    *,
    roa: float | None,
    roe: float | None,
    div_yield: float | None,
    debt_to_equity: float | None,
    equity_ratio: float | None,
    cash_to_liabilities: float | None,
) -> tuple[int, float, dict[str, float]]:
    points = 0.0
    available = 0.0
    breakdown: dict[str, float] = {}

    if roa is not None:
        available += 20.0
        if roa >= 15:
            p = 20.0
        elif roa >= 10:
            p = 15.0
        elif roa >= 5:
            p = 10.0
        elif roa > 0:
            p = 5.0
        else:
            p = 0.0
        points += p
        breakdown["ROA"] = p

    if roe is not None:
        available += 25.0
        if roe >= 20:
            p = 25.0
        elif roe >= 15:
            p = 20.0
        elif roe >= 10:
            p = 14.0
        elif roe >= 5:
            p = 8.0
        else:
            p = 0.0
        points += p
        breakdown["ROE"] = p

    if div_yield is not None:
        available += 15.0
        if div_yield >= 6:
            p = 15.0
        elif div_yield >= 4:
            p = 12.0
        elif div_yield >= 2:
            p = 7.0
        elif div_yield > 0:
            p = 3.0
        else:
            p = 0.0
        points += p
        breakdown["Yield"] = p

    if debt_to_equity is not None:
        available += 20.0
        if debt_to_equity <= 0.8:
            p = 20.0
        elif debt_to_equity <= 1.5:
            p = 15.0
        elif debt_to_equity <= 2.5:
            p = 9.0
        elif debt_to_equity <= 3.5:
            p = 4.0
        else:
            p = 0.0
        points += p
        breakdown["D/E"] = p

    if equity_ratio is not None:
        available += 12.0
        if equity_ratio >= 0.50:
            p = 12.0
        elif equity_ratio >= 0.35:
            p = 9.0
        elif equity_ratio >= 0.25:
            p = 6.0
        elif equity_ratio >= 0.15:
            p = 3.0
        else:
            p = 0.0
        points += p
        breakdown["Equity/Assets"] = p

    if cash_to_liabilities is not None:
        available += 8.0
        if cash_to_liabilities >= 0.25:
            p = 8.0
        elif cash_to_liabilities >= 0.15:
            p = 6.0
        elif cash_to_liabilities >= 0.08:
            p = 4.0
        elif cash_to_liabilities >= 0.03:
            p = 2.0
        else:
            p = 0.0
        points += p
        breakdown["Cash/Liabilities"] = p

    confidence = (available / 100.0) if available > 0 else 0.0
    normalized = int(round((points / available) * 100)) if available > 0 else 0
    normalized = max(0, min(100, normalized))
    return normalized, confidence, breakdown


def _format_set_financial_with_analysis(data: dict) -> str:
    base = format_set_financial_summary(data)

    symbol = normalize_stock_symbol(data.get("symbol", ""))
    siam = _get_siamchart_metrics_for_symbol(symbol) if symbol else {"roa": None, "roe": None, "yield_pct": None}
    debt_to_equity, equity_ratio, cash_to_liabilities = _extract_balance_ratios(data)
    score, confidence, _ = _compute_stock_health_score(
        roa=siam.get("roa"),
        roe=siam.get("roe"),
        div_yield=siam.get("yield_pct"),
        debt_to_equity=debt_to_equity,
        equity_ratio=equity_ratio,
        cash_to_liabilities=cash_to_liabilities,
    )

    if score >= 75:
        verdict = "🟢 Stronger overall profile (rule-based check)"
    elif score >= 55:
        verdict = "🟡 Mixed overall profile (rule-based check)"
    else:
        verdict = "🔴 Weaker overall profile (rule-based check)"

    analysis_lines = [
        "",
        "🧠 Unified health analysis",
        f"- Overall score: {score}/100",
        f"- Verdict: {verdict}",
        f"- Data coverage: {int(round(confidence * 100))}%",
    ]
    if siam.get("roa") is not None:
        analysis_lines.append(f"- ROA: {siam['roa']:.2f}%")
    if siam.get("roe") is not None:
        analysis_lines.append(f"- ROE: {siam['roe']:.2f}%")
    if siam.get("yield_pct") is not None:
        analysis_lines.append(f"- Dividend Yield: {siam['yield_pct']:.2f}%")
    if debt_to_equity is not None:
        analysis_lines.append(f"- Debt/Equity: {debt_to_equity:.2f}")
    if equity_ratio is not None:
        analysis_lines.append(f"- Equity/Assets: {equity_ratio:.2f}")
    if cash_to_liabilities is not None:
        analysis_lines.append(f"- Cash/Liabilities: {cash_to_liabilities:.2f}")

    analysis_lines.append("- Note: Unified rule-based score used across `!setfinancial` and `stock_seeking`, not investment advice.")
    return base + "\n" + "\n".join(analysis_lines)


def _extract_set_financial_intent(text: str) -> str | None:
    raw = text or ""
    lower = raw.lower().strip()
    if not lower:
        return None

    # Guard: ignore long pasted transcripts/outputs that merely contain !setfinancial
    # unless the message itself starts like a direct setfinancial request.
    is_direct_like = bool(re.match(r"^\s*!?i?set(?:financial|balance)\b", lower))
    if ("\n" in raw and not is_direct_like):
        return None

    # Support direct command-like typos without '!':
    # setfinancial AAV, isetfinancial AAV, setbalance AAV, isetbalance AAV
    patterns = [
        r"\b!?i?setfinancial\s+([A-Za-z0-9._-]{2,10})\b",
        r"\b!?i?setbalance\s+([A-Za-z0-9._-]{2,10})\b",
        r"\bset\s+financial\s+([A-Za-z0-9._-]{2,10})\b",
        r"\bset\s+balance\s+([A-Za-z0-9._-]{2,10})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower, re.I)
        if match:
            candidate = re.sub(r"[^A-Za-z0-9._-]", "", match.group(1)).upper()
            stopwords = {
                "YOU", "YOUR", "YOURSELF", "ME", "MY", "MINE",
                "FOR", "THE", "THIS", "THAT", "THEY", "THEM", "BUT", "AND",
                "ANALYSIS", "ANALYZE", "UPGRADE", "SKILL", "SUPPORT",
            }
            if candidate and candidate not in stopwords:
                return candidate

    return None


def _extract_auto_command_preference_intent(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    remember_markers = ["remember", "จำไว้", "จำ", "จดจำ"]
    auto_markers = ["auto", "match", "command", "natural language", "text command", "run it"]
    return any(marker in lower for marker in remember_markers) and any(marker in lower for marker in auto_markers)


def _extract_time_now_intent(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    return any(marker in lower for marker in ["time now", "what is time", "current time", "เวลาตอนนี้", "time now is"])


def _extract_timezone_intent(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    return any(marker in lower for marker in ["time zone", "timezone", "โซนเวลา", "tiume zone"])


def _extract_file_path(text: str) -> str | None:
    match = re.search(r"([A-Za-z0-9_./-]+\.(?:py|txt|md|json|yml|yaml|toml|ini|cfg))", text)
    if not match:
        return None
    path = match.group(1).replace("\\", "/")
    if path.startswith("/app/"):
        path = path[5:]
    return path


def _extract_content_hint(text: str) -> str | None:
    pattern = r"(?:contain|contains|content|with|string|strin|ข้อความ)\s+(.+)$"
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    content = match.group(1).strip().strip('"').strip("'")
    if not content:
        return None
    return content


def _extract_file_intent(text: str) -> dict | None:
    lower = text.lower().strip()
    path = _extract_file_path(text)

    create_markers = ["create", "make", "สร้าง"]
    update_markers = ["update", "edit", "write", "overwrite", "แก้", "เขียน"]
    delete_markers = ["delete", "remove", "ลบ"]

    if any(marker in lower for marker in create_markers):
        if not path:
            return None
        content = _extract_content_hint(text)
        if not content and path.endswith(".txt"):
            content = "hello world"
        if not content:
            content = ""
        return {"action": "create", "path": path, "content": content}

    if any(marker in lower for marker in update_markers):
        if not path:
            return None
        content = _extract_content_hint(text)
        if not content:
            return None
        return {"action": "update", "path": path, "content": content}

    if any(marker in lower for marker in delete_markers):
        if not path:
            return None
        return {"action": "delete", "path": path}

    return None


def _resolve_followup_file_intent(text: str, history: list) -> dict | None:
    lower = text.lower().strip()
    create_markers = ["create it", "make it", "do it", "สร้างมัน", "ทำเลย", "create again", "try to create it again"]
    delete_markers = ["delete it", "remove it", "then remove it", "ลบมัน", "ลบเลย", "delete this", "remove this"]
    update_markers = ["update it", "edit it", "write it", "แก้มัน", "เขียนทับ"]

    target_action = None
    if any(marker in lower for marker in create_markers):
        target_action = "create"
    elif any(marker in lower for marker in delete_markers):
        target_action = "delete"
    elif any(marker in lower for marker in update_markers):
        target_action = "update"

    if not target_action:
        return None

    for turn in reversed(history[-12:]):
        user_text = (turn.get("user") or "").strip()
        intent = _extract_file_intent(user_text)
        if not intent:
            continue
        if target_action == "create":
            if intent.get("action") == "create":
                return intent
            continue

        path = intent.get("path")
        if not path:
            continue

        if target_action == "delete":
            return {"action": "delete", "path": path}

        if target_action == "update":
            content = _extract_content_hint(text) or intent.get("content")
            if content:
                return {"action": "update", "path": path, "content": content}
            return None
    return None


def _run_file_intent(intent: dict) -> str:
    action = intent.get("action")
    path = intent.get("path")
    content = intent.get("content")

    operation = {"action": action, "path": path}
    if action in {"create", "update"}:
        if not isinstance(content, str) or not content:
            return "Please provide file content, for example: create test.txt with hello world"
        operation["content"] = content + ("\n" if not content.endswith("\n") else "")

    payload = {"operations": [operation], "post_apply": "none"}
    try:
        changed_files, backup_paths, _, verification = apply_self_improvement(payload)
    except Exception as exc:
        return f"File operation failed: {exc}\n{format_path_state(path)}"

    if not changed_files:
        return "No file changes were applied."

    state = format_path_state(path)
    lines = [f"✅ File operation applied: {action} /app/{path}"]
    if backup_paths:
        lines.append("Backup: " + backup_paths[-1])
    if verification:
        check = verification[-1]
        lines.append(
            f"Verification: before_exists={check.get('before', {}).get('exists')}, after_exists={check.get('after', {}).get('exists')}"
        )
    lines.append(state)
    return "\n".join(lines)


def _looks_like_self_upgrade_goal(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False

    explicit_upgrade = [
        "self improve",
        "self-improve",
        "upgrade yourself",
        "upgrade your self",
        "process the upgrade",
        "improve yourself",
        "อัพเกรดตัวเอง",
        "อัปเกรดตัวเอง",
        "เขียนโค้ดอัพเกรดตัวเอง",
    ]
    if any(token in lower for token in explicit_upgrade):
        return True

    upgrade_markers = ["upgrade", "improve", "อัพเกรด", "อัปเกรด"]
    code_markers = ["code", "coding", "โค้ด", "automation", "อัตโนมัติ", "prompt", "pormt", "skill", "analy", "analysis"]
    return any(token in lower for token in upgrade_markers) and any(token in lower for token in code_markers)


def _extract_upgrade_followup_goal(text: str, history: list) -> str | None:
    lower = (text or "").strip().lower()
    if not lower:
        return None

    followup_markers = {
        "upgrade",
        "process the upgrade",
        "do the upgrade",
        "proceed with upgrade",
        "proceed upgrade",
        "yes upgrade",
        "อัพเกรด",
        "อัปเกรด",
        "ดำเนินการอัปเกรด",
    }
    if lower not in followup_markers:
        return None

    for turn in reversed((history or [])[-12:]):
        assistant_text = str(turn.get("assistant") or "")
        quoted_goal = re.search(r"!upgrade\s+[\"']([^\"']+)[\"']", assistant_text, re.I)
        if quoted_goal:
            goal = quoted_goal.group(1).strip()
            if goal:
                return goal

        plain_goal = re.search(r"!upgrade\s+([^\n]+)", assistant_text, re.I)
        if plain_goal:
            goal = plain_goal.group(1).strip().strip('"').strip("'")
            if goal:
                return goal

    return "Improve task automation reliability: ensure interval updates persist and operational natural-language commands route deterministically."


async def _run_self_upgrade_goal(goal: str) -> str:
    try:
        proposal = await propose_self_improvement(genai_client, AI_MODEL, goal)
        changed_files, backup_paths, post_apply, verification = apply_self_improvement(proposal)
    except Exception as exc:
        return f"⚠️ Self-improvement failed: {exc}"

    if not changed_files:
        return "No code changes were produced."

    summary = proposal.get("summary", "No summary provided.")
    record_self_improvement(summary, changed_files, backup_paths, post_apply, verification)

    if any(path in {"requirements.txt", "Dockerfile"} for path in changed_files):
        post_apply = "rebuild"

    changed = ", ".join(changed_files)
    backups = "\n".join(backup_paths[-5:]) if backup_paths else "No backups created."

    message = (
        f"🛠️ Self-improvement applied to: {changed}\n"
        f"Summary: {summary}\n"
        f"Backups:\n{backups}"
    )

    if post_apply == "restart":
        message += "\nPost apply: restart required (use !restart)."
    elif post_apply == "rebuild":
        message += "\nPost apply: rebuild required (run docker compose up --build -d)."

    return message


def format_path_state(relative_path: str) -> str:
    state = get_path_state(relative_path)
    return (
        "📁 Path check\n"
        f"- Path: {state['path'] or '(empty)'}\n"
        f"- Allowed: {state['allowed']}\n"
        f"- Exists: {state['exists']}\n"
        f"- Is file: {state['is_file']}\n"
        f"- Is dir: {state['is_dir']}\n"
        f"- Resolved: {state['resolved'] or 'outside-workspace'}"
    )


def format_tasks_file_state() -> str:
    tasks_path = Path(DATA_DIR) / "tasks.json"
    exists = tasks_path.exists()
    size = tasks_path.stat().st_size if exists else 0
    return (
        "📁 Task store check\n"
        f"- Path: {tasks_path}\n"
        f"- Exists: {exists}\n"
        f"- Size: {size} bytes"
    )


def _extract_task_list_intent(text: str) -> bool:
    # Detect natural-language requests to list scheduled tasks (cronjobs).
    lower = (text or "").lower()
    if not lower:
        return False
    patterns = [
        "cronjob",
        "cron job",
        "list cron",
        "show cron",
        "how many cron",
        "cronjobs we have",
        "cronjob we have",
        "automation task",
        "tasks we have",
        "task we have",
        "all task",
        "all tasks",
        "list tasks",
        "show tasks",
        "task list",
        "มีงานอะไร",
        "รายการงาน",
    ]
    return any(pattern in lower for pattern in patterns)


def _run_list_tasks_intent() -> str:
    # Render scheduled tasks with interval, status, and last-run metadata.
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []
    if not tasks_list:
        return "🧩 **Cronjobs: 0**\nNo scheduled tasks found in /app/data/tasks.json"

    lines = [f"🧩 **Cronjobs: {len(tasks_list)}**"]
    for index, task in enumerate(tasks_list[:20], start=1):
        if not isinstance(task, dict):
            continue
        name = task.get("name", "(unnamed)")
        interval = task.get("interval", "?")
        unit = task.get("unit", "?")
        active = bool(task.get("active", True))
        status = "enabled" if active else "disabled"
        task_id = task.get("id", "?")
        last_run_text = _format_time_utc7(task.get("last_run"))
        prompt = str(task.get("prompt", ""))
        prompt_preview = (prompt[:120] + "...") if len(prompt) > 120 else prompt
        lines.append(f"\n**{index}) #{task_id} {name}**")
        lines.append(f"• Interval: every {interval} {unit}")
        lines.append(f"• Status: {status}")
        lines.append(f"• Last run: {last_run_text}")
        lines.append(f"• Prompt: {prompt_preview}")

    lines.append(f"\n{format_tasks_file_state()}")
    return "\n".join(lines)


def _extract_tasks_raw_intent(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    has_tasks_file = "tasks.json" in lower or "/app/data/tasks.json" in lower
    has_raw_marker = any(marker in lower for marker in ["raw", "show", "file data", "ดูไฟล์", "แสดงไฟล์"])
    return has_tasks_file and has_raw_marker


def _run_tasks_raw_intent() -> str:
    tasks_path = Path(DATA_DIR) / "tasks.json"
    if not tasks_path.exists():
        return "⚠️ /app/data/tasks.json not found on disk."
    content = tasks_path.read_text(encoding="utf-8")
    clipped = content[:3500]
    note = "\n(Trimmed to first 3500 chars)" if len(content) > 3500 else ""
    return (
        "📂 Raw data for /app/data/tasks.json\n\n"
        f"{clipped}{note}\n\n"
        "Verification:\n"
        "- File path: /app/data/tasks.json\n"
        f"- Exists: {tasks_path.exists()}\n"
        f"- Size: {tasks_path.stat().st_size} bytes"
    )


def _extract_raw_file_view_intent(text: str) -> str | None:
    lower = (text or "").lower().strip()
    if not lower:
        return None
    raw_markers = ["show raw", "raw data", "raw file", "show file data", "cat ", "show content"]
    if not any(marker in lower for marker in raw_markers):
        return None
    path = _extract_file_path(text)
    if not path:
        return None
    return path


def _run_raw_file_view_intent(path: str) -> str:
    state = get_path_state(path)
    if not state.get("allowed"):
        return f"⚠️ Access denied for path: /app/{path}\n{format_path_state(path)}"
    if not state.get("exists") or not state.get("is_file"):
        return f"⚠️ File not found: /app/{path}\n{format_path_state(path)}"

    try:
        resolved = Path(str(state.get("resolved")))
        content = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return f"⚠️ Failed to read /app/{path}: {exc}\n{format_path_state(path)}"

    clipped = content[:3500]
    note = "\n(Trimmed to first 3500 chars)" if len(content) > 3500 else ""
    return (
        f"📂 Raw data for /app/{path}\n\n"
        f"{clipped}{note}\n\n"
        "Verification:\n"
        f"- File path: /app/{path}\n"
        f"- Exists: {state.get('exists')}\n"
        f"- Size: {Path(str(state.get('resolved'))).stat().st_size} bytes"
    )


def _extract_simple_task_create_intent(text: str) -> dict | None:
    lower = (text or "").lower()
    if not lower:
        return None

    if not ("task" in lower or "งาน" in lower):
        return None
    if not any(marker in lower for marker in ["make", "create", "add", "ตั้ง", "สร้าง"]):
        return None

    # Try pattern: create a task name "stock seeking" and prompt of the task is "..."
    named_pattern = re.search(
        r'(?:create|add)\s+a\s+task\s+name\s+["\']?([^"\']+)["\']?\s+and\s+prompt\s+(?:of\s+the\s+task)?\s+is\s+["\'](.+?)["\']?\s+(?:repater|repeat|every)\s+(?:the\s+task\s+)?every\s*(\d+)\s*(minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี)',
        text,
        re.I | re.DOTALL,
    )
    
    if named_pattern:
        task_name_raw = named_pattern.group(1).strip().strip('"').strip("'")
        prompt = named_pattern.group(2).strip().strip('"').strip("'")
        interval = int(named_pattern.group(3))
        raw_unit = named_pattern.group(4).lower()
        
        normalized_name = re.sub(r"[^a-z0-9]+", "_", task_name_raw.lower()).strip("_")
        task_name = (normalized_name or "task")[:40]
        
        unit_map = {
            "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes", "นาที": "minutes",
            "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "ชั่วโมง": "hours",
            "day": "days", "days": "days", "วัน": "days",
            "month": "months", "months": "months", "เดือน": "months",
            "year": "years", "years": "years", "ปี": "years",
        }
        unit = unit_map.get(raw_unit)
        
        if interval > 0 and unit and prompt:
            return {"name": task_name, "prompt": prompt, "interval": interval, "unit": unit}

    # Original pattern: make a task to <action> every <N> <unit>
    interval_match = re.search(
        r"(?:every|ervery|run\s+every|run\s+ervery|ทุก)\s*(\d+)\s*(minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี)",
        text,
        re.I,
    )
    if not interval_match:
        return None

    interval = int(interval_match.group(1))
    raw_unit = interval_match.group(2).lower()
    unit_map = {
        "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes", "นาที": "minutes",
        "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "ชั่วโมง": "hours",
        "day": "days", "days": "days", "วัน": "days",
        "month": "months", "months": "months", "เดือน": "months",
        "year": "years", "years": "years", "ปี": "years",
    }
    unit = unit_map.get(raw_unit)
    if interval <= 0 or not unit:
        return None

    action_match = re.search(
        r"(?:task\s+to|task\s+for|task\s*:\s*|งาน\s*ให้|งาน\s*เพื่อ|to)\s*(.+?)\s*(?:every|ervery|run\s+every|run\s+ervery|ทุก)\s*\d+\s*(?:minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี)",
        text,
        re.I,
    )
    action_text = action_match.group(1).strip() if action_match else "say hello"
    if not action_text:
        action_text = "say hello"

    normalized_name = re.sub(r"[^a-z0-9]+", "_", action_text.lower()).strip("_")
    task_name = (normalized_name or "automation_task")[:40]
    prompt = action_text[0].upper() + action_text[1:] if len(action_text) > 1 else action_text.upper()
    return {"name": task_name, "prompt": prompt, "interval": interval, "unit": unit}


def _extract_task_remove_intent(text: str) -> str | None:
    lower = (text or "").lower()
    if not lower:
        return None

    remove_markers = ["remove", "delete", "ลบ"]
    if not any(marker in lower for marker in remove_markers):
        return None
    if "task" not in lower and "tasks" not in lower and "tash" not in lower and "งาน" not in lower:
        return None

    patterns = [
        r"(?:remove|delete|ลบ)\s+(?:task|tash|tasks)\s+([a-zA-Z0-9_\- ]+)",
        r"(?:remove|delete|ลบ)\s+([a-zA-Z0-9_\- ]+)\s+(?:task|tash|tasks)",
        r"(?:ลบงาน)\s+([a-zA-Z0-9_\- ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            candidate = match.group(1).strip().strip('"').strip("'")
            if candidate:
                # Filter out articles (the, this, a, an) as noise words
                noise_words = {"the", "this", "a", "an", "tha", "thi"}
                if candidate.lower() in noise_words:
                    return "__FALLBACK__"  # Signal to use fallback logic
                return candidate
    
    # Check for patterns like "remove the task" without task name → fallback
    if re.search(r"(?:remove|delete|ลบ)\s+(?:the|this|a|an)?\s*(?:task|tash|tasks)\s*$", text, re.I):
        return "__FALLBACK__"
    
    return None


def _normalize_task_lookup_name(name: str) -> str:
    cleaned = (name or "").strip().lower()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def _run_remove_task_by_name(task_name: str) -> str:
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []

    normalized = _normalize_task_lookup_name(task_name)
    kept = []
    removed = None
    for task in tasks_list:
        if not isinstance(task, dict):
            kept.append(task)
            continue
        name = str(task.get("name", "")).strip().lower()
        normalized_name = _normalize_task_lookup_name(name)
        if removed is None and (
            name == normalized
            or normalized_name == normalized
            or name == task_name.strip().lower()
            or normalized in normalized_name
            or normalized_name in normalized
        ):
            removed = task
            continue
        kept.append(task)

    if removed is None:
        return f"Task not found: {task_name}"

    data["tasks"] = kept
    save_scheduled_tasks(data)

    return (
        f"✅ Removed task: {removed.get('name', '(unnamed)')}\n"
        f"- Task ID: {removed.get('id', '?')}\n"
        f"- Remaining tasks: {len(kept)}\n"
        f"{format_tasks_file_state()}"
    )


def _extract_task_run_intent(text: str) -> str | None:
    lower = (text or "").lower()
    if not lower:
        return None
    if re.search(r"\b(run|execute|trigger|start)\s+the\s+task\s+now\b", lower):
        data = load_scheduled_tasks()
        tasks_list = [item for item in data.get("tasks", []) if isinstance(item, dict) and item.get("active", True)]
        if len(tasks_list) == 1:
            return str(tasks_list[0].get("name", "")).strip()
    match = re.search(r"(?:run|execute|trigger|start)\s+task\s+(.+)$", text, re.I)
    if not match:
        return None
    task_name = match.group(1).strip().strip('"').strip("'")
    if not task_name:
        return None
    return task_name


def _extract_task_schedule_update_intent(text: str) -> dict | None:
    lower = (text or "").lower()
    if not lower:
        return None

    has_task_marker = ("task" in lower or "งาน" in lower)
    has_interval_marker = ("interval" in lower or "ความถี่" in lower)
    if not has_task_marker and not has_interval_marker:
        return None
    if not any(token in lower for token in ["change", "update", "up date", "uo date", "set", "every", "ervery", "instead", "จาก", "เป็น"]):
        return None

    interval_match = re.search(
        r"(?:every|ervery|run\s+every|run\s+ervery|ทุก)\s*(\d+)\s*(minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี)",
        text,
        re.I,
    )
    interval = None
    unit = None
    if not interval_match:
        interval_match = re.search(
            r"(?:to|เป็น)\s*(\d+)\s*(minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี)",
            text,
            re.I,
        )
    if interval_match:
        interval = int(interval_match.group(1))
        raw_unit = interval_match.group(2).lower()
        unit_map = {
            "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes", "นาที": "minutes",
            "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "ชั่วโมง": "hours",
            "day": "days", "days": "days", "วัน": "days",
            "month": "months", "months": "months", "เดือน": "months",
            "year": "years", "years": "years", "ปี": "years",
        }
        unit = unit_map.get(raw_unit)
        if interval <= 0 or not unit:
            return None
    else:
        # Accept shorthand JSON-style phrase: change "interval": 6 to 1
        short_interval_match = re.search(r"interval\s*\"?\s*:?\s*(\d+)\s*(?:to|->|เป็น)\s*(\d+)", text, re.I)
        if not short_interval_match:
            return None
        interval = int(short_interval_match.group(2))
        if interval <= 0:
            return None

    task_id = None
    id_patterns = [
        r"task\s*#?(\d+)",
        r"งาน\s*#?(\d+)",
        r"#(\d+)",
    ]
    for pat in id_patterns:
        m = re.search(pat, lower, re.I)
        if m:
            task_id = int(m.group(1))
            break

    task_name = None
    if task_id is None:
        m = re.search(
            r"(?:change|update|up\s*date|uo\s*date|set)\s+task\s+([a-zA-Z0-9_\- ]+?)\s+(?:(?:to\s+)?(?:do\s+)?(?:every|ervery)\b|(?:to|เป็น)\s*\d+\s*(?:minutes?|mins?|min|hours?|hrs?|hr|days?|months?|years?|นาที|ชั่วโมง|วัน|เดือน|ปี))",
            text,
            re.I,
        )
        if m:
            candidate = m.group(1).strip().strip('"').strip("'")
            if candidate and candidate.lower() not in {"the", "this", "a", "an"}:
                task_name = candidate

    if task_id is None and not task_name:
        return {"task_id": None, "task_name": None, "interval": interval, "unit": unit}

    return {"task_id": task_id, "task_name": task_name, "interval": interval, "unit": unit}


def _run_update_task_schedule(task_id: int | None, task_name: str | None, interval: int, unit: str | None) -> str:
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []
    target = None

    for task in tasks_list:
        if not isinstance(task, dict):
            continue
        if task_id is not None and task.get("id") == task_id:
            target = task
            break

    if target is None and task_name:
        normalized = _normalize_task_lookup_name(task_name)
        for task in tasks_list:
            if not isinstance(task, dict):
                continue
            normalized_name = _normalize_task_lookup_name(str(task.get("name", "")))
            if (
                normalized_name == normalized
                or normalized in normalized_name
                or normalized_name in normalized
            ):
                target = task
                break

    if target is None and task_id is None and not task_name:
        active_tasks = [t for t in tasks_list if isinstance(t, dict) and t.get("active", True)]
        if len(active_tasks) == 1:
            target = active_tasks[0]

    if target is None:
        return "❌ Task not found for schedule update."

    old_interval = target.get("interval")
    old_unit = target.get("unit")
    target["interval"] = int(interval)
    target["unit"] = unit or old_unit or "hours"
    save_scheduled_tasks(data)

    return (
        f"✅ Updated task schedule: {target.get('name', '(unnamed)')}\n"
        f"- Task ID: {target.get('id', '?')}\n"
        f"- Old schedule: every {old_interval} {old_unit}\n"
        f"- New schedule: every {target.get('interval')} {target.get('unit')}\n"
        f"{format_tasks_file_state()}"
    )


def _extract_task_prompt_recovery_intent(text: str) -> dict | None:
    lower = (text or "").lower()
    if not lower:
        return None

    recovery_markers = ["recover", "recovery", "rollback", "restore", "ย้อน", "คืนค่า"]
    prompt_markers = ["prompt", "promt", "promp"]
    if not any(marker in lower for marker in recovery_markers):
        return None
    if not any(marker in lower for marker in prompt_markers):
        return None

    task_id = None
    m = re.search(r"(?:task|งาน)\s*#?(\d+)", lower)
    if m:
        task_id = int(m.group(1))

    task_name = None
    if task_id is None:
        m = re.search(r"(?:task|งาน)\s+([a-zA-Z0-9_\- ]+)$", text, re.I)
        if m:
            candidate = m.group(1).strip().strip('"').strip("'")
            if candidate and candidate.lower() not in {"the", "this", "a", "an", "back"}:
                task_name = candidate

    if task_id is None and not task_name:
        return {"task_id": None, "task_name": None}
    return {"task_id": task_id, "task_name": task_name}


def _run_recover_task_prompt(task_id: int | None, task_name: str | None) -> str:
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []
    target = None

    if task_id is not None:
        for task in tasks_list:
            if isinstance(task, dict) and task.get("id") == task_id:
                target = task
                break
    elif task_name:
        normalized = _normalize_task_lookup_name(task_name)
        for task in tasks_list:
            if not isinstance(task, dict):
                continue
            t_norm = _normalize_task_lookup_name(str(task.get("name", "")))
            if t_norm == normalized or normalized in t_norm or t_norm in normalized:
                target = task
                break
    elif len(tasks_list) == 1:
        target = tasks_list[0]

    if target is None:
        if not tasks_list:
            return "❌ No tasks found."
        return "❌ Task not found for prompt recovery."

    history = target.get("prompt_history", []) if isinstance(target.get("prompt_history"), list) else []
    recovered = None
    if history:
        latest = history.pop()
        if isinstance(latest, dict):
            recovered = str(latest.get("prompt", "")).strip()
        elif isinstance(latest, str):
            recovered = latest.strip()

    if not recovered:
        name = str(target.get("name", ""))
        current_prompt = str(target.get("prompt", "")).strip().lower()
        m = re.match(r"check_price_of_stock_([a-z0-9._-]+)$", name, re.I)
        if m and current_prompt.startswith("do every"):
            recovered = f"!setfinancial {m.group(1).upper()}"

    if not recovered:
        return "❌ No previous prompt snapshot found to recover."

    target["prompt"] = recovered
    target["prompt_history"] = history[-10:]
    save_scheduled_tasks(data)

    return (
        f"✅ Recovered task prompt: {target.get('name', '(unnamed)')}\n"
        f"- Task ID: {target.get('id', '?')}\n"
        f"- New prompt: {recovered}\n"
        f"{format_tasks_file_state()}"
    )


def _extract_task_prompt_update_intent(text: str) -> dict | None:
    lower = (text or "").lower()
    if not lower:
        return None
    prompt_markers = ["update prompt", "change prompt", "แก้ prompt", "เปลี่ยน prompt", "promp", "promt"]
    if not any(marker in lower for marker in prompt_markers):
        return None

    task_name = None
    task_id = None

    # Pattern: "change prompt of <name> to <new>" / "change prompt of task 2 to <new>"
    of_name_patterns = [
        # "change prompt of task 2 to ..."  OR  "change prompt of check_price to ..."
        r"(?:update|change)\s+prompt\s+of\s+task\s+#?(\d+)\s+to\s+([\s\S]+)$",
        r"(?:update|change)\s+prompt\s+of\s+([a-zA-Z0-9_\- ]+?)\s+to\s+([\s\S]+)$",
    ]
    for pat in of_name_patterns:
        m = re.search(pat, text, re.I)
        if m:
            name_or_id = m.group(1).strip()
            new_prompt = m.group(2).strip().strip('"').strip("'")
            if name_or_id.isdigit():
                task_id = int(name_or_id)
            else:
                noise = {"update", "change", "task", "automation", "automate", "the", "a", "an"}
                if name_or_id.lower() not in noise and new_prompt:
                    task_name = name_or_id
            if new_prompt:
                # Resolve task_name via id if needed
                if task_id and not task_name:
                    data = load_scheduled_tasks()
                    for t in data.get("tasks", []):
                        if isinstance(t, dict) and t.get("id") == task_id:
                            task_name = str(t.get("name", ""))
                            break
                if not task_name:
                    data = load_scheduled_tasks()
                    active = [t for t in data.get("tasks", []) if isinstance(t, dict) and t.get("active", True)]
                    if len(active) == 1:
                        task_name = str(active[0].get("name", ""))
                if task_name:
                    payload = {"task_name": task_name, "prompt": new_prompt}
                    if task_id:
                        payload["task_id"] = task_id
                    return payload

    id_match = re.search(r"#(\d+)\s*([a-zA-Z0-9_\- ]+)?", text)
    if id_match:
        task_id = int(id_match.group(1))
    name_patterns = [
        r"^\s*([a-zA-Z0-9_\- ]+?)\s+task\s+(?:update|change)\s+(?:prompt|promp|promt)\b",
        r"(?:automation|automate)?\s*task\s+([a-zA-Z0-9_\- ]+?)\s+(?:prompt|promp|promt)\b",
        r"(?:task|งาน)\s+([a-zA-Z0-9_\- ]+?)\s+(?:prompt|promp|promt)\b",
        r"^\s*([a-zA-Z0-9_\- ]+?)\s+(?:update|change)\s+(?:prompt|promp|promt)\b",
    ]
    name_match = None
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.I)
        if name_match:
            break
    if name_match:
        task_name = name_match.group(1).strip().strip('"').strip("'")
        if task_name.lower() in {"update", "change", "task", "automation", "automate"}:
            task_name = None

    if not task_name and id_match and id_match.group(2):
        candidate_name = id_match.group(2).strip().strip('"').strip("'")
        if candidate_name:
            task_name = candidate_name

    prompt_text = None
    quoted_patterns = [
        r"(?:update|change)\s+(?:prompt|promp|promt)\s+of\s+the\s+task\s+to(?:\s+this)?\s+[\"']([\s\S]+)[\"']\s*$",
        r"(?:update|change)\s+(?:automation|automate)?\s*task(?:\s+[a-zA-Z0-9_\- ]+?)?\s+(?:prompt|promp|promt)\s+to(?:\s+this)?\s+[\"']([\s\S]+)[\"']\s*$",
        r"^[a-zA-Z0-9_\- ]+?\s+(?:update|change)\s+(?:prompt|promp|promt)\s+of\s+the\s+task\s+to(?:\s+this)?\s+[\"']([\s\S]+)[\"']\s*$",
    ]
    for pattern in quoted_patterns:
        quoted_match = re.search(pattern, text, re.I)
        if quoted_match:
            prompt_text = quoted_match.group(1).strip()
            break

    if not prompt_text:
        direct_patterns = [
            r"(?:update|change)\s+(?:prompt|promp|promt)\s+of\s+the\s+task\s+to(?:\s+this)?\s+([\s\S]+)$",
            r"(?:update|change)\s+(?:automation|automate)?\s*task(?:\s+[a-zA-Z0-9_\- ]+?)?\s+(?:prompt|promp|promt)\s+to(?:\s+this)?\s+([\s\S]+)$",
            r"^[a-zA-Z0-9_\- ]+?\s+(?:update|change)\s+(?:prompt|promp|promt)\s+of\s+the\s+task\s+to(?:\s+this)?\s+([\s\S]+)$",
        ]
        for pattern in direct_patterns:
            direct_match = re.search(pattern, text, re.I)
            if direct_match:
                prompt_text = direct_match.group(1).strip().strip('"').strip("'")
                break

    if not prompt_text:
        return None

    if not task_name:
        data = load_scheduled_tasks()
        active_tasks = [item for item in data.get("tasks", []) if isinstance(item, dict) and item.get("active", True)]
        if len(active_tasks) == 1:
            task_name = str(active_tasks[0].get("name", "")).strip()

    if not task_name:
        return None

    payload = {"task_name": task_name, "prompt": prompt_text}
    if task_id is not None:
        payload["task_id"] = task_id
    return payload


def _extract_direct_run_intent(text: str) -> str | None:
    # Detect 'try this', 'run this prompt', 'can you execute this' + a following text block.
    # Returns the extracted prompt string to run directly, or None.
    raw = (text or "").strip()
    if not raw:
        return None
    lower = raw.lower()

    trigger_phrases = [
        "can you try this to run this",
        "try this to run this",
        "try to run this",
        "can you run this",
        "try to execute this",
        "can you execute this",
        "execute this prompt",
        "run this prompt",
        "run this please",
        "try running this",
        "ลองรันอันนี้",
        "ลองรัน",
        "รันอันนี้",
    ]

    extracted = None
    for phrase in trigger_phrases:
        if lower.startswith(phrase):
            extracted = raw[len(phrase):].strip().strip('"').strip("'").strip()
            break

    # Also handle: trigger phrase anywhere in first 80 chars, rest is the prompt
    if extracted is None:
        for phrase in trigger_phrases:
            idx = lower.find(phrase)
            if 0 <= idx <= 60:
                after = raw[idx + len(phrase):].strip().strip('"').strip("'").strip()
                if len(after) > 50:  # must have substantial content after
                    extracted = after
                    break

    if extracted and len(extracted) > 50:
        return extracted
    return None


def _extract_brief_chat_reply(text: str) -> str | None:
    lower = (text or "").strip().lower()
    if not lower:
        return None

    greeting_tokens = {
        "hi", "hello", "hey", "yo", "hiya", "sup", "สวัสดี", "หวัดดี"
    }
    thanks_tokens = {
        "thanks", "thank you", "thx", "ty", "ขอบคุณ", "ขอบใจ"
    }
    positive_ack_tokens = {
        "great", "good", "nice", "cool", "ok", "okay", "sure", "เยี่ยม", "โอเค", "ดี"
    }

    if lower in greeting_tokens:
        return "Hello."
    if lower in thanks_tokens:
        return "You're welcome."
    if lower in positive_ack_tokens:
        return "Understood."
    if lower in {"how are you", "how are you?", "เป็นไงบ้าง", "สบายดีไหม"}:
        return "Ready."
    return None


def _looks_like_task_prompt_update_request(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    # Don't trigger on "show" commands — those are read-only intent
    show_markers = ["show prompt", "display prompt", "what is prompt", "what's prompt", "view prompt", "read prompt", "get prompt"]
    if any(marker in lower for marker in show_markers):
        return False
    prompt_markers = ["update prompt", "change prompt", "prompt of the task", "promp", "promt"]
    task_markers = ["task", "automation task", "automate task", "งาน"]
    has_prompt_marker = any(marker in lower for marker in prompt_markers)
    has_task_reference = any(marker in lower for marker in task_markers) or re.search(r"#\d+", lower) is not None
    return has_prompt_marker and has_task_reference


def _extract_task_show_prompt_intent(text: str) -> dict | None:
    # Detect 'show prompt of task X', 'show prompt task 2', etc.
    lower = (text or "").lower()
    if not lower:
        return None
    show_markers = ["show prompt", "display prompt", "what is prompt", "what's prompt", "view prompt", "read prompt", "get prompt"]
    if not any(marker in lower for marker in show_markers):
        return None

    # Match by task id: "show prompt of task 2", "show prompt task #2"
    id_match = re.search(r"(?:task\s*)?#?(\d+)", lower)
    if id_match:
        return {"task_id": int(id_match.group(1)), "task_name": None}

    # Match by name: "show prompt of stock seeking"
    name_match = re.search(
        r"(?:show|display|view|get|read)\s+prompt\s+(?:of\s+)?(?:task\s+)?([a-zA-Z0-9_\- ]+?)(?:\s*$)",
        text, re.I
    )
    if name_match:
        candidate = name_match.group(1).strip()
        if candidate.lower() not in {"task", "the", "a", "an", "this"}:
            return {"task_id": None, "task_name": candidate}

    # fallback — show prompt of whatever single active task exists
    return {"task_id": None, "task_name": None}


def _run_show_task_prompt(task_id: int | None, task_name: str | None) -> str:
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []

    found = None
    if task_id is not None:
        for t in tasks_list:
            if isinstance(t, dict) and t.get("id") == task_id:
                found = t
                break
    elif task_name:
        normalized = _normalize_task_lookup_name(task_name)
        for t in tasks_list:
            if not isinstance(t, dict):
                continue
            t_norm = _normalize_task_lookup_name(str(t.get("name", "")))
            if normalized in t_norm or t_norm in normalized:
                found = t
                break
    elif len(tasks_list) == 1:
        found = tasks_list[0]

    if found is None:
        if len(tasks_list) == 0:
            return "❌ No tasks found."
        task_list = "\n".join(f"- #{t.get('id')} {t.get('name')}" for t in tasks_list if isinstance(t, dict))
        return f"Please specify which task:\n{task_list}"

    prompt = str(found.get("prompt", "(no prompt)"))
    return (
        f"📋 Prompt of task #{found.get('id')} — {found.get('name')}\n"
        f"```\n{prompt}\n```"
    )


def _run_update_task_prompt(task_name: str, prompt: str, task_id: int | None = None) -> str:
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []
    normalized = _normalize_task_lookup_name(task_name)
    updated = None
    for task in tasks_list:
        if not isinstance(task, dict):
            continue
        if task_id is not None and task.get("id") == task_id:
            previous_prompt = str(task.get("prompt", "")).strip()
            history = task.get("prompt_history", [])
            if not isinstance(history, list):
                history = []
            if previous_prompt:
                history.append({"prompt": previous_prompt, "at": datetime.now().isoformat()})
            task["prompt_history"] = history[-10:]
            task["prompt"] = prompt.strip()
            updated = task
            break
        name = str(task.get("name", "")).lower()
        normalized_name = _normalize_task_lookup_name(name)
        if (
            name == normalized
            or normalized_name == normalized
            or normalized in normalized_name
            or normalized_name in normalized
        ):
            previous_prompt = str(task.get("prompt", "")).strip()
            history = task.get("prompt_history", [])
            if not isinstance(history, list):
                history = []
            if previous_prompt:
                history.append({"prompt": previous_prompt, "at": datetime.now().isoformat()})
            task["prompt_history"] = history[-10:]
            task["prompt"] = prompt.strip()
            updated = task
            break

    if updated is None:
        return f"Task not found for prompt update: {task_name}"

    save_scheduled_tasks(data)
    return (
        f"✅ Updated task prompt: {updated.get('name', '(unnamed)')}\n"
        f"- Prompt length: {len(str(updated.get('prompt', '')))} chars\n"
        f"- Active: {updated.get('active', False)}\n"
        f"{format_tasks_file_state()}"
    )


def _task_interval_timedelta(interval: int, unit: str) -> timedelta:
    normalized = (unit or "").strip().lower()
    if normalized == "minutes":
        return timedelta(minutes=interval)
    if normalized == "hours":
        return timedelta(hours=interval)
    if normalized == "days":
        return timedelta(days=interval)
    if normalized == "months":
        return timedelta(days=30 * interval)
    if normalized == "years":
        return timedelta(days=365 * interval)
    return timedelta(minutes=max(interval, 1))


def _parse_datetime(value: str | None) -> datetime | None:
    # Parse ISO datetime string safely; return None when parsing fails.
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _now_utc_naive() -> datetime:
    # Return UTC timestamp as naive datetime for legacy JSON compatibility.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _format_time_utc7(value: str | None) -> str:
    # Convert stored timestamp to UTC+7 display format for Discord output.
    parsed = _parse_datetime(value)
    if parsed is None:
        return "never"

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    return parsed.astimezone(UTC_PLUS_7).strftime("%Y-%m-%d %H:%M:%S")


def _is_task_due(task: dict, now: datetime) -> bool:
    # Evaluate whether one scheduled task should run at current time.
    if not isinstance(task, dict):
        return False
    if not task.get("active", True):
        return False
    interval = task.get("interval")
    unit = task.get("unit")
    if not isinstance(interval, int) or interval <= 0 or not isinstance(unit, str):
        return False

    last_run = _parse_datetime(task.get("last_run"))
    if last_run is None:
        created_at = _parse_datetime(task.get("created_at"))
        if created_at is None:
            return True
        due_time = created_at + _task_interval_timedelta(interval, unit)
        return now + timedelta(seconds=TASK_DUE_GRACE_SECONDS) >= due_time
    due_time = last_run + _task_interval_timedelta(interval, unit)
    return now + timedelta(seconds=TASK_DUE_GRACE_SECONDS) >= due_time


def _record_task_last_run(task_id: int, timestamp: str) -> None:
    # Persist `last_run` timestamp for one scheduled task.
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []
    for item in tasks_list:
        if isinstance(item, dict) and item.get("id") == task_id:
            item["last_run"] = timestamp
            break
    save_scheduled_tasks(data)


def _is_auto_learning_job_due(job: dict, now: datetime) -> bool:
    # Evaluate whether one auto-learning job is due.
    if not isinstance(job, dict) or not job.get("active", True):
        return False
    interval = job.get("interval")
    unit = job.get("unit")
    if not isinstance(interval, int) or interval <= 0 or not isinstance(unit, str):
        return False
    last_run = _parse_datetime(job.get("last_run"))
    if last_run is None:
        created_at = _parse_datetime(job.get("created_at"))
        if created_at is None:
            return True
        due_time = created_at + _task_interval_timedelta(interval, unit)
        return now + timedelta(seconds=TASK_DUE_GRACE_SECONDS) >= due_time
    due_time = last_run + _task_interval_timedelta(interval, unit)
    return now + timedelta(seconds=TASK_DUE_GRACE_SECONDS) >= due_time


def _record_auto_learning_last_run(name: str, timestamp: str) -> None:
    # Persist `last_run` timestamp for one auto-learning job.
    data = load_auto_learning_jobs()
    for job in data.get("jobs", []):
        if isinstance(job, dict) and str(job.get("name", "")).strip().lower() == name.strip().lower():
            job["last_run"] = timestamp
            break
    save_auto_learning_jobs(data)


async def _run_single_auto_learning_job(job: dict) -> dict:
    # Execute one learning cycle for a single auto-learning job.
    return await asyncio.to_thread(
        run_financial_knowledge_learning_cycle,
        genai_client,
        AI_MODEL,
        str(job.get("name") or "set_financial"),
        str(job.get("seed_question") or ""),
        str(job.get("purpose") or "SET stock financial analysis"),
    )


async def _execute_task_and_get_result(task: dict) -> str:
    prompt = str(task.get("prompt", "")).strip()
    if not prompt:
        return "(empty prompt)"

    task_name = _normalize_task_lookup_name(str(task.get("name", "")))
    if task_name == "stock_seeking":
        return await _execute_stock_seeking_task(prompt)

    # 1) Execute direct command-style prompts first (deterministic)
    cmd_match = re.match(r"^!([a-zA-Z0-9_-]+)(?:\s+([\s\S]+))?$", prompt)
    if cmd_match:
        cmd = cmd_match.group(1).lower()
        args = (cmd_match.group(2) or "").strip()

        if cmd in {"setfinancial", "setbalance"}:
            if not args:
                return "❌ Missing symbol. Usage: !setfinancial <SYMBOL>"
            try:
                data = await asyncio.to_thread(fetch_set_financial_data, args)
            except Exception as exc:
                return f"⚠️ Could not fetch SET financial data: {exc}"
            return _format_set_financial_with_analysis(data)[:1800]

        if cmd in {"price", "setprice"}:
            if not args:
                return "❌ Missing symbol. Usage: !price <SYMBOL>"
            try:
                data = await asyncio.to_thread(fetch_set_financial_data, args)
            except Exception as exc:
                return f"⚠️ Could not fetch price data: {exc}"
            return _format_set_price_snapshot(data, args)[:1500]

    # 2) For short single-command prompts only, try NL routing
    # NOTE: Never pass long AI-prompt text through handle_natural_language_tools —
    # it will match keywords inside the prompt and mutate task data as a side effect.
    # Only route if prompt is short and looks like a standalone command (no newlines).
    if len(prompt) < 120 and "\n" not in prompt:
        tool_reply = await handle_natural_language_tools(prompt, [])
        if tool_reply:
            return str(tool_reply)[:1500]

    lower = prompt.lower()
    if "hello" in lower or "สวัสดี" in lower:
        return "Hello!"

    # Multi-step execution: fetch URLs and financial data first, then ask Gemini with real data
    has_url = bool(re.search(r"https?://[^\s\"']+", prompt, re.I))
    has_setfinancial_ref = bool(re.search(r"!setfinancial\s+[A-Z0-9]{1,10}", prompt, re.I))
    if has_url or has_setfinancial_ref or (len(prompt) > 200 and "\n" in prompt):
        result = await _execute_complex_prompt(prompt, [])
        return result[:3000]

    memory = load_memory()
    answer = await ask_gemini(prompt, [], memory)
    return enforce_grounded_reply(answer, prompt)[:1500]


def _safe_score_from_metrics(chg_pct: float | None, roa: float | None, roe: float | None, div_yield: float | None) -> int:
    score, _, _ = _compute_stock_health_score(
        roa=roa,
        roe=roe,
        div_yield=div_yield,
        debt_to_equity=None,
        equity_ratio=None,
        cash_to_liabilities=None,
    )
    return score


async def _execute_stock_seeking_task(prompt: str) -> str:
    prompt_text = prompt or ""
    prompt_lower = prompt_text.lower()

    score_threshold = 70
    score_matches = re.findall(r"score\s*>\s*(\d{1,3})", prompt_text, re.I)
    if score_matches:
        try:
            score_threshold = max(0, min(100, int(score_matches[0])))
        except Exception:
            score_threshold = 70

    target_verified = 100 if re.search(r"top\s*100", prompt_text, re.I) else 25

    requires_de_screen = any(token in prompt_lower for token in ["debt/equit", "debt/equity", "d/e"])
    de_threshold = 1.5
    de_match = re.search(r"debt/equit(?:y)?\s*<\s*([0-9]+(?:\.[0-9]+)?)", prompt_text, re.I)
    if de_match:
        try:
            de_threshold = float(de_match.group(1))
        except Exception:
            de_threshold = 1.5

    reselection_trigger = None
    reselect_match = re.search(r"score\s*>\s*(\d{1,3}).{0,120}(?:re-select|reselect|ซ้ำ)", prompt_text, re.I | re.S)
    if reselect_match:
        try:
            reselection_trigger = max(0, min(100, int(reselect_match.group(1))))
        except Exception:
            reselection_trigger = None
    elif re.search(r"score\s*>\s*90", prompt_text, re.I):
        reselection_trigger = 90

    min_reselect_count = 0
    min_count_match = re.search(r"(?:at\s*least|อย่างน้อย)\s*(\d{1,3})", prompt_text, re.I)
    if min_count_match:
        try:
            min_reselect_count = max(0, min(200, int(min_count_match.group(1))))
        except Exception:
            min_reselect_count = 0

    # ── Stage 0: load Siamchart row extractor ────────────────────────────
    try:
        from khundech.web_scraping import get_siamchart_stock_rows as _scrape_rows
    except Exception as exc:
        return f"⚠️ stock_seeking failed while loading web_scraping skill: {exc}"

    # ── Stage 1: bulk-fetch up to 600 rows from Siamchart ────────────────
    try:
        siam_rows = await asyncio.to_thread(_scrape_rows, 600)
    except Exception as exc:
        return f"⚠️ stock_seeking failed while loading Siamchart data: {exc}"

    # ── Stage 2: apply Siamchart-only filter + pre-score ─────────────────
    pre_candidates: list[dict] = []
    seen_symbols: set[str] = set()

    for row in siam_rows:
        symbol = (row.get("Name") or "").strip().upper()
        if not symbol:
            continue
        try:
            symbol = normalize_stock_symbol(symbol)
        except Exception:
            continue
        if symbol in seen_symbols:
            continue

        chg_pct   = _to_float_or_none(row.get("Chg%"))
        roa       = _to_float_or_none(row.get("ROA%"))
        roe       = _to_float_or_none(row.get("ROE%"))
        div_yield = _to_float_or_none(row.get("Yield%"))
        row_de    = _to_float_or_none(row.get("D/E") or row.get("Debt/Equit") or row.get("Debt/Equity"))

        if chg_pct is None or roa is None or roe is None or div_yield is None:
            continue
        if not (chg_pct < -3 and roa > 0 and roe > 0 and div_yield > 4):
            continue
        if requires_de_screen:
            if row_de is None:
                continue
            if row_de >= de_threshold:
                continue

        pre_score = _safe_score_from_metrics(chg_pct, roa, roe, div_yield)
        seen_symbols.add(symbol)
        pre_candidates.append(
            {
                "symbol":    symbol,
                "chg_pct":   chg_pct,
                "roa":       roa,
                "roe":       roe,
                "yield_pct": div_yield,
                "pe":        (row.get("P/E")  or "N/A").strip() or "N/A",
                "pbv":       (row.get("P/BV") or "N/A").strip() or "N/A",
                "de_row":    row_de,
                "pre_score": pre_score,
            }
        )

    if not pre_candidates:
        cond_text = "(Chg% < -3, ROA > 0, ROE > 0, Yield% > 4"
        if requires_de_screen:
            cond_text += f", Debt/Equit < {de_threshold:g}"
        cond_text += ")"
        return (
            "❌ stock_seeking: no stocks matched the screening conditions "
            f"{cond_text} from current Siamchart data."
        )

    # Sort by pre-score descending so we query the best candidates first
    pre_candidates.sort(key=lambda x: x["pre_score"], reverse=True)

    # ── Stage 3: verify with SET, accumulate until 25 *confirmed* SET rows ─
    TARGET = target_verified
    results: list[dict] = []
    set_verified_count = 0  # counts only successful SET fetches

    for item in pre_candidates:
        if set_verified_count >= TARGET:
            break
        symbol = item["symbol"]
        try:
            set_data  = await asyncio.to_thread(fetch_set_financial_data, symbol)
            price_now = set_data.get("price")
            from_set  = True
            set_verified_count += 1
        except Exception:
            set_data  = None
            price_now = None
            from_set  = False

        debt_to_equity = None
        equity_ratio = None
        cash_to_liabilities = None
        if isinstance(set_data, dict):
            debt_to_equity, equity_ratio, cash_to_liabilities = _extract_balance_ratios(set_data)

        score, _, _ = _compute_stock_health_score(
            roa=item["roa"],
            roe=item["roe"],
            div_yield=item["yield_pct"],
            debt_to_equity=debt_to_equity,
            equity_ratio=equity_ratio,
            cash_to_liabilities=cash_to_liabilities,
        )
        stock_name = symbol
        one_year_hl = "N/A"
        if isinstance(set_data, dict):
            high_1y = set_data.get("high_1y") or set_data.get("high52w") or set_data.get("52w_high")
            low_1y = set_data.get("low_1y") or set_data.get("low52w") or set_data.get("52w_low")
            if high_1y is not None and low_1y is not None:
                one_year_hl = f"{low_1y}-{high_1y}"
        results.append(
            {
                "symbol":    symbol,
                "stock_name": stock_name,
                "one_year_hl": one_year_hl,
                "price_now": price_now,
                "roa":       item["roa"],
                "roe":       item["roe"],
                "pe":        item["pe"],
                "pbv":       item["pbv"],
                "yield_pct": item["yield_pct"],
                "de_ratio":  debt_to_equity,
                "de_row":    item.get("de_row"),
                "eq_ratio":  equity_ratio,
                "score":     score,
                "from_set":  from_set,
            }
        )

    # ── Stage 4: final filter — keep only score above prompt threshold ───
    total_verified = set_verified_count   # lock in count before we filter
    total_attempted = len(results)        # total candidates attempted through SET step
    results.sort(key=lambda x: x["score"], reverse=True)
    high_score = [r for r in results if r["score"] > score_threshold]

    reselection_note = None
    if reselection_trigger is not None and min_reselect_count > 0:
        trigger_count = sum(1 for r in results if r["score"] > reselection_trigger)
        if trigger_count == 0:
            if len(results) >= min_reselect_count:
                high_score = results[:min_reselect_count]
                reselection_note = (
                    f"ℹ️ Re-selected top {min_reselect_count} from Chg% < -3 group because no stock scored > {reselection_trigger}."
                )
            else:
                high_score = results
                reselection_note = (
                    f"⚠️ Re-select requested minimum {min_reselect_count}, but only {len(results)} stocks had complete verified data."
                )

    now_local = datetime.now().astimezone()

    if not high_score:
        return (
            f"📊 **Stock Seeking — Deterministic** ({now_local.strftime('%Y-%m-%d %H:%M %Z')})\n"
            f"Siamchart pool: 600 → screened {len(pre_candidates)} candidates → "
            f"SET-verified {total_verified}/{total_attempted} fetched → **0 scored > {score_threshold}**\n"
            f"No stocks passed all conditions **and** scored > {score_threshold} in this run.\n"
            "Try running again later or relax the Yield/ROA/ROE thresholds."
        )

    results = high_score

    # --- build aligned monospace table for Discord (ticker-based for clean width) ---
    cols = ["Sym", "Px", "1Y H/L", "ROA", "ROE", "PE", "PBV", "D/E", "Div%", "Score"]
    table_rows: list[list[str]] = []

    for r in results:
        price_text = str(r["price_now"]) if r["price_now"] is not None else "N/A"
        roa_text = f"{r['roa']:.2f}" if r["roa"] is not None else "N/A"
        roe_text = f"{r['roe']:.2f}" if r["roe"] is not None else "N/A"
        de_value = r["de_ratio"] if r["de_ratio"] is not None else r.get("de_row")
        de_text = f"{de_value:.2f}" if de_value is not None else "N/A"
        yld_text = f"{r['yield_pct']:.2f}" if r["yield_pct"] is not None else "N/A"
        name_text = str(r.get("symbol") or r.get("stock_name") or "N/A")
        hl_text = str(r.get("one_year_hl") or "N/A")
        table_rows.append([name_text, price_text, hl_text, roa_text, roe_text, str(r["pe"]), str(r["pbv"]), de_text, yld_text, str(r["score"])])

    widths = [len(h) for h in cols]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    table_lines = [_fmt_row(cols), "-|-".join("-" * w for w in widths)]
    table_lines.extend(_fmt_row(row) for row in table_rows)

    set_ok = sum(1 for r in results if r["from_set"])
    now_local = datetime.now().astimezone()

    output_parts = [
        f"📊 **Stock Seeking — Deterministic** ({now_local.strftime('%Y-%m-%d %H:%M %Z')})",
        (
            f"Siamchart pool: 600 → screened {len(pre_candidates)} → "
            f"SET-verified {total_verified}/{total_attempted} fetched → "
            f"showing {len(results)} (Score > {score_threshold})"
        ),
        "Price source: SET real-time (same as `!setfinancial`)",
        "N/A means that value was not available from grounded sources at run time.",
        "```",
        "\n".join(table_lines),
        "```",
        f"✅ SET price OK: {set_ok}/{len(results)} symbols",
    ]
    if reselection_note:
        output_parts.append(reselection_note)
    return "\n".join(output_parts)[:3000]


async def _run_task_by_name(task_name: str) -> str:
    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []
    normalized = _normalize_task_lookup_name(task_name)
    target = None
    for task in tasks_list:
        if not isinstance(task, dict):
            continue
        name = str(task.get("name", "")).lower()
        normalized_name = _normalize_task_lookup_name(name)
        if (
            name == normalized
            or normalized_name == normalized
            or name == task_name.strip().lower()
            or normalized in normalized_name
            or normalized_name in normalized
        ):
            target = task
            break
    if target is None:
        # Handle __FALLBACK__: show task list for user to pick from
        if task_name in {"__FALLBACK__", "FALLBACK"}:
            active = [t for t in tasks_list if isinstance(t, dict) and t.get("active", True)]
            if not active:
                return "❌ No active tasks found."
            task_list = "\n".join(f"- #{t.get('id')} {t.get('name')}" for t in active)
            return f"Multiple tasks found. Which task do you want to run?\n{task_list}"
        return f"Task not found: {task_name}"

    result = await _execute_task_and_get_result(target)
    run_at = datetime.now().isoformat()
    task_id = target.get("id")
    if isinstance(task_id, int):
        await asyncio.to_thread(_record_task_last_run, task_id, run_at)

    return (
        f"✅ Executed task: {target.get('name', '(unnamed)')}\n"
        f"Result:\n{result}\n"
        f"Recorded last_run: {run_at}"
    )


def format_last_upgrade_report() -> str:
    last = load_last_self_improvement()
    if not last:
        return "No self-upgrade record found yet."

    lines = ["🧾 Last self-upgrade (physical log)"]
    lines.append(f"- Time: {last.get('timestamp', 'unknown')}")
    lines.append(f"- Summary: {last.get('summary', 'n/a')}")
    lines.append(f"- Post apply: {last.get('post_apply', 'n/a')}")
    lines.append("- Changed files:")

    changed_files = last.get("changed_files", [])
    if not changed_files:
        lines.append("  - (none)")
    else:
        for relative in changed_files:
            status = "exists" if workspace_file_exists(relative) else "not found"
            lines.append(f"  - /app/{relative} [{status}]")

    verification = last.get("verification", [])
    if verification:
        lines.append("- Verified operations:")
        for item in verification[-10:]:
            action = item.get("action", "?")
            before_exists = item.get("before", {}).get("exists")
            after_exists = item.get("after", {}).get("exists")
            target = item.get("after", {}).get("path") or item.get("before", {}).get("path")
            lines.append(f"  - {action} {target} (before={before_exists}, after={after_exists})")

    return "\n".join(lines)


def format_actual_paths_report(max_items: int = 200) -> str:
    files = list_actual_files(max_files=max_items + 1)
    if not files:
        return "No physical files found under /app."

    manifest = load_skills_manifest()
    role_map = {item.get("path"): item.get("role", "") for item in manifest.get("files", []) if item.get("path")}

    lines = ["📂 Actual physical file paths under /app"]
    shown = files[:max_items]
    for relative in shown:
        absolute = f"/app/{relative}"
        role = role_map.get(relative)
        if role:
            lines.append(f"- {absolute} — {role}")
        else:
            lines.append(f"- {absolute}")

    if len(files) > max_items:
        lines.append(f"...and {len(files) - max_items} more files")

    data_count = sum(1 for item in files if item.startswith("data/"))
    lines.append(f"Summary: {len(files)} file(s) listed{' (truncated)' if len(files) > max_items else ''}; data files: {data_count}")
    return "\n".join(lines)


def _is_file_or_path_request(user_message: str) -> bool:
    lower = (user_message or "").lower()
    if not lower:
        return False

    explicit_markers = [
        "checkpath",
        "actualpaths",
        "hostpath",
        "full path",
        "actual path",
        "give me path",
        "give me your path",
    ]
    if any(marker in lower for marker in explicit_markers):
        return True

    if "/app/" in lower:
        return True

    if re.search(r"[a-z0-9_./-]+\.(?:py|txt|md|json|yml|yaml|toml|ini|cfg)\b", lower):
        return True

    operation_with_target = re.search(
        r"(create|make|update|edit|write|delete|remove|สร้าง|แก้|เขียน|ลบ)\s+(it|this|file|path|ไฟล์|[a-z0-9_./-]+\.(?:py|txt|md|json|yml|yaml|toml|ini|cfg))",
        lower,
    )
    if operation_with_target:
        return True

    return False


def _strip_bot_action_hallucinations(reply: str) -> str:
    # Remove lines that look like hallucinated bot action confirmations from Gemini output.
    lines = reply.splitlines()
    cleaned = []
    skip_patterns = [
        r"^\u2705 Updated task prompt",
        r"^\u2705 Task prompt updated",
        r"^Prompt length: \d+ chars",
        r"^Active: (True|False)$",
        r"^File status: /app/data/tasks\.json",
        r"^Path: /app/data/tasks\.json",
        r"^Exists: (True|False)$",
        r"^Size: \d+ bytes",
        r"^Task Manual Execution Initiated",
        r"^\(Processing Data Stream\.\.\.\)",
        r"^System Note: The automation engine",
    ]
    for line in lines:
        if any(re.match(p, line.strip()) for p in skip_patterns):
            continue
        cleaned.append(line)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    return "\n".join(cleaned)


async def _execute_complex_prompt(prompt: str, history: list) -> str:
    # Execute a complex multi-step prompt: fetch URLs, gather !setfinancial data, then ask Gemini.
    gathered = []

    # 1. Fetch all web URLs mentioned in the prompt
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s\"'<>]+", prompt, re.I)))
    for url in urls[:3]:
        try:
            fetch_fn = _load_web_scraping_fetcher(wants_raw=False)
            content = await asyncio.to_thread(fetch_fn, url)
            gathered.append(f"=== Data scraped from {url} ({len(content)} chars) ===\n{content[:3000]}")
        except Exception as exc:
            gathered.append(
                f"=== Could not fetch {url}: {exc}. "
                "If the page requires browser-side JavaScript or waiting for dynamic DOM updates, "
                "the current scraper may not have enough capability to collect that table. ==="
            )

    # 2. Fetch financial data for any !setfinancial SYMBOL references
    fin_symbols = list(
        dict.fromkeys(
            re.findall(
                r"(?:!|\b)i?set(?:financial|balance)\s+([A-Z0-9._-]{1,10})|\bset\s+(?:financial|balance)\s+([A-Z0-9._-]{1,10})",
                prompt,
                re.I,
            )
        )
    )
    normalized_symbols = []
    for item in fin_symbols:
        if isinstance(item, tuple):
            candidate = next((part for part in item if part), "")
        else:
            candidate = item
        candidate = re.sub(r"[^A-Za-z0-9._-]", "", str(candidate)).upper()
        if candidate and candidate not in normalized_symbols:
            normalized_symbols.append(candidate)
    for symbol in normalized_symbols[:8]:
        if symbol in {"YOU", "YOUR", "YOURSELF", "ME", "MY", "FOR", "THE", "THIS", "THAT"}:
            continue
        try:
            data = await asyncio.to_thread(fetch_set_financial_data, symbol.upper())
            summary = _format_set_financial_with_analysis(data)
            gathered.append(f"=== Financial data: {symbol.upper()} ===\n{summary[:1500]}")
        except Exception as exc:
            gathered.append(f"=== Could not get {symbol} data: {exc} ===")

    if gathered:
        real_data = "\n\n".join(gathered)
        augmented = (
            f"{prompt}\n\n"
            f"=== REAL-TIME DATA COLLECTED (analyse using ONLY this data, do NOT make up prices or metrics) ===\n"
            f"{real_data}\n"
            f"=== END OF REAL-TIME DATA ===\n\n"
            "When real data is missing or a site could not be fetched, state that clearly. "
            "Do NOT invent or hallucinate any stock prices, financial metrics, or other values."
        )
    else:
        augmented = prompt

    memory = load_memory()
    raw_answer = await ask_gemini(augmented, history, memory)
    answer = enforce_grounded_reply(raw_answer, prompt)
    return _strip_bot_action_hallucinations(answer)


def enforce_grounded_reply(reply: str, user_message: str = "") -> str:
    text = (reply or "").strip()
    if not text:
        return "I don't have a response right now."

    action_markers = [
        "i created",
        "i updated",
        "i wrote",
        "i deleted",
        "i added",
        "i installed",
        "ผมได้สร้าง",
        "ผมได้เขียน",
        "ผมได้อัปเดต",
        "ผมได้ลบ",
        "สร้างไฟล์",
        "อัปเดตไฟล์",
        "ลบไฟล์",
    ]

    lower = text.lower()
    path_matches = re.findall(r"(?:/app/)?[A-Za-z0-9_./-]+\.(?:py|txt|md|json|yml|yaml|toml|ini|cfg)", text)
    has_action_claim = any(marker in lower for marker in action_markers)
    has_path_list_claim = (
        "full path" in lower
        or "actual path" in lower
        or "รายการ" in lower and "path" in lower
        or "/app/" in text
    )

    if not path_matches:
        return text
    if not has_action_claim and not has_path_list_claim:
        return text
    if not _is_file_or_path_request(user_message):
        return text

    normalized = []
    for match in path_matches:
        candidate = match.replace("\\", "/")
        if candidate.startswith("/app/"):
            candidate = candidate[5:]
        normalized.append(candidate)

    unique_paths = list(dict.fromkeys(normalized))
    missing = [path for path in unique_paths if not workspace_file_exists(path)]
    if not missing:
        return text

    existing = [path for path in unique_paths if workspace_file_exists(path)]
    lines = ["I can’t confirm those file paths physically yet."]
    lines.append("Missing path(s): " + ", ".join(f"/app/{path}" for path in missing))
    if existing:
        lines.append("Verified existing path(s): " + ", ".join(f"/app/{path}" for path in existing))
    lines.append("Use !checkpath <path>, !actualpaths, or !hostpath to verify real file state.")
    return "\n".join(lines)


async def _classify_task_intent_via_ai(text: str, tasks: list) -> dict | None:
    # Use Gemini as a lightweight structured intent classifier for task commands.
    # Only called when task-related keywords are detected but no deterministic route matched.
    # Returns a dict with action and params, or None if not a task command.
    if not tasks:
        tasks_summary = "(no tasks currently exist)"
    else:
        tasks_summary = "\n".join(
            f"ID:{t.get('id')} name:{t.get('name')} prompt_preview:{str(t.get('prompt',''))[:80]}"
            for t in tasks if isinstance(t, dict)
        )

    system = (
        "You are a task intent classifier. Given a user message and task list, "
        "return ONLY a single-line JSON object — no markdown, no explanation.\n"
        "Possible actions: show_prompt, update_prompt, delete_task, create_task, list_tasks, run_task, none\n"
        "Fields: action (string), task_id (int or null), task_name (string or null), new_prompt (string or null)\n"
        "Examples:\n"
        '"show prompt of task 2" -> {"action":"show_prompt","task_id":2,"task_name":null,"new_prompt":null}\n'
        '"change prompt of check price to use !setfinancial CPF" -> {"action":"update_prompt","task_id":null,"task_name":"check_price_of_stock_cpf","new_prompt":"!setfinancial CPF"}\n'
        '"delete task 1" -> {"action":"delete_task","task_id":1,"task_name":null,"new_prompt":null}\n'
        '"run stock seeking now" -> {"action":"run_task","task_id":null,"task_name":"stock_seeking","new_prompt":null}\n'
        "If not clearly task-related, return: {\"action\":\"none\"}"
    )
    prompt = f"Tasks:\n{tasks_summary}\n\nUser message: {text}"
    try:
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model=AI_MODEL,
            contents=f"{system}\n\n{prompt}",
        )
        raw = (response.text or "").strip()
        json_match = re.search(r"\{[^{}]+\}", raw)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    return None


async def handle_natural_language_tools(message_text: str, history: list | None = None) -> str | None:
    text = message_text.strip()
    lower = text.lower()
    history = history or []

    if not text:
        return None

    brief_chat_reply = _extract_brief_chat_reply(text)
    if brief_chat_reply is not None:
        return brief_chat_reply

    direct_intent = _extract_file_intent(text)
    if direct_intent:
        return await asyncio.to_thread(_run_file_intent, direct_intent)

    followup_intent = _resolve_followup_file_intent(text, history)
    if followup_intent:
        return await asyncio.to_thread(_run_file_intent, followup_intent)

    skill_add_intent = _extract_skill_add_intent(text)
    if skill_add_intent:
        return await asyncio.to_thread(_run_add_skill_intent, skill_add_intent)

    if _extract_skill_status_intent(text):
        return await asyncio.to_thread(_run_skill_status_intent)

    if _extract_skill_usage_intent(text):
        return await asyncio.to_thread(_run_skill_usage_intent)

    auto_learning_add = parse_auto_learning_add_intent(text)
    if auto_learning_add:
        return await asyncio.to_thread(_run_add_auto_learning_intent, auto_learning_add)

    auto_learning_update = parse_auto_learning_update_intent(text)
    if auto_learning_update:
        return await asyncio.to_thread(_run_update_auto_learning_intent, auto_learning_update)

    auto_learning_delete = parse_auto_learning_delete_intent(text)
    if auto_learning_delete:
        return await asyncio.to_thread(_run_delete_auto_learning_intent, auto_learning_delete)

    if is_auto_learning_list_intent(text):
        return await asyncio.to_thread(_run_list_auto_learning_intent)

    if _extract_pretty_table_output_intent(text):
        return await asyncio.to_thread(_run_pretty_table_output_intent)

    if _extract_move_web_scraping_intent(text):
        return await asyncio.to_thread(_run_move_web_scraping_to_khundech)

    if _extract_web_scraping_where_intent(text):
        return await asyncio.to_thread(_run_web_scraping_where)

    if _extract_task_list_intent(text):
        return await asyncio.to_thread(_run_list_tasks_intent)

    if _extract_tasks_raw_intent(text):
        return await asyncio.to_thread(_run_tasks_raw_intent)

    raw_file_view_path = _extract_raw_file_view_intent(text)
    if raw_file_view_path:
        return await asyncio.to_thread(_run_raw_file_view_intent, raw_file_view_path)

    web_scraping_use_intent = _extract_web_scraping_use_intent(text)
    if web_scraping_use_intent:
        return await asyncio.to_thread(_run_web_scraping_fetch_intent, web_scraping_use_intent)

    followup_web_scraping_intent = _resolve_followup_web_scraping_intent(text, history)
    if followup_web_scraping_intent:
        return await asyncio.to_thread(_run_web_scraping_fetch_intent, followup_web_scraping_intent)

    task_run_name = _extract_task_run_intent(text)
    if task_run_name:
        return await _run_task_by_name(task_run_name)

    # Check task creation BEFORE prompt update to avoid false detection
    simple_task_intent = _extract_simple_task_create_intent(text)
    if simple_task_intent:
        task = await asyncio.to_thread(
            add_scheduled_task,
            simple_task_intent["name"],
            simple_task_intent["prompt"],
            simple_task_intent["interval"],
            simple_task_intent["unit"],
        )
        return (
            f"✅ Scheduled task added: {task['name']}\n"
            f"- Task ID: {task['id']}\n"
            f"- Interval: every {task['interval']} {task['unit']}\n"
            f"- Prompt: {task['prompt']}\n"
            f"- Active: {task['active']}\n"
            f"{format_tasks_file_state()}"
        )

    # Update task schedule (e.g., "change task 2 to every 1 hour")
    task_schedule_update = _extract_task_schedule_update_intent(text)
    if task_schedule_update:
        return await asyncio.to_thread(
            _run_update_task_schedule,
            task_schedule_update.get("task_id"),
            task_schedule_update.get("task_name"),
            task_schedule_update["interval"],
            task_schedule_update["unit"],
        )

    # Recover/rollback task prompt to previous snapshot
    task_prompt_recovery = _extract_task_prompt_recovery_intent(text)
    if task_prompt_recovery:
        return await asyncio.to_thread(
            _run_recover_task_prompt,
            task_prompt_recovery.get("task_id"),
            task_prompt_recovery.get("task_name"),
        )

    # Direct execution: "can you try this to run this <prompt>" - must be BEFORE task_prompt_update
    # to prevent pasted prompt text from being stored as a task update
    direct_run_prompt = _extract_direct_run_intent(text)
    if direct_run_prompt:
        return await _execute_complex_prompt(direct_run_prompt, history)

    task_prompt_update = _extract_task_prompt_update_intent(text)
    if task_prompt_update:
        return await asyncio.to_thread(
            _run_update_task_prompt,
            task_prompt_update["task_name"],
            task_prompt_update["prompt"],
            task_prompt_update.get("task_id"),
        )

    if _looks_like_task_prompt_update_request(text):
        return (
            "Please include the new prompt text, for example:\n"
            "- update prompt of the task to \"new prompt text\"\n"
            "- stock seeking task update prompt of the task to \"new prompt text\"\n"
            "- update prompt of the task to #1 stock seeking :: new prompt text"
        )

    task_remove_name = _extract_task_remove_intent(text)
    if task_remove_name:
        # Handle fallback: remove single task when name not specified
        if task_remove_name == "__FALLBACK__":
            tasks_data = load_scheduled_tasks()
            tasks_list = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else []
            if len(tasks_list) == 1:
                task_name = tasks_list[0].get("name", "unknown")
                return await asyncio.to_thread(_run_remove_task_by_name, task_name)
            elif len(tasks_list) == 0:
                return "❌ No tasks found to remove."
            else:
                return f"Multiple tasks found. Please specify which one to remove:\n" + "\n".join(
                    f"- {t.get('name')}" for t in tasks_list if isinstance(t, dict)
                )
        return await asyncio.to_thread(_run_remove_task_by_name, task_remove_name)

    # Show task prompt (read-only, no mutation)
    show_prompt_intent = _extract_task_show_prompt_intent(text)
    if show_prompt_intent:
        return await asyncio.to_thread(
            _run_show_task_prompt,
            show_prompt_intent["task_id"],
            show_prompt_intent["task_name"],
        )

    # AI-powered fallback: classify unmatched task-related natural language
    # Guard: skip classifier for very long messages that look like prompt content, not commands
    task_nl_keywords = ["task", "prompt", "งาน", "schedule", "automation", "recap", "remind"]
    _is_long_content_message = len(text) > 300 and text.count("\n") > 3
    if any(kw in lower for kw in task_nl_keywords) and not _is_long_content_message:
        _data = load_scheduled_tasks()
        _all_tasks = _data.get("tasks", []) if isinstance(_data, dict) else []
        if _all_tasks:
            ai_intent = await _classify_task_intent_via_ai(text, _all_tasks)
            if ai_intent and ai_intent.get("action") not in {None, "none"}:
                action = ai_intent["action"]
                # Safety guard: never auto-apply update_prompt with a massive block
                # (prevents AI classifier from treating pasted prompt text as a task update)
                if action == "update_prompt" and len(ai_intent.get("new_prompt") or "") > 400:
                    action = "none"
                    ai_intent["action"] = "none"
                t_id = ai_intent.get("task_id")
                t_name = ai_intent.get("task_name") or ""
                new_prompt = ai_intent.get("new_prompt") or ""
                if action == "show_prompt":
                    return await asyncio.to_thread(_run_show_task_prompt, t_id, t_name or None)
                elif action == "update_prompt" and new_prompt:
                    if t_id and not t_name:
                        for _t in _all_tasks:
                            if isinstance(_t, dict) and _t.get("id") == t_id:
                                t_name = str(_t.get("name", ""))
                                break
                    if t_name:
                        return await asyncio.to_thread(_run_update_task_prompt, t_name, new_prompt, t_id or None)
                elif action == "delete_task":
                    if t_id and not t_name:
                        for _t in _all_tasks:
                            if isinstance(_t, dict) and _t.get("id") == t_id:
                                t_name = str(_t.get("name", ""))
                                break
                    if t_name:
                        return await asyncio.to_thread(_run_remove_task_by_name, t_name)
                elif action == "run_task":
                    if not t_name and not t_id:
                        # Can't determine which task — list them
                        active = [t for t in _all_tasks if isinstance(t, dict) and t.get("active", True)]
                        if len(active) == 1:
                            t_name = str(active[0].get("name", ""))
                        else:
                            task_list = "\n".join(f"- #{t.get('id')} {t.get('name')}" for t in active)
                            return f"Which task do you want to run?\n{task_list}"
                    if t_id and not t_name:
                        for _t in _all_tasks:
                            if isinstance(_t, dict) and _t.get("id") == t_id:
                                t_name = str(_t.get("name", ""))
                                break
                    if t_name:
                        return await _run_task_by_name(t_name)
                elif action == "list_tasks":
                    return await asyncio.to_thread(_run_list_tasks_intent)
                elif action == "create_task":
                    pass  # fall through to parse_task_add_intent

    task_add_intent = parse_task_add_intent(text)
    if task_add_intent:
        task = await asyncio.to_thread(
            add_scheduled_task,
            task_add_intent["name"],
            task_add_intent["prompt"],
            task_add_intent["interval"],
            task_add_intent["unit"],
        )
        return (
            f"✅ Scheduled task added: {task['name']}\n"
            f"- Task ID: {task['id']}\n"
            f"- Interval: every {task['interval']} {task['unit']}\n"
            f"- Prompt length: {len(task['prompt'])} chars\n"
            f"- Active: {task['active']}\n"
            f"{format_tasks_file_state()}"
        )

    upgrade_followup_goal = _extract_upgrade_followup_goal(text, history)
    if upgrade_followup_goal:
        return await _run_self_upgrade_goal(upgrade_followup_goal)

    if _looks_like_self_upgrade_goal(text):
        return await _run_self_upgrade_goal(text)

    # Natural-language self-restart via Docker
    restart_markers = [
        "restart yourself", "restart the bot", "restart bot", "reboot yourself",
        "reboot the bot", "reboot bot", "restart khundech", "restart your self",
        "รีสตาร์ทตัวเอง", "รีสตาร์ท", "รีบูต",
    ]
    if any(marker in lower for marker in restart_markers):
        try:
            status = await asyncio.to_thread(restart_self)
            return f"✅ {status}\nKhunDech will be back in a moment."
        except Exception as exc:
            return (
                f"⚠️ Docker restart unavailable: {exc}\n"
                "Use `!restart` to exit and let Docker restart via policy."
            )

    path_list_keywords = [
        "give me path",
        "give me your path",
        "giev me",
        "full path",
        "actual path",
        "จริง path",
        "path จริง",
        "path ทั้งหมด",
        "your path again",
        "all path",
    ]
    broad_path_intent = (
        "path" in lower
        and any(token in lower for token in ["give", "giev", "show", "list", "again", "your", "actual", "full"])
    )
    if any(keyword in lower for keyword in path_list_keywords) or broad_path_intent:
        return format_actual_paths_report()

    host_path_keywords = [
        "my side",
        "host path",
        "local path",
        "pc path",
        "path for my side",
        "ฝั่งเครื่อง",
        "ฝั่ง pc",
    ]
    if any(keyword in lower for keyword in host_path_keywords):
        return await asyncio.to_thread(format_host_path_report)

    sync_keywords = ["sync", "linked", "bind mount", "อัปเดตถึงกัน", "sync probe"]
    if any(keyword in lower for keyword in sync_keywords) and any(
        keyword in lower for keyword in ["check", "test", "verify", "probe", "ตรวจ"]
    ):
        return await asyncio.to_thread(run_sync_probe)

    if lower.startswith("checkpath ") or lower.startswith("pathcheck "):
        target = text.split(" ", 1)[1].strip()
        return format_path_state(target)

    path_query_keywords = ["check path", "path file", "ตรวจ path", "เช็คไฟล์", "check file path"]
    if any(keyword in lower for keyword in path_query_keywords):
        match = re.search(r"(?:/app/)?[a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+", text)
        if match:
            candidate = match.group(0).replace("/app/", "")
            return format_path_state(candidate)

    upgrade_keywords = ["อัพเกรด", "upgrade", "self improve", "self-improve"]
    location_keywords = ["ตรงไหน", "ที่ไหน", "หาไม่เจอ", "where", "not found", "last", "log", "record"]
    if any(keyword in lower for keyword in upgrade_keywords) and any(keyword in lower for keyword in location_keywords):
        return format_last_upgrade_report()

    if lower.startswith("docker "):
        args = text[7:].strip() or "ps -a"
        return await asyncio.to_thread(run_docker_command, args, False)

    if lower.startswith("compose "):
        args = text[8:].strip() or "ps"
        return await asyncio.to_thread(run_docker_command, args, True)

    runtime_keywords = [
        "runtime",
        "run on docker",
        "running on docker",
        "docker socket",
        "where are you running",
    ]
    if any(keyword in lower for keyword in runtime_keywords):
        return format_runtime_facts(get_runtime_facts())

    if "container" in lower and any(keyword in lower for keyword in ["show", "list", "status", "running"]):
        return await asyncio.to_thread(run_docker_command, "ps -a", False)

    if "compose" in lower and any(keyword in lower for keyword in ["show", "list", "status", "service"]):
        return await asyncio.to_thread(run_docker_command, "ps", True)

    if _extract_time_now_intent(text):
        now_bkk = datetime.utcnow() + timedelta(hours=7)
        return f"🕒 Current time: {now_bkk.strftime('%Y-%m-%d %H:%M:%S')} UTC+7 (Bangkok)"

    if _extract_timezone_intent(text):
        return "🌐 Runtime timezone: UTC+7 (Bangkok / Asia/Bangkok)"

    if _extract_auto_command_preference_intent(text):
        return (
            "✅ Understood. I will auto-match your natural language to executable tools when a verified route exists.\n"
            "I will not claim execution unless the action is physically or programmatically verified."
        )

    finance_symbol = _extract_set_financial_intent(text)
    if finance_symbol:
        data = await asyncio.to_thread(fetch_set_financial_data, finance_symbol)
        return _format_set_financial_with_analysis(data)

    price_symbol = _extract_set_price_intent(text)
    if price_symbol:
        data = await asyncio.to_thread(fetch_set_financial_data, price_symbol)
        return _format_set_price_snapshot(data, price_symbol)

    finance_keywords = [
        "set financial",
        "financial statement",
        "company highlights",
        "balance sheet",
        "stock financial",
    ]
    if any(keyword in lower for keyword in finance_keywords):
        symbol = _extract_stock_symbol(text)
        if not symbol:
            return "Please include a stock symbol, for example: TISCO"
        data = await asyncio.to_thread(fetch_set_financial_data, symbol)
        return _format_set_financial_with_analysis(data)

    return None


def build_system_instruction(memory: dict) -> str:
    facts_block = ""
    if memory.get("learned_facts"):
        facts_block = "\n\nKnown facts:\n" + "\n".join(
            f"- {fact}" for fact in memory["learned_facts"][-30:]
        )

    manifest_context = build_skills_context(load_skills_manifest())
    runtime_context = format_runtime_facts(get_runtime_facts())
    return (
        f"{BASE_SYSTEM_INSTRUCTION}\n\n"
        f"Runtime state:\n{runtime_context}\n\n"
        f"Project skills and file map:\n{manifest_context}"
        f"{facts_block}"
    )


async def ask_gemini(user_message: str, history: list, memory: dict) -> str:
    system_prompt = build_system_instruction(memory)
    knowledge_context = build_knowledge_research_context(user_message)

    turns = []
    for turn in history[-20:]:
        turns.append(f"User: {turn['user']}")
        turns.append(f"Assistant: {turn['assistant']}")

    history_block = "\n".join(turns) if turns else "(no previous conversation)"
    prompt = (
        f"{system_prompt}\n\n"
        f"{knowledge_context}\n\n" if knowledge_context else f"{system_prompt}\n\n"
    ) + (
        "Conversation history:\n"
        f"{history_block}\n\n"
        f"User: {user_message}\n"
        "Assistant:"
    )

    response = await asyncio.to_thread(
        genai_client.models.generate_content,
        model=AI_MODEL,
        contents=prompt,
    )
    return response.text or "I don't have a response right now."


@bot.event
async def on_ready():
    print(f"[KhunDech] Online as {bot.user}  (id={bot.user.id})")
    print(f"[KhunDech] AI model : {AI_MODEL}")
    print(f"[KhunDech] Data dir : {DATA_DIR}")

    channel = bot.get_channel(MAIN_CHANNEL_ID)
    if channel:
        try:
            await channel.send("i am online now")
        except Exception as exc:
            print(f"[KhunDech] Online message error: {exc}")

    if not daily_report.is_running():
        daily_report.start()
    if not task_runner.is_running():
        task_runner.start()
    ensure_default_auto_learning_job()
    if not knowledge_learning_runner.is_running():
        knowledge_learning_runner.start()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.author.id != ALLOWED_USER_ID:
        return

    raw_content = message.content or ""
    if raw_content.strip().startswith("!"):
        await bot.process_commands(message)
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions
    is_reply_to_bot = (
        message.reference is not None
        and message.reference.resolved is not None
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author.id == bot.user.id
    )

    if not (is_dm or is_mentioned or is_reply_to_bot or raw_content.strip()):
        await bot.process_commands(message)
        return

    content = raw_content.replace(f"<@{bot.user.id}>", "")
    content = content.replace(f"<@!{bot.user.id}>", "").strip() or "Hello!"

    history = load_history(message.author.id)
    memory = load_memory()

    try:
        # First, try deterministic tools without typing indicator
        tool_reply = await handle_natural_language_tools(content, history)
        if tool_reply is not None:
            reply = tool_reply
        else:
            # Only show typing if we need to call AI
            async with message.channel.typing():
                if _looks_like_operational_request(content):
                    reply = (
                        "I couldn't execute that operational request with a verified tool path yet.\n"
                        "Please use a concrete format, for example:\n"
                        "- add task name \"stock seeking\" promt \"...\" every 6 hours\n"
                        "- remove task say_hello\n"
                        "- add skill name web_scraping with required parameter url\n"
                        "- move web_scraping to /app/khundech/\n"
                        "- checkpath khundech/web_scraping.py"
                    )
                else:
                    reply = enforce_grounded_reply(await ask_gemini(content, history, memory), content)
    except Exception as exc:
        await message.reply(f"⚠️ Error talking to AI: {exc}")
        return

    history.append(
        {
            "user": content,
            "assistant": reply,
            "timestamp": datetime.now().isoformat(),
        }
    )
    save_history(message.author.id, history)

    memory["total_messages"] = memory.get("total_messages", 0) + 1
    save_memory(memory)

    for chunk in [reply[i:i + 1990] for i in range(0, len(reply), 1990)]:
        await message.reply(chunk)

    await bot.process_commands(message)


@tasks.loop(hours=24)
async def daily_report():
    channel = bot.get_channel(CRON_CHANNEL_ID)
    if not channel:
        print(f"[KhunDech] Cron channel {CRON_CHANNEL_ID} not found.")
        return

    memory = load_memory()
    runtime = get_runtime_facts()
    report = (
        "📋 **Daily Report**\n\n"
        f"Date: {datetime.now().strftime('%A, %B %d, %Y')}\n"
        f"Total messages processed: {memory.get('total_messages', 0)}\n"
        f"Learned facts count: {len(memory.get('learned_facts', []))}\n"
        f"Running in Docker: {runtime['in_docker']}\n"
        f"Docker socket mounted: {runtime['docker_socket']}\n"
        "Source: physical files and runtime checks only."
    )
    await channel.send(report)


@tasks.loop(seconds=TASK_RUNNER_INTERVAL_SECONDS)
async def task_runner():
    # Background loop that executes due scheduled tasks and posts results.
    channel = bot.get_channel(CRON_CHANNEL_ID)
    if not channel:
        return

    data = load_scheduled_tasks()
    tasks_list = data.get("tasks", []) if isinstance(data, dict) else []
    for task in tasks_list:
        now = _now_utc_naive()
        if not _is_task_due(task, now):
            continue
        task_name = str(task.get("name", "(unnamed)"))
        task_id = task.get("id")
        try:
            result = await _execute_task_and_get_result(task)
            run_at = _now_utc_naive().isoformat()
            if isinstance(task_id, int):
                await asyncio.to_thread(_record_task_last_run, task_id, run_at)

            message = (
                f"⏱️ Task run: {task_name}\n"
                f"Schedule: every {task.get('interval')} {task.get('unit')}\n"
                f"Result:\n{result}\n"
                f"Recorded last_run: {run_at}"
            )
            for chunk in [message[i:i + 1990] for i in range(0, len(message), 1990)]:
                await channel.send(chunk)
        except Exception as exc:
            await channel.send(f"⚠️ Task run failed: {task_name} | {exc}")


@tasks.loop(seconds=AUTO_LEARNING_RUNNER_INTERVAL_SECONDS)
async def knowledge_learning_runner():
    # Background loop that executes due auto-learning jobs.
    jobs = list_auto_learning_jobs()
    if not jobs:
        return
    now = _now_utc_naive()
    for job in jobs:
        if not _is_auto_learning_job_due(job, now):
            continue
        try:
            result = await _run_single_auto_learning_job(job)
            await asyncio.to_thread(_record_auto_learning_last_run, str(job.get("name")), _now_utc_naive().isoformat())
        except Exception as exc:
            print(f"[KhunDech] Financial knowledge learning failed for {job.get('name')}: {exc}")
            continue

        memory = load_memory()
        memory.setdefault("learned_facts", []).append(
            f"[{datetime.now().strftime('%Y-%m-%d')}] Financial knowledge job {job.get('name')}: Q={result.get('question')}"
        )
        memory["learned_facts"] = memory["learned_facts"][-500:]
        save_memory(memory)

        channel = bot.get_channel(LEARNING_CHANNEL_ID)
        if not channel:
            continue

        msg = (
            "🧠 Financial knowledge learned\n"
            f"- Job: {result.get('job_name')}\n"
            f"- Question: {result.get('question')}\n"
            f"- Next question: {result.get('next_question')}\n"
            f"- Total knowledge entries: {result.get('total_entries')}"
        )
        for chunk in [msg[i:i + 1990] for i in range(0, len(msg), 1990)]:
            await channel.send(chunk)


@daily_report.before_loop
async def before_daily_report():
    await bot.wait_until_ready()
    await asyncio.sleep(24 * 60 * 60)


@task_runner.before_loop
async def before_task_runner():
    await bot.wait_until_ready()


@knowledge_learning_runner.before_loop
async def before_knowledge_learning_runner():
    await bot.wait_until_ready()
    ensure_default_auto_learning_job()


@bot.command(name="learn")
async def cmd_learn(ctx: commands.Context, *, fact: str):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    memory = load_memory()
    memory.setdefault("learned_facts", []).append(f"[{datetime.now().strftime('%Y-%m-%d')}] {fact}")
    save_memory(memory)
    await ctx.send(f"✅ Learned: *{fact}*")


@bot.command(name="forget")
async def cmd_forget(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    save_history(ctx.author.id, [])
    await ctx.send("🗑️ Conversation history cleared.")


@bot.command(name="memory")
async def cmd_memory(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    memory = load_memory()
    history = load_history(ctx.author.id)
    embed = discord.Embed(title="🧠 KhunDech Memory", color=0x5865F2)
    embed.add_field(name="Total Messages", value=str(memory.get("total_messages", 0)), inline=True)
    embed.add_field(name="Learned Facts", value=str(len(memory.get("learned_facts", []))), inline=True)
    embed.add_field(name="Chat History", value=f"{len(history)} turns", inline=True)
    embed.set_footer(text=f"Online since {memory.get('first_seen', 'unknown')[:10]}")
    await ctx.send(embed=embed)


@bot.command(name="klearn")
async def cmd_klearn(ctx: commands.Context, *, job_name: str = "set_financial"):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    jobs = list_auto_learning_jobs()
    target = None
    for job in jobs:
        if isinstance(job, dict) and str(job.get("name", "")).strip().lower() == job_name.strip().lower():
            target = job
            break
    if target is None and job_name.strip().lower() == "set_financial":
        target = ensure_default_auto_learning_job()
    if target is None:
        await ctx.send(f"⚠️ Auto-learning job not found: {job_name}")
        return
    async with ctx.typing():
        try:
            result = await _run_single_auto_learning_job(target)
        except Exception as exc:
            await ctx.send(f"⚠️ Knowledge learning failed: {exc}")
            return

    memory = load_memory()
    memory.setdefault("learned_facts", []).append(
        f"[{datetime.now().strftime('%Y-%m-%d')}] Financial knowledge job {result.get('job_name')}: Q={result.get('question')}"
    )
    memory["learned_facts"] = memory["learned_facts"][-500:]
    save_memory(memory)
    await asyncio.to_thread(_record_auto_learning_last_run, str(target.get("name")), _now_utc_naive().isoformat())

    await ctx.send(
        "🧠 Financial knowledge learned now\n"
        f"- Job: {result.get('job_name')}\n"
        f"- Question: {result.get('question')}\n"
        f"- Next question: {result.get('next_question')}\n"
        f"- Total knowledge entries: {result.get('total_entries')}"
    )


@bot.command(name="autolearn")
async def cmd_autolearn(ctx: commands.Context, *, args: str = ""):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    text = (args or "").strip()
    if not text:
        await ctx.send(_run_list_auto_learning_intent())
        return

    lower = text.lower()
    if lower.startswith("add "):
        intent = parse_auto_learning_add_intent(f"add auto learn {text[4:]}")
        if intent:
            await ctx.send(_run_add_auto_learning_intent(intent))
            return
    elif lower.startswith("update "):
        intent = parse_auto_learning_update_intent(f"update auto learn {text[7:]}")
        if intent:
            await ctx.send(_run_update_auto_learning_intent(intent))
            return
    elif lower.startswith("delete ") or lower.startswith("remove "):
        marker_len = 7 if lower.startswith("delete ") else 7
        name = parse_auto_learning_delete_intent(f"delete auto learn {text[marker_len:]}")
        if name:
            await ctx.send(_run_delete_auto_learning_intent(name))
            return
    elif lower in {"list", "show", "status"}:
        await ctx.send(_run_list_auto_learning_intent())
        return

    await ctx.send(
        "Usage:\n"
        "!autolearn\n"
        "!autolearn add name \"job_name\" purpose \"what this learns\" prompt \"first question\" every 10 minutes enable\n"
        "!autolearn update name \"job_name\" purpose \"new purpose\" prompt \"new first question\" every 30 minutes disable\n"
        "!autolearn delete name \"job_name\""
    )


@bot.command(name="status")
async def cmd_status(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    latency = round(bot.latency * 1000)
    await ctx.send(f"✅ KhunDech is online | model: `{AI_MODEL}` | latency: `{latency}ms`")


@bot.command(name="skills", aliases=["paths"])
async def cmd_skills(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    manifest = load_skills_manifest()
    lines = ["🧭 KhunDech Skills"]
    for skill in manifest.get("skills", []):
        lines.append(f"- {skill['name']}: {skill['description']}")
    lines.append("")
    lines.append("📁 Known Paths")
    for file_info in manifest.get("files", []):
        lines.append(f"- {file_info['path']}: {file_info['role']}")
    message = "\n".join(lines)
    for chunk in [message[i:i + 1990] for i in range(0, len(message), 1990)]:
        await ctx.send(chunk)


@bot.command(name="runtime")
async def cmd_runtime(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    output = format_runtime_facts(get_runtime_facts())
    await ctx.send(output)


@bot.command(name="docker")
async def cmd_docker(ctx: commands.Context, *, args: str = ""):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    command_args = args.strip() or "ps -a"
    output = await asyncio.to_thread(run_docker_command, command_args, False)
    for chunk in [output[i:i + 1990] for i in range(0, len(output), 1990)]:
        await ctx.send(chunk)


@bot.command(name="compose")
async def cmd_compose(ctx: commands.Context, *, args: str = ""):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    command_args = args.strip() or "ps"
    output = await asyncio.to_thread(run_docker_command, command_args, True)
    for chunk in [output[i:i + 1990] for i in range(0, len(output), 1990)]:
        await ctx.send(chunk)


@bot.command(name="hostpath", aliases=["localpath", "mypath"])
async def cmd_hostpath(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    output = await asyncio.to_thread(format_host_path_report)
    for chunk in [output[i:i + 1990] for i in range(0, len(output), 1990)]:
        await ctx.send(chunk)


@bot.command(name="synccheck", aliases=["syncprobe"])
async def cmd_synccheck(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    output = await asyncio.to_thread(run_sync_probe)
    for chunk in [output[i:i + 1990] for i in range(0, len(output), 1990)]:
        await ctx.send(chunk)


@bot.command(name="setfinancial")
async def cmd_setfinancial(ctx: commands.Context, symbol: str):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    async with ctx.typing():
        try:
            data = await asyncio.to_thread(fetch_set_financial_data, symbol)
        except Exception as exc:
            await ctx.send(f"⚠️ Could not fetch SET financial data: {exc}")
            return
    summary = _format_set_financial_with_analysis(data)
    for chunk in [summary[i:i + 1990] for i in range(0, len(summary), 1990)]:
        await ctx.send(chunk)


@bot.command(name="setbalance")
async def cmd_setbalance(ctx: commands.Context, symbol: str):
    await cmd_setfinancial(ctx, symbol)


@bot.command(name="stock")
async def cmd_stock(ctx: commands.Context, ticker: str):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    if _finance_get_stock_data is None:
        await ctx.send("⚠️ finance_module is not available. Use !setfinancial <SYMBOL> instead.")
        return
    async with ctx.typing():
        try:
            result = _finance_get_stock_data(ticker)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            await ctx.send(f"⚠️ Could not fetch stock data: {exc}")
            return
    await ctx.send(str(result)[:1900])


@bot.command(name="price", aliases=["setprice"])
async def cmd_price(ctx: commands.Context, symbol: str):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    async with ctx.typing():
        try:
            data = await asyncio.to_thread(fetch_set_financial_data, symbol)
        except Exception as exc:
            await ctx.send(f"⚠️ Could not fetch SET price data: {exc}")
            return
    await ctx.send(_format_set_price_snapshot(data, symbol))


@bot.command(name="compareprice", aliases=["pricecheck"])
async def cmd_compareprice(ctx: commands.Context, symbol: str):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    async with ctx.typing():
        try:
            result = await asyncio.to_thread(_compare_set_vs_siamchart_price, symbol)
        except Exception as exc:
            await ctx.send(f"⚠️ Could not compare prices: {exc}")
            return
    await ctx.send(result)


@bot.command(name="scrape")
async def cmd_scrape(ctx: commands.Context, url: str, *, flags: str = ""):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    async with ctx.typing():
        wants_raw = "raw" in flags.lower()
        result = await asyncio.to_thread(
            _run_web_scraping_fetch_intent, {"url": url, "raw": wants_raw}
        )
    for chunk in [result[i:i + 1990] for i in range(0, len(result), 1990)]:
        await ctx.send(chunk)


@bot.command(name="improve", aliases=["upgrade"])
async def cmd_improve(ctx: commands.Context, *, goal: str):
    if ctx.author.id != ALLOWED_USER_ID:
        return

    async with ctx.typing():
        try:
            proposal = await propose_self_improvement(genai_client, AI_MODEL, goal)
            changed_files, backup_paths, post_apply, verification = apply_self_improvement(proposal)
        except Exception as exc:
            await ctx.send(f"⚠️ Self-improvement failed: {exc}")
            return

    if not changed_files:
        await ctx.send("No code changes were produced.")
        return

    summary = proposal.get("summary", "No summary provided.")
    record_self_improvement(summary, changed_files, backup_paths, post_apply, verification)
    changed = ", ".join(changed_files)
    backups = "\n".join(backup_paths[-5:]) if backup_paths else "No backups created."
    if any(path in {"requirements.txt", "Dockerfile"} for path in changed_files):
        post_apply = "rebuild"

    if post_apply == "restart":
        await ctx.send(
            f"🛠️ Self-improvement applied to: {changed}\n"
            f"Summary: {summary}\n"
            f"Backups:\n{backups}\n"
            "Restarting now to load the new code."
        )
        await bot.close()
        return

    if post_apply == "rebuild":
        await ctx.send(
            f"🛠️ Self-improvement applied to: {changed}\n"
            f"Summary: {summary}\n"
            f"Backups:\n{backups}\n"
            "Run `docker compose up --build -d` to load dependency or container changes."
        )
        return

    await ctx.send(
        f"🛠️ Self-improvement applied to: {changed}\n"
        f"Summary: {summary}\n"
        f"Backups:\n{backups}"
    )


@bot.command(name="upgradewhere", aliases=["lastupgrade", "upgradelog"])
async def cmd_upgradewhere(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    report = format_last_upgrade_report()
    for chunk in [report[i:i + 1990] for i in range(0, len(report), 1990)]:
        await ctx.send(chunk)


@bot.command(name="checkpath", aliases=["pathcheck"])
async def cmd_checkpath(ctx: commands.Context, *, relative_path: str):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    report = format_path_state(relative_path)
    await ctx.send(report)


@bot.command(name="actualpaths", aliases=["realpaths", "fullpaths"])
async def cmd_actualpaths(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    report = format_actual_paths_report()
    for chunk in [report[i:i + 1990] for i in range(0, len(report), 1990)]:
        await ctx.send(chunk)


@bot.command(name="restart")
async def cmd_restart(ctx: commands.Context):
    if ctx.author.id != ALLOWED_USER_ID:
        return
    async with ctx.typing():
        try:
            status = await asyncio.to_thread(restart_self)
            await ctx.send(f"✅ {status}\nKhunDech will be back in a moment.")
        except Exception as exc:
            # Docker restart failed — fall back to graceful process exit
            await ctx.send(
                f"⚠️ Docker restart unavailable ({exc}).\n"
                "Falling back to process exit (Docker will restart via policy)..."
            )
            await bot.close()


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)

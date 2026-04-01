import json
import re
from datetime import datetime
from pathlib import Path

from khundech.config import DATA_DIR


KNOWLEDGE_JSON_PATH = Path(DATA_DIR) / "financial_knowledge.json"
KNOWLEDGE_MD_PATH = Path(DATA_DIR) / "financial_knowledge.md"
INITIAL_QUESTION = "How should I read Book Value and P/BV from SET financial output to judge technical rebound probability with ROA/ROE/Yield/Debt metrics?"
KNOWLEDGE_ENTRY_CAP = None  # None means infinite; keep all entries


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return slug or "knowledge"


def _store_paths(job_name: str | None = None) -> tuple[Path, Path]:
    if not job_name or job_name.strip().lower() == "set_financial":
        return KNOWLEDGE_JSON_PATH, KNOWLEDGE_MD_PATH
    slug = _slugify(job_name)
    return Path(DATA_DIR) / f"financial_knowledge_{slug}.json", Path(DATA_DIR) / f"financial_knowledge_{slug}.md"


def _load_store(job_name: str | None = None, initial_question: str | None = None) -> dict:
    json_path, _ = _store_paths(job_name)
    seed_question = initial_question or INITIAL_QUESTION
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "entries": [],
        "next_question": seed_question,
        "updated_at": datetime.now().isoformat(),
        "stats": {
            "total_learned": 0,
        },
    }


def _ensure_store_stats(store: dict) -> None:
    # Ensure stats schema exists for cumulative counters.
    stats = store.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    total_learned = stats.get("total_learned")
    if not isinstance(total_learned, int) or total_learned < 0:
        current_entries = store.get("entries", [])
        total_learned = len(current_entries) if isinstance(current_entries, list) else 0
    stats["total_learned"] = total_learned
    store["stats"] = stats


def _save_store(store: dict, job_name: str | None = None) -> None:
    json_path, _ = _store_paths(job_name)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_store_stats(store)
    store["updated_at"] = datetime.now().isoformat()
    json_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_markdown_entry(question: str, answer: str, next_question: str, job_name: str | None = None) -> None:
    _, md_path = _store_paths(job_name)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"## {ts}\n"
        f"Q: {question}\n\n"
        f"A:\n{answer}\n\n"
        f"Next question: {next_question}\n\n"
        "---\n\n"
    )
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(block)


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-zก-๙0-9]{2,}", (text or "").lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "have", "from", "your", "you",
        "ของ", "และ", "กับ", "หรือ", "ที่", "ใน", "เป็น", "ให้", "ได้", "ควร", "อะไร",
    }
    return {w for w in raw if w not in stop}


def _iter_entries(job_name: str | None = None) -> list[dict]:
    # Collect entries from one job store or all financial knowledge stores.
    entries: list[dict] = []
    if job_name:
        store = _load_store(job_name=job_name)
        current_entries = store.get("entries", []) if isinstance(store, dict) else []
        if isinstance(current_entries, list):
            entries.extend(item for item in current_entries if isinstance(item, dict))
        return entries

    json_files = sorted(Path(DATA_DIR).glob("financial_knowledge*.json"))
    for path in json_files:
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        current_entries = store.get("entries", []) if isinstance(store, dict) else []
        if isinstance(current_entries, list):
            entries.extend(item for item in current_entries if isinstance(item, dict))
    return entries


def build_knowledge_research_context_with_stats(user_message: str, max_items: int = 5, job_name: str | None = None) -> dict:
    entries = _iter_entries(job_name=job_name)
    if not entries:
        return {
            "context": "",
            "used_count": 0,
            "available_count": 0,
            "max_items": max_items,
        }

    query_tokens = _tokenize(user_message)
    ranked: list[tuple[int, dict]] = []
    for entry in entries:
        question = str(entry.get("question", ""))
        answer = str(entry.get("answer", ""))
        hay_tokens = _tokenize(question + " " + answer[:1500])
        overlap = len(query_tokens & hay_tokens)
        freshness_boost = 1 if entry.get("created_at") else 0
        score = overlap * 10 + freshness_boost
        ranked.append((score, entry))

    ranked.sort(key=lambda item: item[0], reverse=True)
    picked = [entry for score, entry in ranked[:max_items] if score > 0]
    if not picked:
        picked = [entry for _, entry in ranked[: min(max_items, len(ranked))]]

    lines = ["Relevant internal financial knowledge:"]
    used_count = 0
    for idx, entry in enumerate(picked, start=1):
        q = str(entry.get("question", "")).strip()
        a = str(entry.get("answer", "")).strip()[:700]
        if not q or not a:
            continue
        lines.append(f"[{idx}] Q: {q}")
        lines.append(f"[{idx}] A: {a}")
        used_count += 1

    context = "\n".join(lines) if used_count > 0 else ""
    return {
        "context": context,
        "used_count": used_count,
        "available_count": len(entries),
        "max_items": max_items,
    }


def build_knowledge_research_context(user_message: str, max_items: int = 5) -> str:
    payload = build_knowledge_research_context_with_stats(user_message, max_items=max_items, job_name=None)
    return str(payload.get("context") or "")


def run_financial_knowledge_learning_cycle(genai_client, model: str, job_name: str = "set_financial", seed_question: str | None = None, topic: str | None = None) -> dict:
    store = _load_store(job_name=job_name, initial_question=seed_question)
    entries = store.setdefault("entries", [])
    _ensure_store_stats(store)

    current_question = str(store.get("next_question") or seed_question or INITIAL_QUESTION).strip() or (seed_question or INITIAL_QUESTION)

    prior_notes = []
    for item in entries[-8:]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()[:500]
        if q and a:
            prior_notes.append(f"Q: {q}\nA: {a}")

    prior_block = "\n\n".join(prior_notes) if prior_notes else "(none yet)"

    learn_prompt = (
        f"You are teaching an internal assistant about {topic or 'Thailand SET stock analysis'}. "
        "Answer precisely and practically with direct use for stock screening outputs.\n\n"
        "Main learning objective:\n"
        "- Improve reading of SET-style financial fields: Price Now, 1Y High/Low, ROA, ROE, P/E, P/BV, Dividend %, Book Value, Debt/Equity.\n"
        "- Focus on book value/PBV interpretation and rebound probability logic.\n"
        "- Produce rules that can be applied directly in deterministic screening.\n\n"
        f"Existing learned knowledge:\n{prior_block}\n\n"
        f"Current learning question: {current_question}\n\n"
        "Return a concise but high-value learning note in plain text. Include keywords: book value, P/BV, ROA, ROE, Yield, Debt/Equity, rebound."
    )

    answer_response = genai_client.models.generate_content(model=model, contents=learn_prompt)
    answer_text = (answer_response.text or "").strip()
    if not answer_text:
        answer_text = "No knowledge content returned."

    next_prompt = (
        "Based on the learned knowledge below, propose exactly ONE next question to continue learning "
        "about SET book value reading and rebound prediction from setfinancial-style output. "
        "Keep it short and specific. Return only the question.\n\n"
        f"Current question: {current_question}\n"
        f"Learned answer: {answer_text[:2000]}"
    )
    next_response = genai_client.models.generate_content(model=model, contents=next_prompt)
    next_question = (next_response.text or "").strip().splitlines()[0].strip()
    if not next_question:
        next_question = f"What should I learn next about {topic or 'SET stock analysis'} after this?"

    entry = {
        "created_at": datetime.now().isoformat(),
        "question": current_question,
        "answer": answer_text,
        "answer_chars": len(answer_text),
    }
    entries.append(entry)

    # Deduplicate by normalized question and keep a bounded recent window
    # to avoid unbounded growth with repeated/near-identical learning loops.
    dedup_map: dict[str, dict] = {}
    for item in reversed(entries):
        if not isinstance(item, dict):
            continue
        question_key = re.sub(r"\s+", " ", str(item.get("question", "")).strip().lower())
        if not question_key:
            continue
        if question_key in dedup_map:
            continue
        dedup_map[question_key] = item

    deduped_entries = list(reversed(list(dedup_map.values())))
    if KNOWLEDGE_ENTRY_CAP is None:
        store["entries"] = deduped_entries
    else:
        cap = max(1, int(KNOWLEDGE_ENTRY_CAP))
        store["entries"] = deduped_entries[-cap:]
    stats = store.setdefault("stats", {})
    stats["total_learned"] = int(stats.get("total_learned", 0)) + 1
    store["next_question"] = next_question
    _save_store(store, job_name=job_name)
    _append_markdown_entry(current_question, answer_text, next_question, job_name=job_name)

    json_path, md_path = _store_paths(job_name)

    return {
        "job_name": job_name,
        "question": current_question,
        "answer_preview": answer_text[:220],
        "next_question": next_question,
        "total_entries": int(store.get("stats", {}).get("total_learned", len(store["entries"]))),
        "stored_entries": len(store["entries"]),
        "entry_cap": KNOWLEDGE_ENTRY_CAP,
        "knowledge_json": str(json_path),
        "knowledge_md": str(md_path),
    }

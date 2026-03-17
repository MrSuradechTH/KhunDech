import json
import re
from datetime import datetime
from pathlib import Path

from khundech.config import DATA_DIR


KNOWLEDGE_JSON_PATH = Path(DATA_DIR) / "financial_knowledge.json"
KNOWLEDGE_MD_PATH = Path(DATA_DIR) / "financial_knowledge.md"
INITIAL_QUESTION = "what knowledge i shoulde be know about book value of set stock"


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
    return {"entries": [], "next_question": seed_question, "updated_at": datetime.now().isoformat()}


def _save_store(store: dict, job_name: str | None = None) -> None:
    json_path, _ = _store_paths(job_name)
    json_path.parent.mkdir(parents=True, exist_ok=True)
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


def build_knowledge_research_context(user_message: str, max_items: int = 5) -> str:
    json_files = sorted(Path(DATA_DIR).glob("financial_knowledge*.json"))
    entries = []
    for path in json_files:
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        current_entries = store.get("entries", []) if isinstance(store, dict) else []
        if isinstance(current_entries, list):
            entries.extend(current_entries)
    if not entries:
        return ""

    query_tokens = _tokenize(user_message)
    ranked: list[tuple[int, dict]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
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
    for idx, entry in enumerate(picked, start=1):
        q = str(entry.get("question", "")).strip()
        a = str(entry.get("answer", "")).strip()[:700]
        if not q or not a:
            continue
        lines.append(f"[{idx}] Q: {q}")
        lines.append(f"[{idx}] A: {a}")
    return "\n".join(lines)


def run_financial_knowledge_learning_cycle(genai_client, model: str, job_name: str = "set_financial", seed_question: str | None = None, topic: str | None = None) -> dict:
    store = _load_store(job_name=job_name, initial_question=seed_question)
    entries = store.setdefault("entries", [])

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
        "Answer precisely and practically. Focus on financial reasoning, ratios, and caveats.\n\n"
        f"Existing learned knowledge:\n{prior_block}\n\n"
        f"Current learning question: {current_question}\n\n"
        "Return a concise but high-value learning note in plain text."
    )

    answer_response = genai_client.models.generate_content(model=model, contents=learn_prompt)
    answer_text = (answer_response.text or "").strip()
    if not answer_text:
        answer_text = "No knowledge content returned."

    next_prompt = (
        "Based on the learned knowledge below, propose exactly ONE next question to continue learning "
        "about SET stock financial analysis. Keep it short and specific. Return only the question.\n\n"
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
    store["entries"] = entries[-300:]
    store["next_question"] = next_question
    _save_store(store, job_name=job_name)
    _append_markdown_entry(current_question, answer_text, next_question, job_name=job_name)

    json_path, md_path = _store_paths(job_name)

    return {
        "job_name": job_name,
        "question": current_question,
        "answer_preview": answer_text[:220],
        "next_question": next_question,
        "total_entries": len(store["entries"]),
        "knowledge_json": str(json_path),
        "knowledge_md": str(md_path),
    }

import ast
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

    lines = ["\t".join(SIAMCHART_TABLE_HEADERS)]
    for row in rows:
        lines.append("\t".join(row.get(header, "") for header in SIAMCHART_TABLE_HEADERS))
    return "\n".join(lines)


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
    cleaned = re.sub(r"\s+", " ", content).strip()
    return cleaned

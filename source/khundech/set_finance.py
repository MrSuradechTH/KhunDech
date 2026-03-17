import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SET_USER_AGENT = "Mozilla/5.0 (compatible; KhunDech/1.0; +https://www.set.or.th/)"
SET_BASE_QUOTE_URL = "https://www.set.or.th/th/market/product/stock/quote/{symbol}"


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": SET_USER_AGENT})
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def normalize_stock_symbol(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", symbol.strip().upper())
    if not cleaned:
        raise ValueError("Stock symbol is required.")
    return cleaned


def split_js_arguments(text: str) -> list[str]:
    items = []
    current = []
    depth_paren = depth_brace = depth_bracket = 0
    in_string = False
    string_char = ""
    escape = False

    for char in text:
        if in_string:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            string_char = char
            current.append(char)
            continue

        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket -= 1
        elif char == "," and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            items.append("".join(current).strip())
            current = []
            continue

        current.append(char)

    if current:
        items.append("".join(current).strip())

    return items


def parse_js_literal(token: str):
    value = token.strip()
    if not value:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    if value == "void 0":
        return None
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].encode("utf-8").decode("unicode_escape")
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value


def extract_nuxt_variable_map(html: str) -> dict[str, object]:
    script_start = html.find("__NUXT__=(function(")
    if script_start == -1:
        return {}

    script_end = html.find("</script>", script_start)
    if script_end == -1:
        return {}

    script = html[script_start:script_end]
    params_start = script.find("__NUXT__=(function(") + len("__NUXT__=(function(")
    params_end = script.find("){return ", params_start)
    args_start = script.rfind("}(")
    args_end = script.rfind(");")
    if params_end == -1 or args_start == -1 or args_end == -1 or args_end <= args_start:
        return {}

    params = [item.strip() for item in script[params_start:params_end].split(",") if item.strip()]
    args = split_js_arguments(script[args_start + 2:args_end])
    mapping = {}
    for name, raw_value in zip(params, args):
        mapping[name] = parse_js_literal(raw_value)
    return mapping


def resolve_payload_value(raw_value: str, variable_map: dict[str, object]):
    value = parse_js_literal(raw_value)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", value):
        return variable_map.get(value, value)
    return value


def parse_balance_metrics_from_payload(html: str) -> list[tuple[str, float]]:
    variable_map = extract_nuxt_variable_map(html)
    metrics = []
    seen = set()
    pattern = re.compile(
        r'accountCode:"[^"]+",accountName:(?P<label>"(?:[^"\\]|\\.)*"|[A-Za-z_$][A-Za-z0-9_$]*),amount:(?P<amount>-?(?:\d+(?:\.\d+)?)|[A-Za-z_$][A-Za-z0-9_$]*)'
    )

    for match in pattern.finditer(html):
        raw_label = match.group("label")
        raw_amount = match.group("amount")
        label = resolve_payload_value(raw_label, variable_map)
        amount = resolve_payload_value(raw_amount, variable_map)
        if not isinstance(label, str):
            continue
        if not isinstance(amount, (int, float)):
            continue
        label = label.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        metrics.append((label, round(float(amount) / 1000, 2)))

    return metrics


def extract_quote_metadata(html: str) -> dict[str, object]:
    variable_map = extract_nuxt_variable_map(html)

    def extract_field(pattern: str):
        match = re.search(pattern, html)
        if not match:
            return None
        return resolve_payload_value(match.group(1), variable_map)

    return {
        "company_name": extract_field(r'nameTH:("(?:[^"\\]|\\.)*"|[A-Za-z_$][A-Za-z0-9_$]*)'),
        "price": extract_field(r'last:([A-Za-z_$][A-Za-z0-9_$]*|-?(?:\d+(?:\.\d+)?|\.\d+))'),
        "change": extract_field(r'change:([A-Za-z_$][A-Za-z0-9_$]*|-?(?:\d+(?:\.\d+)?|\.\d+))'),
        "percent_change": extract_field(r'percentChange:([A-Za-z_$][A-Za-z0-9_$]*|-?(?:\d+(?:\.\d+)?|\.\d+))'),
    }


def extract_latest_year(text: str) -> str | None:
    match = re.search(r"งบ(?:ปี|ไตรมาส)\s+([0-9]{4})", text)
    if match:
        return match.group(1)
    return None


def fetch_set_financial_data(symbol: str) -> dict:
    stock = normalize_stock_symbol(symbol)
    quote_url = SET_BASE_QUOTE_URL.format(symbol=stock)
    highlights_url = quote_url + "/financial-statement/company-highlights"
    balance_url = quote_url + "/financial-statement/latest/balance"

    try:
        highlights_html = fetch_text(highlights_url)
        balance_html = fetch_text(balance_url)
    except HTTPError as exc:
        raise ValueError(f"SET returned HTTP {exc.code} for stock {stock}.") from exc
    except URLError as exc:
        raise ValueError(f"Cannot reach SET for stock {stock}: {exc.reason}") from exc

    quote = extract_quote_metadata(highlights_html)

    # Financial statement pages often lack real-time price; fall back to main quote page
    if quote.get("price") is None:
        try:
            main_html = fetch_text(quote_url)
            main_quote = extract_quote_metadata(main_html)
            for field in ("price", "change", "percent_change", "company_name"):
                if quote.get(field) is None and main_quote.get(field) is not None:
                    quote[field] = main_quote[field]
        except Exception:
            pass
    latest_year = extract_latest_year(highlights_html) or extract_latest_year(balance_html)
    balance_metrics = parse_balance_metrics_from_payload(balance_html)

    selected_balance_labels = [
        "รวมสินทรัพย์",
        "รวมหนี้สิน",
        "รวมส่วนของผู้ถือหุ้น",
        "เงินสด",
        "เงินให้สินเชื่อแก่ลูกหนี้และดอกเบี้ยค้างรับ - สุทธิ",
        "กำไร (ขาดทุน) สะสม - ยังไม่ได้จัดสรร",
    ]
    selected_balance = []
    for label in selected_balance_labels:
        for row_label, row_value in balance_metrics:
            if row_label == label:
                selected_balance.append((row_label, row_value))
                break

    if not selected_balance:
        selected_balance = balance_metrics[:8]

    return {
        "symbol": stock,
        "company_name": quote.get("company_name") or stock,
        "price": quote.get("price"),
        "change": quote.get("change"),
        "percent_change": quote.get("percent_change"),
        "latest_year": latest_year,
        "highlights_url": highlights_url,
        "balance_url": balance_url,
        "balance_metrics": selected_balance,
    }


def format_set_financial_summary(data: dict) -> str:
    lines = [
        f"📊 SET financial summary for {data['symbol']}",
        data["company_name"],
    ]
    price_val = data.get("price")
    price_line = f"📈 Price: {price_val if price_val is not None else 'N/A'} THB"
    if data.get("change") is not None:
        price_line += f" | Change: {data['change']}"
    if data.get("percent_change") is not None:
        try:
            pct = round(float(data["percent_change"]), 2)
            price_line += f" ({pct:+.2f}%)"
        except (TypeError, ValueError):
            pass
    lines.append(price_line)
    if data.get("latest_year"):
        lines.append(f"Latest statement: {data['latest_year']}")

    if data.get("balance_metrics"):
        lines.append("Key balance metrics:")
        for label, value in data["balance_metrics"]:
            lines.append(f"- {label}: {value} ล้านบาท")

    lines.append(f"SET highlights: {data['highlights_url']}")
    return "\n".join(lines)
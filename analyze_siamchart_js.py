import urllib.request
import re

print("Analyzing Siamchart's main JavaScript file...\n")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "http://siamchart.com/stock/"
}

try:
    print("⏳ Fetching /all_js.js (this may be large)...")
    req = urllib.request.Request("http://siamchart.com/all_js.js", headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        js_code = resp.read().decode('utf-8', errors='ignore')
    
    print(f"✅ Fetched {len(js_code)} bytes of JavaScript\n")
    
    # Look for important patterns
    patterns = {
        "API/endpoint URLs": r"['\"]([^\s'\"]*(?:api|jdata|fetch|load)[^\s'\"]*)['\"]",
        "Data loading methods": r"(?:fetch|XMLHttpRequest|ajax|getJSON)\s*\(\s*['\"]([^'\"]+)['\"]",
        "Variable assignments": r"(?:var|let|const)\s+(\w+)\s*=\s*new\s+(?:XMLHttpRequest|Fetch)",
        "Stock table references": r"(?:stock|ticker|symbol|data)\s*[=:]",
    }
    
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, js_code, re.I)
        if matches:
            unique_matches = list(set(matches))[:5]
            print(f"🔍 {pattern_name}:")
            for match in unique_matches:
                if len(match) > 3:
                    print(f"   - {match[:80]}")
    
    # Look specifically for jdata references
    if "jdata" in js_code.lower():
        print(f"\n✅ Found 'jdata' references")
        jdata_matches = re.findall(r"jdata[^\s'\"]*", js_code, re.I)
        for match in set(jdata_matches)[:5]:
            print(f"   - {match}")
    
    # Check for any absolute URLs
    urls = re.findall(r'["\']([a-z][a-z0-9+.-]*://[^\s"\'<>]+)["\']', js_code, re.I)
    if urls:
        print(f"\n✅ Found {len(set(urls))} unique URLs:")
        for url in list(set(urls))[:10]:
            print(f"   - {url[:80]}")
    
    # Look at the beginning to understand structure
    print(f"\n📄 First 2000 characters (to understand overall structure):")
    print("="*80)
    print(js_code[:2000])
    
except Exception as e:
    print(f"❌ Error: {e}")

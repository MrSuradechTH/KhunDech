import urllib.request
import re

print("Analyzing Siamchart stock page HTML structure...\n")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

try:
    req = urllib.request.Request("http://siamchart.com/stock/", headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        
    print(f"✅ Fetched HTML: {len(html)} bytes\n")
    
    # Look for data in different places
    checks = [
        ("Contains <table", "<table" in html),
        ("Contains table_data id", "table_data" in html),
        ("Contains var data", re.search(r"var\s+\w+\s*=\s*\[", html) is not None),
        ("Contains jdata.jsp ref", "jdata.jsp" in html),
        ("Contains ajax/fetch", "ajax" in html.lower() or "fetch" in html.lower()),
        ("Is it a redirect?", html.count('<') < 200),
    ]
    
    for check_name, result in checks:
        print(f"{check_name}: {'✅ YES' if result else '❌ NO'}")
    
    print("\n📄 Checking page structure...")
    print(f"HTML length: {len(html)} bytes")
    print(f"Line count: {html.count(chr(10))}")
    
    # Look for JavaScript in page
    js_match = re.search(r"<script[^>]*>(.*?)</script>", html, re.S)
    if js_match:
        js_code = js_match.group(1)
        print(f"✅ Found embedded JavaScript: {len(js_code)} bytes")
        # Check if it loads from external file
        src_match = re.search(r"<script[^>]*src=['\"]([^'\"]+)['\"]", html)
        if src_match:
            print(f"   External script: {src_match.group(1)}")
    
    # Check for iframe or similar
    if "iframe" in html.lower():
        print("⚠️  Page uses iframe(s)")
    
    # Search for JSON data
    json_match = re.search(r"window\.\w+\s*=\s*(\{|\[)", html)
    if json_match:
        print("✅ Found JSON-like data assignment")
    
    print("\n📊 Checking for table or grid setup...")
    if "table_data" in html:
        print("✅ Found 'table_data' reference")
        idx = html.find("table_data")
        print(f"   Context: ...{html[max(0,idx-100):idx+100]}...")
    
    # Check the first 2000 chars for clues
    print("\n" + "="*80)
    print("First 1500 characters of HTML:")
    print("="*80)
    print(html[:1500])
    
except Exception as e:
    print(f"❌ Error: {e}")

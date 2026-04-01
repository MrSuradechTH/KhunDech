import urllib.request
import re

print("Testing alternative Siamchart endpoints...\n")

endpoints = [
    ("jdata.jsp (current)", "http://siamchart.com/jdata.jsp"),
    ("stock page HTML", "http://siamchart.com/stock/"),
    ("api endpoint", "http://siamchart.com/api/stock"),
    ("snapshot", "http://siamchart.com/snapshot.jsp"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

for name, url in endpoints:
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            size = len(data)
            is_json = b"json" in data[:500].lower()
            is_html = data.startswith(b"<")
            has_stocks = b"BTG" in data or b"FM" in data or b"RCL" in data
            
            print(f"✅ {name:20} - {size:6} bytes")
            print(f"   Type: {'JSON' if is_json else 'HTML' if is_html else 'Unknown'}")
            print(f"   Contains stocks: {'YES' if has_stocks else 'NO'}")
            if has_stocks:
                print(f"   ⭐ HAS STOCK DATA!")
            print()
    except Exception as e:
        print(f"❌ {name:20} - Error: {str(e)[:60]}\n")

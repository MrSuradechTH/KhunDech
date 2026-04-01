import urllib.request
import time
import re

print("Testing jdata.jsp with different approaches...\n")

headers_variants = [
    ("Standard", {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }),
    ("Browser-like", {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "http://siamchart.com/stock/"
    }),
]

for label, headers in headers_variants:
    print(f"🔍 Attempt: {label}")
    try:
        req = urllib.request.Request("http://siamchart.com/jdata.jsp", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            size = len(data)
            has_stock = b"[['" in data[:500] or b"var" in data[:500]
            print(f"   Response: {size} bytes")
            print(f"   Has array data: {has_stock}")
            
            # Count how many rows
            rows = len(re.findall(rb"\],\[", data))
            print(f"   Estimated rows in response: {rows}")
            
            if size > 100:
                print(f"   First 150 bytes: {data[:150]}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:70]}")
    
    print()
    time.sleep(1)

print("\n⚠️ Analysis:")
print("If responses are empty or much smaller than expected,")
print("Siamchart may have changed their API or blocked the endpoint.")

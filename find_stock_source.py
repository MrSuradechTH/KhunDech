import urllib.request
import json
import time

print("Investigating alternative Thai stock data sources...\n")

sources = [
    ("SET API - All Stocks", "https://api.set.or.th/api/market/stock?page=1&limit=500"),
    ("SET API - Sectors", "https://api.set.or.th/api/stat/trading-stats"),
    ("Settrade API", "https://www.settrade.com/api/quote/"),
    ("Siamchart cached/alternate", "http://siamchart.com/stock/index.php"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*"
}

for name, url in sources:
    print(f"🔍 Testing: {name}")
    print(f"   URL: {url}")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            size = len(data)
            print(f"   ✅ Response: {size} bytes")
            
            # Try to parse as JSON
            try:
                json_data = json.loads(data)
                print(f"   ✅ Valid JSON: {type(json_data).__name__}")
                
                if isinstance(json_data, dict):
                    keys = list(json_data.keys())[:5]
                    print(f"      Keys: {keys}")
                elif isinstance(json_data, list):
                    print(f"      Array length: {len(json_data)}")
                    if json_data:
                        print(f"      First item keys: {list(json_data[0].keys())[:5] if isinstance(json_data[0], dict) else 'not a dict'}")
            except:
                print(f"   ❌ Not JSON, first 100 bytes: {data[:100]}")
            
    except urllib.error.HTTPError as e:
        print(f"   ❌ HTTP Error: {e.code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:70]}")
    
    print()
    time.sleep(1)

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print("1. Check if SET API works - may have stock fundamental data")
print("2. If SET API works, update web_scraping.py to use it instead of jdata.jsp")
print("3. Alternative: Use quandl, Yahoo Finance, or other providers")

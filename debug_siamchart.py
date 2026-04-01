import sys
sys.path.insert(0, 'source')
from khundech.web_scraping import get_siamchart_data, _extract_siamchart_rows

data = get_siamchart_data()
print(f"✅ Fetched {len(data)} bytes from jdata.jsp")
print("\n📋 First 1000 chars of response:")
print(data[:1000])
print("\n...")
print("\n📊 Extracting rows...")
rows = _extract_siamchart_rows(data)
print(f"Found {len(rows)} total rows")
if rows:
    print(f"First row has {len(rows[0])} columns")
    print(f"First row (first 5 cols): {rows[0][:5]}")

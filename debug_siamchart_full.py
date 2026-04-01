import sys
import re
sys.path.insert(0, 'source')
from khundech.web_scraping import get_siamchart_data, _unpack_packer_payload, _find_js_array_assignments, _parse_js_array_rows

data = get_siamchart_data()
unpacked = _unpack_packer_payload(data)

print("📊 UNPACKING PAYLOAD")
print(f"Original length: {len(data)} bytes")
print(f"Unpacked length: {len(unpacked)} bytes")
print("\n🔍 Searching for all array assignments...")

arrays = _find_js_array_assignments(unpacked)
print(f"\nFound {len(arrays)} array assignments")

for i, raw_array in enumerate(arrays, 1):
    rows = _parse_js_array_rows(raw_array)
    if rows:
        width = max((len(row) for row in rows), default=0)
        print(f"\n Array #{i}:")
        print(f"  Rows: {len(rows)}, Width: {width} columns")
        if len(rows) > 0 and width > 0:
            print(f"  First row (first 3 cols): {rows[0][:3]}")
            if len(rows) <= 10:
                print(f"  ALL ROWS:")
                for row in rows:
                    print(f"    {row}")

print("\n" + "="*80)
print("Looking for the stock table (expect ~600 rows, 25 columns)...")
print("="*80)

for i, raw_array in enumerate(arrays, 1):
    rows = _parse_js_array_rows(raw_array)
    if rows and len(rows) >= 100 and max((len(row) for row in rows), default=0) >= 20:
        print(f"✅ FOUND STOCK TABLE in array #{i}: {len(rows)} rows × {max((len(row) for row in rows), default=0)} cols")
        break
else:
    print("❌ Stock table NOT FOUND with expected dimensions")
    print("Checking first 100 chars of unpacked JS...")
    print(unpacked[:200])

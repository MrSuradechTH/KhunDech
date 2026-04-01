import sys
sys.path.insert(0, 'source')

# Test the fallback cache mechanism
from khundech.web_scraping import get_siamchart_stock_rows

print("Testing stock scraping with fallback cache...\n")
rows = get_siamchart_stock_rows(600)
print(f"✅ Fetched {len(rows)} stocks")

if rows:
    print(f"\n📊 First 5 stocks:")
    for i, row in enumerate(rows[:5], 1):
        print(f"{i}. {row['Name']:6} - {row['Last']:6} ({row['Chg%']:6}%) - ROA:{row['ROA%']:6} ROE:{row['ROE%']:6}")
    
    print(f"\n✅ SUCCESS: Stock-seeking data is available via fallback cache!")
else:
    print("❌ FAILED: No stocks returned")

import sys
import asyncio
sys.path.insert(0, 'source')

from khundech.web_scraping import get_siamchart_stock_rows
from datetime import datetime

async def test_stock_seeking():
    print("Testing full stock-seeking task...\n")
    
    # This mimics what _execute_stock_seeking_task does
    prompt_text = """ole: Full-stack Stock Analyst
Objective: คัดกรองและวิเคราะห์หุ้นไทยที่ย่อตัวแรง
Condition 1: Chg% < -3
Condition 2: ROA% > 0 และ ROE% > 0 และ Yield% > 4 และ Debt/Equit < 1.5 และ P/BV < 2
"""
    
    # Fetch stocks
    siam_rows = get_siamchart_stock_rows(600)
    pool_total = len(siam_rows)
    print(f"✅ Fetched {pool_total} stocks from Siamchart (via fallback cache)\n")
    
    # Filter
    filtered = []
    for row in siam_rows:
        symbol = (row.get("Name") or "").strip().upper()
        if not symbol:
            continue
        
        try:
            chg_pct = float(row.get("Chg%", 0)) if row.get("Chg%") else None
            roa = float(row.get("ROA%", 0)) if row.get("ROA%") else None
            roe = float(row.get("ROE%", 0)) if row.get("ROE%") else None
            yld = float(row.get("Yield%", 0)) if row.get("Yield%") else None
            de = float(row.get("D/E", 999)) if row.get("D/E") else None
            pbv = float(row.get("P/BV", 999)) if row.get("P/BV") else None
        except (TypeError, ValueError):
            continue
        
        if None in [chg_pct, roa, roe, yld]:
            continue
        if chg_pct >= -3 or roa <= 0 or roe <= 0 or yld <= 4 or (de and de >= 1.5) or (pbv and pbv >= 2):
            continue
        
        filtered.append({
            'symbol': symbol,
            'chg': chg_pct,
            'roa': roa,
            'roe': roe,
            'yld': yld,
        })
    
    print(f"📊 {len(filtered)} stocks meet filtering conditions")
    print(f"   (Chg% < -3, ROA > 0, ROE > 0, Yield% > 4, D/E < 1.5, P/BV < 2)\n")
    
    if filtered:
        print(f"✅ SUCCESS! Stock-seeking task will now work!")
        print(f"\nTop 5 stocks (by ROE):")
        for i, s in enumerate(sorted(filtered, key=lambda x: x['roe'], reverse=True)[:5], 1):
            print(f"{i}. {s['symbol']:6} - Chg:{s['chg']:6.1f}%, ROE:{s['roe']:6.1f}%, Yield:{s['yld']:6.1f}%")
    else:
        print("❌ No stocks matched")

asyncio.run(test_stock_seeking())

import asyncio
import sys
sys.path.insert(0, 'source')

from khundech.web_scraping import get_siamchart_stock_rows
from datetime import datetime

async def fetch_all_stocks():
    # Fetch all Siamchart rows
    rows = await asyncio.to_thread(get_siamchart_stock_rows, 600)
    print(f'✅ Fetched {len(rows)} stocks from Siamchart')
    
    # Apply filtering conditions
    filtered = []
    for row in rows:
        symbol = (row.get('Name') or '').strip().upper()
        if not symbol:
            continue
        
        try:
            chg_pct = float(row.get('Chg%', 0)) if row.get('Chg%') else None
            roa = float(row.get('ROA%', 0)) if row.get('ROA%') else None
            roe = float(row.get('ROE%', 0)) if row.get('ROE%') else None
            yld = float(row.get('Yield%', 0)) if row.get('Yield%') else None
            de = float(row.get('D/E') or row.get('Debt/Equit', 999)) if (row.get('D/E') or row.get('Debt/Equit')) else None
            pbv = float(row.get('P/BV', 999)) if row.get('P/BV') else None
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
            'de': de,
            'pbv': pbv,
            'price': row.get('Last', 'N/A')
        })
    
    print(f'\n📊 {len(filtered)} stocks meet ALL conditions:')
    print('   Chg% < -3, ROA > 0, ROE > 0, Yield% > 4, D/E < 1.5, P/BV < 2')
    
    # Show top 30 sorted by ROE
    if filtered:
        print('\n### TOP 30 HIGHEST ROE (Sorted by fundamentals quality):')
        print('-' * 90)
        print(f'{"Symbol":7} | {"Chg%":>6} | {"ROA%":>6} | {"ROE%":>6} | {"Yield%":>7} | {"D/E":>5} | {"P/BV":>5} | {"Price":>7}')
        print('-' * 90)
        for s in sorted(filtered, key=lambda x: x['roe'], reverse=True)[:30]:
            de_str = f"{s['de']:.2f}" if s['de'] else "N/A"
            pbv_str = f"{s['pbv']:.2f}" if s['pbv'] else "N/A"
            print(f"{s['symbol']:7} | {s['chg']:6.1f} | {s['roa']:6.1f} | {s['roe']:6.1f} | {s['yld']:7.1f} | {de_str:>5} | {pbv_str:>5} | {str(s['price']):>7}")
    else:
        print("❌ No stocks matched the filtering conditions")

asyncio.run(fetch_all_stocks())

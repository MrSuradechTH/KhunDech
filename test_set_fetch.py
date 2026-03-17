#!/usr/bin/env python3
"""Quick test to verify set_finance.py fetch_set_financial_data works"""
import sys
sys.path.insert(0, '/app')

from khundech.set_finance import fetch_set_financial_data
import json

try:
    print("Testing fetch_set_financial_data('CPF')...")
    data = fetch_set_financial_data('CPF')
    print("\n✅ SUCCESS - Real data fetched from SET website:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

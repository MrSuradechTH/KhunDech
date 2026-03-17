#!/usr/bin/env python3
import asyncio
from bot import handle_natural_language_tools

async def test_full_flow():
    test_cases = [
        "make a task to give me stock report every 2 hours",
        "show auto task we have",
        "delete this task",
        "show auto task we have",
    ]
    
    for msg in test_cases:
        result = await handle_natural_language_tools(msg, [])
        print(f"\n📝 INPUT: {msg}")
        print(f"✅ OUTPUT:\n{result}")
        print("=" * 70)

asyncio.run(test_full_flow())

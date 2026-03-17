#!/usr/bin/env python3
import asyncio
from bot import handle_natural_language_tools

async def test_task_removal():
    test_cases = [
        "show auto task we have",
        "remove the task",
    ]
    
    for msg in test_cases:
        result = await handle_natural_language_tools(msg, [])
        print(f"INPUT: {msg}")
        print(f"OUTPUT: {result}")
        print("=" * 60 + "\n")

asyncio.run(test_task_removal())

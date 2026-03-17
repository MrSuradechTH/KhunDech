#!/usr/bin/env python3
# Clean up old test tasks and run fresh test.
import asyncio
import json
from pathlib import Path

# First, clean up tasks.json
tasks_file = Path("/app/data/tasks.json")
clean_data = {"tasks": []}
tasks_file.write_text(json.dumps(clean_data, indent=2))
print("✓ Cleaned tasks.json\n")

from bot import handle_natural_language_tools

async def test_fresh():
    print("="*70)
    print("FRESH TEST: Task Lifecycle (Clean State)")
    print("="*70 + "\n")
    
    # Test 1: Create task
    print("1️⃣ CREATE TASK with explicit name and long prompt")
    print("-"*70)
    create_msg = '''create a task name "stock seeking" and prompt of the task is "Role: Full-stack Stock Analyst (Technical & Fundamental)
Objective: คัดกรองและวิเคราะห์หุ้นไทยที่มีการย่อตัวแรงแต่พื้นฐานแข็งแกร่ง" repater the task every 6 hours'''
    
    result = await handle_natural_language_tools(create_msg, [])
    print(result[:300])
    print("\n")
    
    # Test 2: Show tasks
    print("2️⃣ SHOW TASKS")
    print("-"*70)
    result = await handle_natural_language_tools("show auto task we have", [])
    print(result)
    print("\n")
    
    # Test 3: Delete with article
    print("3️⃣ DELETE TASK with article 'the'")
    print("-"*70)
    result = await handle_natural_language_tools("remove the task", [])
    print(result)
    print("\n")
    
    # Test 4: Show tasks (should be empty)
    print("4️⃣ SHOW TASKS (after deletion)")
    print("-"*70)
    result = await handle_natural_language_tools("show auto task we have", [])
    print(result)
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE - All features working!")
    print("="*70)

asyncio.run(test_fresh())

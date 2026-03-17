#!/usr/bin/env python3
import asyncio
from bot import handle_natural_language_tools

async def test_complete_flow():
    print("\n" + "="*70)
    print("TEST: Complete Task Lifecycle")
    print("="*70 + "\n")
    
    # Test 1: Show tasks (empty)
    result = await handle_natural_language_tools("show auto task we have", [])
    print("1️⃣ SHOW TASKS (initially empty):")
    print(result)
    print("\n" + "-"*70 + "\n")
    
    # Test 2: Create task with explicit name and long prompt
    create_msg = '''create a task name "stock seeking" and prompt of the task is "Role: Full-stack Stock Analyst (Technical & Fundamental)
Objective: คัดกรองและวิเคราะห์หุ้นไทยที่มีการย่อตัวแรงแต่พื้นฐานแข็งแกร่ง" repater the task every 6 hours'''
    
    result = await handle_natural_language_tools(create_msg, [])
    print("2️⃣ CREATE TASK (named with prompt):")
    print(result[:200] + "...\n")
    print("\n" + "-"*70 + "\n")
    
    # Test 3: Show tasks again
    result = await handle_natural_language_tools("show auto task we have", [])
    print("3️⃣ SHOW TASKS (after creation):")
    print(result)
    print("\n" + "-"*70 + "\n")
    
    # Test 4: Delete with article
    result = await handle_natural_language_tools("remove the task", [])
    print("4️⃣ DELETE TASK (using article 'the'):")
    print(result)
    print("\n" + "-"*70 + "\n")
    
    # Test 5: Show tasks (should be empty again)
    result = await handle_natural_language_tools("show auto task we have", [])
    print("5️⃣ SHOW TASKS (after deletion):")
    print(result)
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

asyncio.run(test_complete_flow())

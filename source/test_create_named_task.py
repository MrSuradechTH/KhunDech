#!/usr/bin/env python3
import asyncio
from bot import handle_natural_language_tools

async def test_create_task():
    # Test the exact user input
    msg = '''create a task name "stock seeking" and prompt of the task is "Role: Full-stack Stock Analyst (Technical & Fundamental)
Objective: คัดกรองและวิเคราะห์หุ้นไทยที่มีการย่อตัวแรงแต่พื้นฐานแข็งแกร่ง เพื่อหาโอกาสในการทำ Technical Rebound
ใช้ !setfinancial ในการดึงข้อมูลจาก website set พยายามลบข้อมูลเว็บที่ไม่จำเป็นออกไปเพื่อลดภาระงาน และ optimize ข้อมูลต่างๆเพื่อลด token" repater the task every 6 hours'''
    
    result = await handle_natural_language_tools(msg, [])
    print("="*70)
    print("INPUT (truncated):")
    print(msg[:100] + "...\n")
    print("OUTPUT:")
    print(result)
    print("="*70)
    
asyncio.run(test_create_task())

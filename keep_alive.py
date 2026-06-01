"""
سكربت Keep-Alive لمنع Replit من إيقاف البوت
يرسل طلبات HTTP دورية لخادم Flask المحلي
"""

import asyncio
import aiohttp
import os

KEEP_ALIVE_URL = os.environ.get("REPLIT_DEV_DOMAIN", "")
FLASK_PORT = os.environ.get("PORT", "8080")

async def keep_alive_ping():
    """إرسال ping دوري لمنع السكون"""
    if not KEEP_ALIVE_URL:
        print("ℹ️ لا يوجد REPLIT_DEV_DOMAIN، تخطي Keep-Alive")
        return

    url = f"https://{KEEP_ALIVE_URL}/health"
    print(f"🔄 Keep-Alive يعمل على: {url}")

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        print("💚 Keep-Alive ping ناجح")
                    else:
                        print(f"💛 Keep-Alive ping: {resp.status}")
        except Exception as e:
            print(f"💛 Keep-Alive خطأ: {e}")

        # ping كل 3 دقائق
        await asyncio.sleep(180)


if __name__ == "__main__":
    asyncio.run(keep_alive_ping())

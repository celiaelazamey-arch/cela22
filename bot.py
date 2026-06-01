"""
🤖 نظام AI Agents متقدم على Telegram
Telegram AI Multi-Agent System

معمارية شاملة:
- Telegram Bot API (Long Polling)
- Multi-Agent System (Planner, Researcher, Coder, Writer)
- Memory Management (JSON-based persistent storage)
- Task Planning & Execution
- Flask Keep-Alive Server for Replit
- Free LLM Integration via Pollinations AI

النشر المجاني: Replit
"""

import os
import sys
import json
import asyncio
import aiohttp
import threading
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

# ================================
# 1️⃣ SYSTEM CONFIGURATION
# ================================

@dataclass
class SystemConfig:
    """إعدادات النظام الرئيسية"""
    TELEGRAM_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_API_URL: str = "https://api.telegram.org/bot"
    # Pollinations AI - مجاني بدون مفتاح API
    LLM_API_URL: str = "https://text.pollinations.ai/"
    LLM_MODEL: str = "openai"
    MAX_MEMORY_ENTRIES: int = 1000
    DEFAULT_TIMEOUT: int = 60
    MEMORY_FILE: str = "data/memory.json"
    SESSIONS_FILE: str = "data/sessions.json"
    FLASK_PORT: int = int(os.environ.get("PORT", 8080))
    POLLING_TIMEOUT: int = 30
    POLLING_LIMIT: int = 100

    def __post_init__(self):
        # التأكد من مجلد البيانات
        os.makedirs("data", exist_ok=True)

config = SystemConfig()

# ================================
# 2️⃣ DATA MODELS
# ================================

@dataclass
class Message:
    """نموذج الرسالة"""
    chat_id: int
    user_id: int
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: int = 0
    is_bot: bool = False
    first_name: str = ""
    username: str = ""

@dataclass
class MemoryEntry:
    """إدخال في ذاكرة النظام"""
    entry_id: str
    chat_id: int
    content: str
    agent_type: str  # "planner", "researcher", "coder", "writer", "system"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    importance: int = 1  # 1-5

@dataclass
class Task:
    """نموذج المهمة"""
    task_id: str
    chat_id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    agent_assigned: str = ""
    result: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

@dataclass
class UserSession:
    """جلسة المستخدم"""
    chat_id: int
    user_name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_preferences: Dict = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)

# ================================
# 3️⃣ LLM CLIENT - Pollinations AI (مجاني)
# ================================

class LLMClient:
    """عميل نماذج اللغة - يستخدم Pollinations AI مجاناً"""

    def __init__(self):
        self.api_url = config.LLM_API_URL
        self.model = config.LLM_MODEL
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """الحصول على جلسة HTTP"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(self, prompt: str, system_prompt: str = "", max_retries: int = 3) -> str:
        """توليد استجابة من نموذج اللغة"""
        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant:"

        for attempt in range(max_retries):
            try:
                session = await self._get_session()
                params = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "system": system_prompt if system_prompt else "أنت مساعد ذكي تتحدث العربية بطلاقة.",
                }

                async with session.post(
                    self.api_url,
                    json=params,
                    timeout=aiohttp.ClientTimeout(total=config.DEFAULT_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        # تنظيف الاستجابة
                        text = text.strip()
                        if text.startswith("Assistant:"):
                            text = text[len("Assistant:"):].strip()
                        return text
                    else:
                        error_text = await response.text()
                        print(f"❌ LLM Error (attempt {attempt+1}): Status {response.status}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                print(f"⏱️ LLM Timeout (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"❌ LLM Exception (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return "⚠️ لم أتمكن من الحصول على استجابة من نموذج اللغة. حاول مرة أخرى."

    async def close(self):
        """إغلاق الجلسة"""
        if self.session and not self.session.closed:
            await self.session.close()

llm_client = LLMClient()

# ================================
# 4️⃣ MEMORY SYSTEM
# ================================

class MemoryManager:
    """مدير الذاكرة المستمر"""

    def __init__(self, memory_file: str = "data/memory.json"):
        self.memory_file = memory_file
        self.memory: Dict[int, List[Dict]] = {}
        self._load_memory()

    def _load_memory(self):
        """تحميل الذاكرة من الملف"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory = {int(k): v for k, v in data.items()}
                print(f"✅ تم تحميل الذاكرة: {sum(len(v) for v in self.memory.values())} إدخال")
        except Exception as e:
            print(f"❌ خطأ في تحميل الذاكرة: {e}")
            self.memory = {}

    def save_memory(self):
        """حفظ الذاكرة"""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"❌ خطأ في حفظ الذاكرة: {e}")

    def add_entry(self, chat_id: int, entry: MemoryEntry):
        """إضافة إدخال جديد"""
        if chat_id not in self.memory:
            self.memory[chat_id] = []

        entry_dict = asdict(entry)
        self.memory[chat_id].append(entry_dict)

        # حافظ على الحد الأقصى
        if len(self.memory[chat_id]) > config.MAX_MEMORY_ENTRIES:
            self.memory[chat_id] = self.memory[chat_id][-config.MAX_MEMORY_ENTRIES:]

        self.save_memory()

    def get_recent(self, chat_id: int, limit: int = 10) -> List[str]:
        """احصل على أحدث الإدخالات"""
        if chat_id not in self.memory:
            return []

        entries = self.memory[chat_id][-limit:]
        return [entry.get('content', str(entry)) for entry in entries]

    def search(self, chat_id: int, keyword: str) -> List[str]:
        """ابحث في الذاكرة"""
        if chat_id not in self.memory:
            return []

        results = []
        keyword_lower = keyword.lower()
        for entry in self.memory[chat_id]:
            content = entry.get('content', '')
            if keyword_lower in content.lower():
                results.append(content)

        return results

    def get_context(self, chat_id: int, limit: int = 15) -> str:
        """احصل على السياق للـ LLM"""
        recent = self.get_recent(chat_id, limit=limit)
        if not recent:
            return "لا يوجد سياق سابق."
        return "\n".join([f"- {entry}" for entry in recent])

    def clear(self, chat_id: int):
        """حذف ذاكرة مستخدم"""
        if chat_id in self.memory:
            self.memory[chat_id] = []
            self.save_memory()

memory_manager = MemoryManager()

# ================================
# 5️⃣ SESSION MANAGER
# ================================

class SessionManager:
    """مدير جلسات المستخدمين"""

    def __init__(self, sessions_file: str = "data/sessions.json"):
        self.sessions_file = sessions_file
        self.sessions: Dict[int, Dict] = {}
        self._load_sessions()

    def _load_sessions(self):
        """تحميل الجلسات"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sessions = {int(k): v for k, v in data.items()}
                print(f"✅ تم تحميل {len(self.sessions)} جلسة")
        except Exception as e:
            print(f"❌ خطأ في تحميل الجلسات: {e}")

    def save_sessions(self):
        """حفظ الجلسات"""
        try:
            os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"❌ خطأ في حفظ الجلسات: {e}")

    def get_or_create_session(self, chat_id: int, user_name: str = "مستخدم") -> Dict:
        """احصل على جلسة أو أنشئ واحدة جديدة"""
        if chat_id not in self.sessions:
            self.sessions[chat_id] = {
                "chat_id": chat_id,
                "user_name": user_name,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "agent_preferences": {},
                "conversation_history": []
            }
        else:
            self.sessions[chat_id]["last_activity"] = datetime.now().isoformat()
            self.sessions[chat_id]["user_name"] = user_name

        self.save_sessions()
        return self.sessions[chat_id]

    def add_to_history(self, chat_id: int, role: str, content: str):
        """أضف إلى تاريخ المحادثات"""
        session = self.get_or_create_session(chat_id)
        session["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # احتفظ بآخر 50 رسالة
        if len(session["conversation_history"]) > 50:
            session["conversation_history"] = session["conversation_history"][-50:]

        self.save_sessions()

    def get_history(self, chat_id: int, limit: int = 10) -> List[Dict]:
        """احصل على تاريخ المحادثات"""
        session = self.get_or_create_session(chat_id)
        return session.get("conversation_history", [])[-limit:]

session_manager = SessionManager()

# ================================
# 6️⃣ AGENT SYSTEM
# ================================

class Agent(ABC):
    """الفئة الأساسية للوكيل"""

    def __init__(self, name: str, description: str, emoji: str = "🤖"):
        self.name = name
        self.description = description
        self.emoji = emoji
        self.llm = llm_client

    @abstractmethod
    async def execute(self, task: str, context: str = "") -> str:
        """تنفيذ المهمة"""
        pass

    def __str__(self):
        return f"{self.emoji} {self.name}: {self.description}"


class PlannerAgent(Agent):
    """وكيل التخطيط - يحلل المهام ويخطط خطوات التنفيذ"""

    def __init__(self):
        super().__init__(
            name="المخطط",
            description="يقسم المهام المعقدة إلى خطوات صغيرة قابلة للتنفيذ",
            emoji="📋"
        )

    async def execute(self, task: str, context: str = "") -> str:
        """خطة المهمة"""
        system_prompt = """أنت وكيل تخطيط ذكي ومتخصص. مهمتك:
1. تحليل المهمة المعطاة بعمق
2. تقسيمها إلى خطوات منطقية واضحة
3. تحديد الأولويات والتبعيات بين الخطوات
4. اقتراح الوكلاء المناسبين لكل خطوة
5. تقدير الوقت والجهد المطلوب

أجب باللغة العربية بطريقة منظمة وواضحة."""

        prompt = f"""المهمة: {task}
السياق السابق: {context if context else 'لا يوجد سياق سابق'}

قم بتحليل هذه المهمة ووضع خطة تنفيذ مفصلة."""

        return await self.llm.generate(prompt, system_prompt)


class ResearcherAgent(Agent):
    """وكيل البحث - يبحث عن المعلومات"""

    def __init__(self):
        super().__init__(
            name="الباحث",
            description="يبحث عن المعلومات ويجمع البيانات من مصادر مختلفة",
            emoji="🔍"
        )

    async def execute(self, task: str, context: str = "") -> str:
        """البحث عن المعلومات"""
        system_prompt = """أنت باحث ذكي متخصص. مهمتك:
1. تحليل استعلام البحث بعناية
2. تقديم معلومات شاملة وموثوقة
3. تنظيم المعلومات بطريقة منطقية
4. الإشارة إلى المصادر المحتملة
5. تقديم خلاصة واضحة

أجب باللغة العربية بطريقة منظمة وموثوقة. قدم معلومات دقيقة ومحدثة."""

        prompt = f"""استعلام البحث: {task}
السياق: {context if context else 'لا يوجد سياق'}

ابحث عن المعلومات المتعلقة بهذا الموضوع وقدمها بطريقة منظمة."""

        return await self.llm.generate(prompt, system_prompt)


class CoderAgent(Agent):
    """وكيل البرمجة - يكتب الأكواد"""

    def __init__(self):
        super().__init__(
            name="المبرمج",
            description="يكتب وينقح الأكواد البرمجية بلغات مختلفة",
            emoji="💻"
        )

    async def execute(self, task: str, context: str = "") -> str:
        """كتابة الكود"""
        system_prompt = """أنت مبرمج ماهر ومتعدد اللغات. مهمتك:
1. كتابة كود نظيف وفعال وموثق
2. إضافة معالجة شاملة للأخطاء
3. توفير أمثلة الاستخدام
4. شرح الكود خطوة بخطوة
5. اقتراح التحسينات الممكنة

أجب باللغة العربية مع كتابة الكود باللغة البرمجية المناسبة. استخدم code blocks."""

        prompt = f"""المتطلبات: {task}
السياق: {context if context else 'لا يوجد سياق'}

اكتب الكود المطلوب مع الشرح والتحسينات."""

        return await self.llm.generate(prompt, system_prompt)


class WriterAgent(Agent):
    """وكيل الكتابة - يكتب المحتوى"""

    def __init__(self):
        super().__init__(
            name="الكاتب",
            description="يكتب المحتوى الإبداعي والمهني بمختلف أنواعه",
            emoji="✍️"
        )

    async def execute(self, task: str, context: str = "") -> str:
        """كتابة المحتوى"""
        system_prompt = """أنت كاتب محترف ومبدع. مهمتك:
1. كتابة محتوى جذاب ومنظم
2. مراعاة الأسلوب والجمهور المستهدف
3. استخدام لغة غنية ومعبرة
4. تنظيم الأفكار بشكل منطقي
5. إضافة أمثلة وتوضيحات عند الحاجة

أجب باللغة العربية بأسلوب احترافي وإبداعي."""

        prompt = f"""المطلوب كتابته: {task}
السياق: {context if context else 'لا يوجد سياق'}

اكتب المحتوى المطلوب بأسلوب احترافي."""

        return await self.llm.generate(prompt, system_prompt)


class AgentManager:
    """مدير الوكلاء"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {
            "planner": PlannerAgent(),
            "researcher": ResearcherAgent(),
            "coder": CoderAgent(),
            "writer": WriterAgent(),
        }

    async def execute_agent(self, agent_name: str, task: str, context: str = "") -> str:
        """تنفيذ وكيل معين"""
        if agent_name not in self.agents:
            available = ", ".join(self.agents.keys())
            return f"❌ الوكيل '{agent_name}' غير موجود.\nالوكلاء المتاحة: {available}"

        agent = self.agents[agent_name]
        try:
            return await agent.execute(task, context)
        except Exception as e:
            return f"❌ خطأ في تنفيذ الوكيل {agent_name}: {str(e)}"

    async def auto_route(self, task: str, context: str = "") -> str:
        """توجيه تلقائي - يختار الوكيل المناسب"""
        task_lower = task.lower()

        # كلمات مفتاحية لكل وكيل
        coder_keywords = ["كود", "برنامج", "دالة", "function", "code", "python", "script",
                          "برمجة", "خوارزمية", "algorithm", "api", "تطبيق", "app",
                          "debug", "اصلاح", "خطأ", "برمج", "اكتب كود", "كلاس", "class"]

        researcher_keywords = ["بحث", "معلومات", "ما هو", "ما هي", "explain", "search",
                               "ابحث", "شرح", "تاريخ", "تعريف", "مفهوم", "فكرة",
                               "مصدر", "مرجع", "مقارنة", "difference"]

        writer_keywords = ["اكتب", "مقال", "قصة", "رسالة", "email", "write", "article",
                           "نص", "محتوى", "إبداعي", "شعر", "خطاب", "تقرير", "ملخص"]

        planner_keywords = ["خطط", "خطة", "خطة عمل", "plan", "تخطيط", "مشروع",
                            "استراتيجية", "خطوات", "مراحل", "تنفيذ"]

        # تحديد الوكيل بناءً على الكلمات المفتاحية
        scores = {
            "coder": sum(1 for kw in coder_keywords if kw in task_lower),
            "researcher": sum(1 for kw in researcher_keywords if kw in task_lower),
            "writer": sum(1 for kw in writer_keywords if kw in task_lower),
            "planner": sum(1 for kw in planner_keywords if kw in task_lower),
        }

        best_agent = max(scores, key=scores.get)

        # إذا لم يتطابق أي كلمة مفتاحية، استخدم المخطط
        if scores[best_agent] == 0:
            best_agent = "planner"

        return await self.execute_agent(best_agent, task, context)

    def list_agents(self) -> str:
        """قائمة بالوكلاء المتاحة"""
        result = "🤖 الوكلاء المتاحة:\n\n"
        for name, agent in self.agents.items():
            result += f"  {agent.emoji} <b>{agent.name}</b> ({name})\n     {agent.description}\n\n"
        result += "\n💡 يمكنك استخدام /auto ليتم توجيه طلبك تلقائياً للوكيل المناسب."
        return result

agent_manager = AgentManager()

# ================================
# 7️⃣ TELEGRAM BOT HANDLER
# ================================

class TelegramBotHandler:
    """معالج بوت Telegram مع Long Polling"""

    def __init__(self, token: str):
        self.token = token
        self.api_url = f"{config.TELEGRAM_API_URL}{token}"
        self.last_update_id = 0
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """الحصول على جلسة HTTP"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML",
                           reply_to: int = None) -> bool:
        """إرسال رسالة"""
        try:
            session = await self._get_session()

            # تقسيم الرسائل الطويلة
            chunks = self._split_message(text, max_length=4000)

            for chunk in chunks:
                payload = {
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": parse_mode
                }
                if reply_to:
                    payload["reply_to_message_id"] = reply_to

                async with session.post(
                    f"{self.api_url}/sendMessage",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        print(f"❌ خطأ في إرسال الرسالة: {error[:200]}")
                    await asyncio.sleep(0.5)  # تأخير بسيط بين الرسائل

            return True
        except Exception as e:
            print(f"❌ خطأ في إرسال الرسالة: {e}")
            return False

    def _split_message(self, text: str, max_length: int = 4000) -> List[str]:
        """تقسيم الرسائل الطويلة"""
        if len(text) <= max_length:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            # ابحث عن نقطة قطع مناسبة
            split_pos = text.rfind('\n', 0, max_length)
            if split_pos == -1:
                split_pos = text.rfind(' ', 0, max_length)
            if split_pos == -1:
                split_pos = max_length

            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip('\n ')

        return chunks

    async def send_typing(self, chat_id: int):
        """إظهار مؤشر الكتابة"""
        try:
            session = await self._get_session()
            await session.post(
                f"{self.api_url}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"}
            )
        except:
            pass

    async def get_updates(self) -> List[Dict]:
        """الحصول على التحديثات (Long Polling)"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.api_url}/getUpdates",
                json={
                    "offset": self.last_update_id + 1,
                    "timeout": config.POLLING_TIMEOUT,
                    "limit": config.POLLING_LIMIT,
                    "allowed_updates": ["message"]
                },
                timeout=aiohttp.ClientTimeout(total=config.POLLING_TIMEOUT + 10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        updates = data.get("result", [])
                        if updates:
                            self.last_update_id = updates[-1]["update_id"]
                        return updates
                return []
        except asyncio.TimeoutError:
            return []
        except Exception as e:
            print(f"❌ خطأ في جلب التحديثات: {e}")
            await asyncio.sleep(5)
            return []

    async def get_me(self) -> Optional[Dict]:
        """الحصول على معلومات البوت"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.api_url}/getMe") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result")
        except Exception as e:
            print(f"❌ خطأ في getMe: {e}")
        return None

    async def close(self):
        """إغلاق الجلسة"""
        if self.session and not self.session.closed:
            await self.session.close()

# ================================
# 8️⃣ COMMAND HANDLER
# ================================

class CommandHandler:
    """معالج الأوامر"""

    def __init__(self):
        self.commands = {
            "/start": self._handle_start,
            "/help": self._handle_help,
            "/agents": self._handle_agents,
            "/planner": self._handle_planner,
            "/research": self._handle_research,
            "/code": self._handle_code,
            "/write": self._handle_write,
            "/auto": self._handle_auto,
            "/memory": self._handle_memory,
            "/search": self._handle_search_memory,
            "/status": self._handle_status,
            "/clear": self._handle_clear,
            "/reset": self._handle_reset,
        }

    async def _handle_start(self, chat_id: int, args: str = "", **kwargs) -> str:
        """بدء البوت"""
        user_name = kwargs.get("user_name", "صديقي")
        session_manager.get_or_create_session(chat_id, user_name)

        return f"""👋 أهلاً {user_name}! أنا بوت ذكي متعدد الوكلاء

🤖 خدماتي:
  📋 /planner [مهمة] - تخطيط المهام المعقدة
  🔍 /research [موضوع] - البحث عن المعلومات
  💻 /code [طلب] - كتابة الأكواد
  ✍️ /write [مطلوب] - كتابة المحتوى
  🤖 /auto [سؤال] - توجيه تلقائي ذكي

📊 إدارة النظام:
  • /agents - عرض الوكلاء المتاحة
  • /memory - عرض آخر العمليات
  • /search [كلمة] - البحث في الذاكرة
  • /status - حالة النظام
  • /clear - حذف الذاكرة
  • /reset - إعادة تعيين كاملة

📝 أمثلة:
  /planner بناء نظام إدارة مشروع
  /research أحدث تطورات الذكاء الاصطناعي
  /code دالة لحساب متتالية فيبوناتشي
  /write مقال عن أهمية التعلم المستمر
  /auto كيف أبني تطبيق موبايل؟

استمتع! 🚀"""

    async def _handle_help(self, chat_id: int, args: str = "", **kwargs) -> str:
        """مساعدة"""
        return """📚 دليل الاستخدام الكامل

🎯 الأوامر الرئيسية:

1️⃣ /planner [وصف المهمة]
   → يقسم المهمة المعقدة إلى خطوات تنفيذية واضحة

2️⃣ /research [الموضوع]
   → يبحث عن المعلومات ويجمع البيانات

3️⃣ /code [المتطلبات]
   → يكتب الكود البرمجي مع الشرح

4️⃣ /write [المطلوب]
   → يكتب المحتوى الإبداعي والمهني

5️⃣ /auto [سؤال أو طلب]
   → يوجه طلبك تلقائياً للوكيل الأنسب

📊 أوامر إدارة النظام:

  • /agents - قائمة الوكلاء المتاحة
  • /memory - آخر 5 عمليات في الذاكرة
  • /search [كلمة] - البحث في الذاكرة
  • /status - حالة النظام والإحصائيات
  • /clear - حذف الذاكرة
  • /reset - إعادة تعيين كاملة

💡 نصائح للاستخدام الأمثل:
  - كن محدداً في طلباتك للحصول على نتائج أفضل
  - استخدم /auto إذا لم تكن متأكداً أي وكيل تختار
  - البوت يتذكر سياق المحادثة تلقائياً
  - يمكنك إرسال أي رسالة بدون أمر وسيتم الرد تلقائياً

🔄 ملاحظة:
  النظام يستخدم Pollinations AI مجاناً.
  قد تكون الاستجابة أبطأ قليلاً مقارنة بالخدمات المدفوعة."""

    async def _handle_agents(self, chat_id: int, args: str = "", **kwargs) -> str:
        """عرض الوكلاء"""
        return agent_manager.list_agents()

    async def _handle_planner(self, chat_id: int, args: str = "", **kwargs) -> str:
        """وكيل التخطيط"""
        if not args.strip():
            return "❌ الرجاء تحديد المهمة\n📝 مثال: /planner بناء تطبيق تجارة إلكترونية"

        await telegram_bot.send_typing(chat_id)
        context = memory_manager.get_context(chat_id)

        result = await agent_manager.execute_agent("planner", args, context)

        # حفظ في الذاكرة
        memory_manager.add_entry(
            chat_id,
            MemoryEntry(
                entry_id=f"plan_{uuid.uuid4().hex[:8]}",
                chat_id=chat_id,
                content=f"تخطيط: {args}",
                agent_type="planner",
                tags=["تخطيط"]
            )
        )
        session_manager.add_to_history(chat_id, "planner", args)

        return f"📋 <b>خطة المهمة</b>\n\n{result}"

    async def _handle_research(self, chat_id: int, args: str = "", **kwargs) -> str:
        """وكيل البحث"""
        if not args.strip():
            return "❌ الرجاء تحديد موضوع البحث\n📝 مثال: /research الذكاء الاصطناعي التوليدي"

        await telegram_bot.send_typing(chat_id)
        context = memory_manager.get_context(chat_id)

        result = await agent_manager.execute_agent("researcher", args, context)

        memory_manager.add_entry(
            chat_id,
            MemoryEntry(
                entry_id=f"search_{uuid.uuid4().hex[:8]}",
                chat_id=chat_id,
                content=f"بحث: {args}",
                agent_type="researcher",
                tags=["بحث"]
            )
        )
        session_manager.add_to_history(chat_id, "researcher", args)

        return f"🔍 <b>نتائج البحث</b>\n\n{result}"

    async def _handle_code(self, chat_id: int, args: str = "", **kwargs) -> str:
        """وكيل البرمجة"""
        if not args.strip():
            return "❌ الرجاء تحديد متطلبات الكود\n📝 مثال: /code دالة لحساب المضروب بلغة Python"

        await telegram_bot.send_typing(chat_id)
        context = memory_manager.get_context(chat_id)

        result = await agent_manager.execute_agent("coder", args, context)

        memory_manager.add_entry(
            chat_id,
            MemoryEntry(
                entry_id=f"code_{uuid.uuid4().hex[:8]}",
                chat_id=chat_id,
                content=f"كود: {args}",
                agent_type="coder",
                tags=["برمجة", "كود"]
            )
        )
        session_manager.add_to_history(chat_id, "coder", args)

        return f"💻 <b>الكود</b>\n\n{result}"

    async def _handle_write(self, chat_id: int, args: str = "", **kwargs) -> str:
        """وكيل الكتابة"""
        if not args.strip():
            return "❌ الرجاء تحديد ما تريد كتابته\n📝 مثال: /write مقال عن التقنية"

        await telegram_bot.send_typing(chat_id)
        context = memory_manager.get_context(chat_id)

        result = await agent_manager.execute_agent("writer", args, context)

        memory_manager.add_entry(
            chat_id,
            MemoryEntry(
                entry_id=f"write_{uuid.uuid4().hex[:8]}",
                chat_id=chat_id,
                content=f"كتابة: {args}",
                agent_type="writer",
                tags=["كتابة", "محتوى"]
            )
        )
        session_manager.add_to_history(chat_id, "writer", args)

        return f"✍️ <b>المحتوى</b>\n\n{result}"

    async def _handle_auto(self, chat_id: int, args: str = "", **kwargs) -> str:
        """توجيه تلقائي"""
        if not args.strip():
            return "❌ الرجاء كتابة طلبك\n📝 مثال: /auto كيف أبني موقع؟"

        await telegram_bot.send_typing(chat_id)
        context = memory_manager.get_context(chat_id)

        result = await agent_manager.auto_route(args, context)

        memory_manager.add_entry(
            chat_id,
            MemoryEntry(
                entry_id=f"auto_{uuid.uuid4().hex[:8]}",
                chat_id=chat_id,
                content=f"تلقائي: {args}",
                agent_type="auto",
                tags=["تلقائي"]
            )
        )
        session_manager.add_to_history(chat_id, "auto", args)

        return f"🤖 <b>رد ذكي</b>\n\n{result}"

    async def _handle_memory(self, chat_id: int, args: str = "", **kwargs) -> str:
        """عرض الذاكرة"""
        limit = 5
        if args.strip().isdigit():
            limit = min(int(args.strip()), 20)

        recent = memory_manager.get_recent(chat_id, limit=limit)

        if not recent:
            return "📦 الذاكرة فارغة حالياً\n💡 ابدأ باستخدام الأوامر لتخزين الأنشطة"

        result = f"📦 آخر {len(recent)} عمليات:\n\n"
        for i, entry in enumerate(recent, 1):
            # اختصار الإدخالات الطويلة
            display = entry if len(entry) <= 100 else entry[:97] + "..."
            result += f"  {i}. {display}\n"

        return result

    async def _handle_search_memory(self, chat_id: int, args: str = "", **kwargs) -> str:
        """البحث في الذاكرة"""
        if not args.strip():
            return "❌ الرجاء تحديد كلمة البحث\n📝 مثال: /search فيبوناتشي"

        results = memory_manager.search(chat_id, args.strip())

        if not results:
            return f"🔍 لم يتم العثور على نتائج لـ '{args.strip()}'"

        response = f"🔍 نتائج البحث عن '{args.strip()}' ({len(results)} نتيجة):\n\n"
        for i, entry in enumerate(results[:10], 1):
            display = entry if len(entry) <= 100 else entry[:97] + "..."
            response += f"  {i}. {display}\n"

        return response

    async def _handle_status(self, chat_id: int, args: str = "", **kwargs) -> str:
        """حالة النظام"""
        session = session_manager.get_or_create_session(chat_id)
        memory_count = len(memory_manager.get_recent(chat_id, 1000))
        history_count = len(session_manager.get_history(chat_id, 1000))

        return f"""✅ <b>حالة النظام</b>

📊 الإحصائيات:
  • إجمالي العمليات في الذاكرة: {memory_count}
  • رسائل المحادثة: {history_count}
  • الوكلاء النشطة: {len(agent_manager.agents)}
  • آخر نشاط: الآن

⚙️ المعلومات:
  • المستخدم: {session.get('user_name', 'غير معروف')}
  • معرف الدردشة: {chat_id}
  • حجم السجل: {history_count}

🟢 الحالة: يعمل بكفاءة
🤖 المحرك: Pollinations AI (مجاني)"""

    async def _handle_clear(self, chat_id: int, args: str = "", **kwargs) -> str:
        """حذف الذاكرة"""
        memory_manager.clear(chat_id)
        return "✅ تم حذف الذاكرة بنجاح\n💡 يمكنك البدء من جديد"

    async def _handle_reset(self, chat_id: int, args: str = "", **kwargs) -> str:
        """إعادة تعيين كاملة"""
        memory_manager.clear(chat_id)
        if chat_id in session_manager.sessions:
            session_manager.sessions[chat_id]["conversation_history"] = []
            session_manager.save_sessions()
        return "🔄 تم إعادة التعيين بالكامل\n💡 ابدأ محادثة جديدة مع /start"

    async def process_message(self, message: Message) -> Optional[str]:
        """معالجة الرسالة الواردة"""
        text = message.text.strip()

        if not text:
            return None

        # حفظ رسالة المستخدم
        session_manager.add_to_history(message.chat_id, "user", text)

        # استخراج الأمر والمعاملات
        parts = text.split(" ", 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # البحث عن الأمر
        for cmd_name, handler in self.commands.items():
            if command == cmd_name or command.startswith(cmd_name + "@"):
                try:
                    return await handler(
                        message.chat_id,
                        args=args,
                        user_name=message.first_name or "مستخدم"
                    )
                except Exception as e:
                    print(f"❌ خطأ في معالجة الأمر {cmd_name}: {e}")
                    return f"❌ حدث خطأ في تنفيذ الأمر. حاول مرة أخرى."

        # إذا لم يكن أمر معروف، استخدم التوجيه التلقائي
        await telegram_bot.send_typing(message.chat_id)
        context = memory_manager.get_context(message.chat_id)

        result = await agent_manager.auto_route(text, context)

        memory_manager.add_entry(
            message.chat_id,
            MemoryEntry(
                entry_id=f"msg_{uuid.uuid4().hex[:8]}",
                chat_id=message.chat_id,
                content=f"رسالة: {text[:100]}",
                agent_type="auto",
                tags=["رسالة"]
            )
        )
        session_manager.add_to_history(message.chat_id, "assistant", result[:200])

        return result

command_handler = CommandHandler()

# ================================
# 9️⃣ FLASK KEEP-ALIVE SERVER
# ================================

def create_flask_app():
    """إنشاء تطبيق Flask للبقاء حياً على Replit"""
    from flask import Flask, jsonify
    import time

    app = Flask(__name__)
    start_time = time.time()

    @app.route('/')
    def home():
        uptime = int(time.time() - start_time)
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        return jsonify({
            "status": "running",
            "bot": "Telegram AI Multi-Agent System",
            "uptime": f"{hours}h {minutes}m",
            "agents": list(agent_manager.agents.keys()),
            "version": "2.0"
        })

    @app.route('/health')
    def health():
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

    @app.route('/stats')
    def stats():
        return jsonify({
            "total_memory_entries": sum(len(v) for v in memory_manager.memory.values()),
            "total_sessions": len(session_manager.sessions),
            "agents": len(agent_manager.agents),
            "active": True
        })

    return app

def run_flask():
    """تشغيل Flask في thread منفصل"""
    app = create_flask_app()
    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False, use_reloader=False)

# ================================
# 🔟 MAIN BOT LOOP
# ================================

class TelegramBot:
    """البوت الرئيسي"""

    def __init__(self, token: str):
        self.handler = TelegramBotHandler(token)
        self.running = False
        self.command_handler = CommandHandler()

    async def process_update(self, update: Dict):
        """معالجة تحديث واحد"""
        try:
            message_data = update.get("message")
            if not message_data:
                return

            # تجاهل رسائل البوت
            if message_data.get("from", {}).get("is_bot", False):
                return

            # استخراج النص
            text = message_data.get("text", "")
            if not text:
                return

            # إنشاء كائن الرسالة
            message = Message(
                chat_id=message_data["chat"]["id"],
                user_id=message_data["from"]["id"],
                text=text,
                message_id=message_data.get("message_id", 0),
                is_bot=False,
                first_name=message_data.get("from", {}).get("first_name", "مستخدم"),
                username=message_data.get("from", {}).get("username", "")
            )

            print(f"📩 [{message.first_name}] {text[:50]}...")

            # معالجة الرسالة
            response = await self.command_handler.process_message(message)

            if response:
                await self.handler.send_message(
                    message.chat_id,
                    response,
                    reply_to=message.message_id
                )
                print(f"✅ تم الرد على {message.first_name}")

        except Exception as e:
            print(f"❌ خطأ في معالجة التحديث: {e}")

    async def start(self):
        """بدء البوت"""
        self.running = True

        # التحقق من التوكن
        if not config.TELEGRAM_TOKEN or config.TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("❌ لم يتم تعيين TELEGRAM_BOT_TOKEN!")
            print("📝 احصل على التوكن من @BotFather على Telegram")
            print("🔑 ضعه في Secrets كـ TELEGRAM_BOT_TOKEN")
            return

        # التحقق من اتصال البوت
        bot_info = await self.handler.get_me()
        if not bot_info:
            print("❌ فشل الاتصال بـ Telegram API. تحقق من التوكن.")
            return

        bot_name = bot_info.get("first_name", "Bot")
        bot_username = bot_info.get("username", "")
        print(f"✅ البوت متصل: {bot_name} (@{bot_username})")
        print("🚀 بدء الاستماع للرسائل...")

        # حلقة التحديثات الرئيسية
        while self.running:
            try:
                updates = await self.handler.get_updates()

                for update in updates:
                    await self.process_update(update)

            except Exception as e:
                print(f"❌ خطأ في حلقة التحديثات: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        """إيقاف البوت"""
        self.running = False
        await self.handler.close()
        await llm_client.close()
        print("🛑 تم إيقاف البوت")


async def main():
    """الدالة الرئيسية"""
    print("=" * 50)
    print("🤖 Telegram AI Multi-Agent System")
    print("🌐 Powered by Pollinations AI (Free)")
    print("=" * 50)

    # بدء Flask في thread منفصل
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"🌐 Flask server running on port {config.FLASK_PORT}")

    # بدء البوت
    bot = TelegramBot(config.TELEGRAM_TOKEN)
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n⏹️ تم طلب الإيقاف...")
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())

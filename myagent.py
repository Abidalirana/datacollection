# myagent.py
"""
FundedFlow Mindset Coach Agent (OpenAI Agents SDK version)
- Fully compatible with OpenAI Agents SDK
- Preserves all DB imports & query logic
- Exports run_datacollector_agent(user_id, db) for FastAPI
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
import asyncio

# --- Project models (unchanged) ---
from db import (
    User, Trade, Journal, Emotion, Session,
    ResetChallenge, FeatureUsage, RecoveryPlan, RulebookVote, SimulatorLog,
    AIInteraction
)

# --- Agent framework & SDK ---
from agents import Agent, handoff, Runner, function_tool
from agents import OpenAIChatCompletionsModel

# --- Gemini / OpenAI client ---
from openai import AsyncOpenAI

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("myagent_sdk")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")

# OpenAI SDK client & model
external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
model = OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client)

# ---------------- Query Helpers (unchanged) ----------------
async def get_recent_trades(db: AsyncSession, user_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Trade)
        .where(Trade.user_id == user_id)
        .order_by(Trade.created_at.desc())
        .limit(limit)
    )
    trades = result.scalars().all()
    return [
        {
            "symbol": t.symbol,
            "pnl": float(t.pnl),
            "emotion": t.emotion_snapshot,
            "note": t.note or "",
            "strategy": t.strategy,
            "rr": t.risk_reward,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "max_drawdown": t.max_drawdown,
            "outcome": t.outcome,
            "time": t.created_at.isoformat(),
        }
        for t in trades
    ]

async def get_recent_journals(db: AsyncSession, user_id: int, limit: int = 2) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Journal)
        .where(Journal.user_id == user_id)
        .order_by(Journal.created_at.desc())
        .limit(limit)
    )
    journals = result.scalars().all()
    return [{"text": j.text, "confidence": j.confidence, "time": j.created_at.isoformat()} for j in journals]

async def get_recent_emotions(db: AsyncSession, user_id: int, days: int = 3) -> List[Dict[str, Any]]:
    since = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        select(Emotion)
        .where(Emotion.user_id == user_id, Emotion.created_at >= since)
        .order_by(Emotion.created_at.desc())
        .limit(10)
    )
    emos = result.scalars().all()
    return [
        {"tag": e.tag, "intensity": e.intensity, "note": e.note, "trade_id": e.trade_id, "time": e.created_at.isoformat()}
        for e in emos
    ]

async def get_feature_usage_glimpse(db: AsyncSession, user_id: int) -> Dict[str, int]:
    result = await db.execute(
        select(FeatureUsage.feature_name, func.count(FeatureUsage.id))
        .where(FeatureUsage.user_id == user_id)
        .group_by(FeatureUsage.feature_name)
        .order_by(desc(func.count(FeatureUsage.id)))
        .limit(5)
    )
    rows = result.all()
    return {name: count for (name, count) in rows}

async def get_reset_status(db: AsyncSession, user_id: int) -> Optional[Dict[str, Any]]:
    result = await db.execute(
        select(ResetChallenge)
        .where(ResetChallenge.user_id == user_id)
        .order_by(ResetChallenge.created_at.desc())
        .limit(1)
    )
    rc = result.scalars().first()
    if not rc:
        return None
    return {
        "name": rc.name,
        "day_target": rc.day_target,
        "days_completed": rc.days_completed,
        "success": rc.success,
        "started_at": rc.started_at.isoformat() if rc.started_at else None,
        "ended_at": rc.ended_at.isoformat() if rc.ended_at else None,
    }

async def load_context(db: AsyncSession, user_id: int) -> Dict[str, Any]:
    trades = await get_recent_trades(db, user_id)
    journals = await get_recent_journals(db, user_id)
    emotions = await get_recent_emotions(db, user_id)
    usage = await get_feature_usage_glimpse(db, user_id)
    reset = await get_reset_status(db, user_id)
    return {
        "trades": trades,
        "journals": journals,
        "emotions_recent": emotions,
        "feature_usage_top": usage,
        "reset_challenge": reset,
    }

# ---------------- Function Tool Helpers ----------------
@function_tool
def sentence_limiter_tool(text: str, max_sentences: int = 2) -> str:
    sentences = text.split(".")
    if len(sentences) > max_sentences:
        return ".".join(sentences[:max_sentences]).strip() + "."
    return text

# ---------------- Mindset Agent + Handoff ----------------
mindset_agent = Agent(
    name="Mindset Coach Agent",
    instructions="Given recent trades, journals, and emotions, generate ONE short actionable mindset tip.",
    tools=[sentence_limiter_tool],
)
mindset_handoff = handoff(mindset_agent, "mindset", "Generate mindset tip")

# ---------------- Agent Invocation ----------------
async def mindset_agent_runner(user_id: int, db: AsyncSession) -> str:
    ctx_data = await load_context(db, user_id)
    if not ctx_data["trades"]:
        tip = "Welcome! پہلے اپنی پہلی ٹریڈ لاگ کرو تاکہ میں تمہیں ذاتی ‘mindset tip’ دے سکوں۔"
        try:
            db.add(AIInteraction(user_id=user_id, question="no-trades-welcome", response=tip, meta={"context": ctx_data}))
            await db.commit()
        except Exception as e:
            logger.exception("Failed AIInteraction log: %s", e)
            await db.rollback()
        return tip

    prompt = f"""
You are the Mindset Coach for FundedFlow traders.
Recent data (JSON):
{ctx_data}
Instructions:
- Give ONE short, actionable tip (MAX 2 sentences).
- Focus on emotional control, recovery discipline, or risk process.
- If tilt patterns detected (loss streak + high-intensity emotions), suggest reset step.
- Avoid generic motivation; be specific.
"""

    try:
        # Run via OpenAI Agents SDK
        result = await Runner.run(
            mindset_agent,
            input=prompt,
            model=model
        )
        tip = result.output_text if hasattr(result, "output_text") else str(result)
        # Use function tool to limit sentences
        tip = sentence_limiter_tool(tip, max_sentences=2)
    except Exception as e:
        logger.exception("Agent SDK call failed: %s", e)
        tip = "Sorry — I'm having trouble generating a tip right now. Take a 10-minute break, breathe, and review your last entry calmly."

    # Log AI interaction
    try:
        db.add(AIInteraction(user_id=user_id, question="mindset_coach_prompt", response=tip, meta={"context": ctx_data}))
        await db.commit()
    except Exception as e:
        logger.exception("Failed AIInteraction log: %s", e)
        await db.rollback()

    return tip

# Attach to handoff
mindset_handoff.on_invoke_handoff = mindset_agent_runner

# ---------------- Public wrapper for FastAPI ----------------
async def run_datacollector_agent(user_id: int, db: AsyncSession) -> str:
    context = {"user_id": user_id, "db": db}
    return await mindset_handoff.on_invoke_handoff(context, "")

# ---------------- Optional CLI Test ----------------
if __name__ == "__main__":
    print("This module runs via OpenAI Agents SDK. Use FastAPI / call run_datacollector_agent().")



#==============================================#==============================================#==============================================
##============================================== alone runner #==============================================
if __name__ == "__main__":
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    # Example DB URL — replace with your real DB
    DATABASE_URL = "sqlite+aiosqlite:///./test.db"

    engine = create_async_engine(DATABASE_URL, echo=True)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def main():
        async with async_session() as db:
            user_id = 1  # replace with a real user_id
            tip = await run_datacollector_agent(user_id, db)
            print("Generated Mindset Tip:")
            print(tip)

    asyncio.run(main())

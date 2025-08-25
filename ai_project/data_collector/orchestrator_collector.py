# data_collector/orchestrator_collector.py
from __future__ import annotations
from .sqlalchemy.ext.asyncio import AsyncSession
from .user_profile import get_user_profile_data
from .trade_data import get_recent_trades_data
from .emotion_tracker import get_recent_emotions_data
from .engagement_logger import get_feature_usage_data
from .ai_interaction_logger import log_ai_interaction

async def collect_user_data(user_id: int, db: AsyncSession) -> dict:
    """
    Collect all user-related data from database.
    """
    user_profile = await get_user_profile_data(user_id, db)  # Demographics & account
    trades = await get_recent_trades_data(user_id, db)
    emotions = await get_recent_emotions_data(user_id, db)
    engagement = await get_feature_usage_data(user_id, db)

    context = {
        "user_profile": user_profile,
        "trades": trades,
        "emotions": emotions,
        "engagement": engagement
    }

    # Optional: log that we collected data
    await log_ai_interaction(user_id, "data_collector", "Collected user data successfully.", db, meta=context)
    return context

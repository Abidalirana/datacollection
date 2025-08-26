# data_collector/orchestrator_collector.py
from ai_project.data_collector.user_profile import log_user_profile
from ai_project.data_collector.trade_data import log_trade
from ai_project.data_collector.emotion_tracker import log_emotion
from ai_project.data_collector.journal_logger import log_journal
from ai_project.data_collector.ai_interaction_logger import log_ai_interaction
from sqlalchemy.ext.asyncio import AsyncSession


async def run_data_collection(sample_data: dict, db: AsyncSession):
    """
    Async orchestrator for anonymous data collection
    """
    # 1️⃣ User profile (anonymous)
    user_id = await log_user_profile(sample_data.get("user", {}), db)

    # 2️⃣ Trades
    for trade in sample_data.get("trades", []):
        trade["user_id"] = user_id
        await log_trade(trade, db)

    # 3️⃣ Emotions
    for emotion in sample_data.get("emotions", []):
        emotion["user_id"] = user_id
        await log_emotion(emotion, db)

    # 4️⃣ Journals
    for journal in sample_data.get("journals", []):
        journal["user_id"] = user_id
        await log_journal(journal, db)

    # 5️⃣ AI interactions
    for ai_data in sample_data.get("ai_interactions", []):
        ai_data["user_id"] = user_id
        await log_ai_interaction(ai_data, db)

    return user_id

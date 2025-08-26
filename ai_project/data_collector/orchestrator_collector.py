# data_collector/orchestrator_collector.py
from ai_project.data_collector.user_profile import log_user_profile
from ai_project.data_collector.trade_data import log_trade
from ai_project.data_collector.emotion_tracker import log_emotion
from ai_project.data_collector.journal_logger import log_journal
from ai_project.data_collector.ai_interaction_logger import log_ai_interaction

from ai_project.data_collector.sessions_logger import log_session
from ai_project.data_collector.reset_challenges_logger import log_reset_challenge
from ai_project.data_collector.feature_usage_logger import log_feature_usage
from ai_project.data_collector.recovery_plans_logger import log_recovery_plan
from ai_project.data_collector.rulebook_votes_logger import log_rulebook_vote

from sqlalchemy.ext.asyncio import AsyncSession


async def run_data_collection(sample_data: dict, db: AsyncSession):
    """
    Async orchestrator for all anonymous data collection
    """
    # 1️⃣ User profile
    user_id = await log_user_profile(sample_data.get("user", {}), db)

    # 2️⃣ Sessions
    for session_data in sample_data.get("sessions", []):
        session_data["user_id"] = user_id
        await log_session(session_data, db)

    # 3️⃣ Trades
    for trade in sample_data.get("trades", []):
        trade["user_id"] = user_id
        await log_trade(trade, db)

    # 4️⃣ Emotions
    for emotion in sample_data.get("emotions", []):
        emotion["user_id"] = user_id
        await log_emotion(emotion, db)

    # 5️⃣ Journals
    for journal in sample_data.get("journals", []):
        journal["user_id"] = user_id
        await log_journal(journal, db)

    # 6️⃣ AI interactions
    for ai_data in sample_data.get("ai_interactions", []):
        ai_data["user_id"] = user_id
        await log_ai_interaction(ai_data, db)

    # 7️⃣ Reset challenges
    for challenge in sample_data.get("reset_challenges", []):
        challenge["user_id"] = user_id
        await log_reset_challenge(challenge, db)

    # 8️⃣ Feature usage
    for usage in sample_data.get("feature_usages", []):
        usage["user_id"] = user_id
        await log_feature_usage(usage, db)

    # 9️⃣ Recovery plans
    for plan in sample_data.get("recovery_plans", []):
        plan["user_id"] = user_id
        await log_recovery_plan(plan, db)

    # 🔟 Rulebook votes
    for vote in sample_data.get("rulebook_votes", []):
        vote["user_id"] = user_id
        await log_rulebook_vote(vote, db)

    return user_id

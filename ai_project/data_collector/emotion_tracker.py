# data_collector/emotion_tracker.py
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

async def get_recent_emotions_data(user_id: int, db: AsyncSession, days: int = 3) -> list[dict]:
    """
    Get recent emotion logs (tag + intensity) for a user.
    """
    since = datetime.utcnow() - timedelta(days=days)
    result = await db.execute(
        "SELECT tag, intensity FROM emotions WHERE user_id = :uid AND created_at >= :since",
        {"uid": user_id, "since": since}
    )
    emos = result.fetchall()
    return [{"tag": e.tag, "intensity": e.intensity} for e in emos]

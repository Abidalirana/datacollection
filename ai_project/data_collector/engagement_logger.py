# data_collector/engagement_logger.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func

async def get_feature_usage_data(user_id: int, db: AsyncSession) -> dict:
    """
    Get top 5 features a user interacted with.
    """
    result = await db.execute(
        "SELECT feature_name, COUNT(id) as cnt FROM feature_usage "
        "WHERE user_id = :uid GROUP BY feature_name ORDER BY cnt DESC LIMIT 5",
        {"uid": user_id}
    )
    rows = result.fetchall()
    return {r.feature_name: r.cnt for r in rows}

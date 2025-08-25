# data_collector/ai_interaction_logger.py
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

async def log_ai_interaction(user_id: int, tool_name: str, message: str, db: AsyncSession, meta: dict = None):
    """
    Log data collection / AI interactions for auditing.
    """
    await db.execute(
        "INSERT INTO ai_interactions (user_id, tool_name, message, meta, created_at) "
        "VALUES (:uid, :tool, :msg, :meta, :time)",
        {
            "uid": user_id,
            "tool": tool_name,
            "msg": message,
            "meta": str(meta) if meta else None,
            "time": datetime.utcnow()
        }
    )
    await db.commit()

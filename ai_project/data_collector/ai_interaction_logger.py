# data_collector/ai_interaction_logger.py
from ai_project.database.models import SimulatorLog
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

async def log_ai_interaction(ai_data: dict, db: AsyncSession):
    """
    Log AI interactions for a user asynchronously
    """
    log_entry = SimulatorLog(
        user_id=ai_data["user_id"],
        action=ai_data.get("action"),
        timestamp=ai_data.get("timestamp") or datetime.utcnow()
    )
    db.add(log_entry)
    await db.commit()

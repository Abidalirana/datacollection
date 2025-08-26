# data_collector/sessions_logger.py
from ai_project.database.models import Session
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

async def log_session(session_data: dict, db: AsyncSession):
    """
    Log a user session anonymously
    """
    session = Session(
        user_id=session_data["user_id"],
        start_time=session_data.get("start_time") or datetime.utcnow(),
        end_time=session_data.get("end_time")
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session.id

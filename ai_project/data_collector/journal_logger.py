# data_collector/journal_logger.py
from ai_project.database.models import Journal
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

async def log_journal(journal_data: dict, db: AsyncSession):
    """
    Async log journal entries for psychological & behavioral analysis
    """
    journal = Journal(
        user_id=journal_data["user_id"],
        content=journal_data.get("content"),
        confidence_score=journal_data.get("confidence_score"),
        created_at=journal_data.get("created_at") or datetime.utcnow()
    )
    try:
        db.add(journal)
        await db.commit()
        await db.refresh(journal)
        return journal
    except Exception as e:
        await db.rollback()
        raise e

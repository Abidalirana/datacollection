# data_collector/emotion_tracker.py
from ai_project.database.models import Emotion
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

async def log_emotion(emotion_data: dict, db: AsyncSession):
    """
    Log emotion tags per trade asynchronously
    """
    emotion = Emotion(
        user_id=emotion_data["user_id"],
        trade_id=emotion_data.get("trade_id"),
        emotion=emotion_data.get("emotion"),
        timestamp=emotion_data.get("timestamp") or datetime.utcnow()
    )
    db.add(emotion)
    await db.commit()
    await db.refresh(emotion)  # optional, to get updated values like id
    return emotion.id  # return the inserted ID

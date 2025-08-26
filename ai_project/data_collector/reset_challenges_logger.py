# data_collector/reset_challenges_logger.py
from ai_project.database.models import ResetChallenge
from sqlalchemy.ext.asyncio import AsyncSession

async def log_reset_challenge(challenge_data: dict, db: AsyncSession):
    """
    Log a reset challenge anonymously
    """
    challenge = ResetChallenge(
        user_id=challenge_data["user_id"],
        completion_percentage=challenge_data.get("completion_percentage", 0.0),
        start_time=challenge_data.get("start_time"),
        end_time=challenge_data.get("end_time")
    )
    db.add(challenge)
    await db.commit()
    await db.refresh(challenge)
    return challenge.id

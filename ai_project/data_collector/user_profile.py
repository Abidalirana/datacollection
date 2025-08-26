# data_collector/user_profile.py
from ai_project.database.models import User, Session
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime


async def log_user_profile(data: dict, db: AsyncSession):
    """
    Async log anonymous user profile data: age, location, account_type, funded_status
    Returns: user_id
    """
    user = User(
        age=data.get("age"),
        location=data.get("location"),
        account_type=data.get("account_type"),
        funded_status=data.get("funded_status")
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Create a session for this user
    session = Session(user_id=user.id, start_time=datetime.utcnow())
    db.add(session)
    await db.commit()

    return user.id

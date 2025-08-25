# data_collector/user_profile.py
from sqlalchemy.ext.asyncio import AsyncSession

async def get_user_profile_data(user_id: int, db: AsyncSession) -> dict:
    """
    Return non-personal profile info.
    Excludes name, email, etc.
    """
    # Example placeholder: fetch funding status, account type, etc.
    result = await db.execute(
        "SELECT funding_status, account_type, location FROM users WHERE id = :uid",
        {"uid": user_id}
    )
    row = result.fetchone()
    if not row:
        return {}
    return {
        "funding_status": row.funding_status,
        "account_type": row.account_type,
        "location": row.location
    }

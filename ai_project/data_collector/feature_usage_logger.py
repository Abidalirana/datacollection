# data_collector/feature_usage_logger.py
from ai_project.database.models import FeatureUsage
from sqlalchemy.ext.asyncio import AsyncSession

async def log_feature_usage(usage_data: dict, db: AsyncSession):
    """
    Log anonymous feature usage
    """
    feature = FeatureUsage(
        user_id=usage_data["user_id"],
        feature_name=usage_data.get("feature_name"),
        usage_count=usage_data.get("usage_count", 1)
    )
    db.add(feature)
    await db.commit()
    await db.refresh(feature)
    return feature.id

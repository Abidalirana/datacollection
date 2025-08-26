# data_collector/recovery_plans_logger.py
from ai_project.database.models import RecoveryPlan
from sqlalchemy.ext.asyncio import AsyncSession

async def log_recovery_plan(plan_data: dict, db: AsyncSession):
    """
    Log anonymous recovery plans
    """
    plan = RecoveryPlan(
        user_id=plan_data["user_id"],
        plan_details=plan_data.get("plan_details"),
        completed=plan_data.get("completed", False)
    )
    db.add(plan)
    await db.commit()
    await db.refresh(plan)
    return plan.id

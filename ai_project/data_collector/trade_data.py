# data_collector/trade_data.py
from ai_project.database.models import Trade
from sqlalchemy.ext.asyncio import AsyncSession


async def log_trade(trade_data: dict, db: AsyncSession):
    """
    Async log a single trade for a user
    """
    trade = Trade(
        user_id=trade_data["user_id"],
        instrument=trade_data.get("instrument"),
        strategy=trade_data.get("strategy"),
        entry_time=trade_data.get("entry_time"),
        exit_time=trade_data.get("exit_time"),
        outcome=trade_data.get("outcome"),
        risk_reward_ratio=trade_data.get("risk_reward_ratio"),
        max_drawdown=trade_data.get("max_drawdown")
    )
    db.add(trade)
    await db.commit()
    await db.refresh(trade)
    return trade.id

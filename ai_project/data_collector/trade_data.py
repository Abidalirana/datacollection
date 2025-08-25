# data_collector/trade_data.py
from sqlalchemy.ext.asyncio import AsyncSession

async def get_recent_trades_data(user_id: int, db: AsyncSession) -> list[dict]:
    """
    Fetch recent trades for a user without sensitive info.
    """
    result = await db.execute(
        "SELECT symbol, pnl, strategy, entry_time, exit_time, outcome FROM trades "
        "WHERE user_id = :uid ORDER BY created_at DESC LIMIT 5",
        {"uid": user_id}
    )
    trades = result.fetchall()
    return [
        {
            "symbol": t.symbol,
            "pnl": float(t.pnl),
            "strategy": t.strategy,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "outcome": t.outcome
        } for t in trades
    ]

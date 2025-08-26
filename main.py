import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from ai_project.data_collector.orchestrator_collector import run_data_collection

# Load .env file
load_dotenv()

# Get database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Async session
async_session = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)

async def main():
    # Sample anonymous data for testing all 10 tables
    sample_data = {
        "user": {
            "age": 30,
            "location": "PK",
            "account_type": "FTMO",
            "funded_status": "demo"
        },
        "sessions": [
            {"start_time": None, "end_time": None}
        ],
        "trades": [
            {"instrument": "US30", "strategy": "scalping", "entry_time": None, "exit_time": None, "outcome": "win", "risk_reward_ratio": 2.0, "max_drawdown": 50}
        ],
        "emotions": [
            {"emotion": "confidence", "timestamp": None}
        ],
        "journals": [
            {"content": "Feeling good", "confidence_score": 0.8, "created_at": None}
        ],
        "ai_interactions": [
            {"action": "asked_risk", "timestamp": None}
        ],
        "reset_challenges": [
            {"completion_percentage": 50, "start_time": None, "end_time": None}
        ],
        "feature_usages": [
            {"feature_name": "risk_tracker", "usage_count": 1}
        ],
        "recovery_plans": [
            {"plan_details": "Follow reset plan A", "completed": False}
        ],
        "rulebook_votes": [
            {"rule_name": "Max Trade Limit", "vote": True}
        ]
    }

    async with async_session() as session:
        user_id = await run_data_collection(sample_data, session)
        print(f"✅ Logged anonymous user_id: {user_id}")

if __name__ == "__main__":
    asyncio.run(main())

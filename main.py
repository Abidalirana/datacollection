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
    sample_data = {
        "user": {"name": "Abid"},
        "trades": [{"instrument": "US30", "strategy": "scalping", "entry_time": None, "exit_time": None, "outcome": "win"}],
        "emotions": [{"emotion": "confidence", "timestamp": None}],
        "journals": [{"content": "Feeling good", "confidence_score": 0.8, "created_at": None}],
        "ai_interactions": [{"action": "asked_risk", "timestamp": None}]
    }

    async with async_session() as session:
        user_id = await run_data_collection(sample_data, session)
        print(f"✅ Logged user_id: {user_id}")

if __name__ == "__main__":
    asyncio.run(main())

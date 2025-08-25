# main.py
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from ai_project.data_collector.orchestrator_collector import collect_user_data

DATABASE_URL = "sqlite+aiosqlite:///./fundedflow.db"  # replace with your production DB

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def main():
    async with async_session() as db:
        user_id = 1  # example user
        data = await collect_user_data(user_id, db)
        print("Collected User Data:")
        print(data)

if __name__ == "__main__":
    asyncio.run(main())

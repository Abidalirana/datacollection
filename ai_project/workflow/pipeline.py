import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import correctly
from data_collector.orchestrator_collector import collect_user_data
from workflow.utils import log_info, pretty_print
from sqlalchemy.ext.asyncio import AsyncSession

async def run_pipeline(user_id: int, db: AsyncSession):
    log_info(f"Starting pipeline for user_id={user_id}")
    user_data = await collect_user_data(user_id, db)
    log_info(f"Collected data for user_id={user_id}")
    pretty_print(user_data)
    return user_data


if __name__ == "__main__":
    import asyncio
    from database.create_db import get_db_session  # Example DB session helper

    async def main():
        async with get_db_session() as db:
            await run_pipeline(user_id=1, db=db)

    asyncio.run(main())


# workflow/pipeline.py
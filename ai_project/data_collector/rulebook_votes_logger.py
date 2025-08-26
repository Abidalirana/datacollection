# data_collector/rulebook_votes_logger.py
from ai_project.database.models import RulebookVote
from sqlalchemy.ext.asyncio import AsyncSession

async def log_rulebook_vote(vote_data: dict, db: AsyncSession):
    """
    Log anonymous rulebook/community votes
    """
    vote = RulebookVote(
        user_id=vote_data["user_id"],
        rule_name=vote_data.get("rule_name"),
        vote=vote_data.get("vote", False)
    )
    db.add(vote)
    await db.commit()
    await db.refresh(vote)
    return vote.id

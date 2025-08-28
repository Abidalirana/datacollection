from ai_project.database.create_db import get_db_session
from ai_project.database.models import ProcessedFeature

def save_features(df):
    session = next(get_db_session())
    for _, row in df.iterrows():
        feature = ProcessedFeature(
            user_id=row.get("user_id_x"),
            trade_id=row.get("id_x"),
            risk_reward_ratio=row.get("risk_reward_ratio"),
            max_drawdown=row.get("max_drawdown"),
            outcome_encoded=row.get("outcome_encoded"),
            journal_length=row.get("journal_length"),
            instr_US30=row.get("instr_US30"),
            strategy_scalping=row.get("strategy_scalping"),
        )
        session.add(feature)
    session.commit()
    session.close()
    print("💾 All processed features saved to DB!")

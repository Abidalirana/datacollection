#database/save_prediction.py
from sqlalchemy.orm import Session
from ai_project.database.create_db import get_db_session
from ai_project.database.models import MLPrediction


def save_prediction(user_id, trade_id, model_name, model_version, prediction):
    session = next(get_db_session())
    ml_pred = MLPrediction(
        user_id=user_id,
        trade_id=trade_id,
        model_name=model_name,
        model_version=model_version,
        prediction=prediction
    )
    session.add(ml_pred)
    session.commit()
    session.close()

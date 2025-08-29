# database/save_prediction.py
from sqlalchemy.orm import Session
from contextlib import contextmanager
from ai_project.database.create_db import get_db_session
from ai_project.database.models import MLPrediction


@contextmanager
def get_session():
    """Context manager to safely handle DB session"""
    session = next(get_db_session())
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"❌ Error saving prediction: {e}")
    finally:
        session.close()


def save_prediction(user_id=None, trade_id=None, model_name="", model_version="v1", prediction=None):
    """
    Save an ML prediction to the database.
    Works with optional user_id/trade_id.
    """
    if not model_name:
        raise ValueError("model_name is required to save prediction")

    with get_session() as session:
        ml_pred = MLPrediction(
            user_id=user_id,
            trade_id=trade_id,
            model_name=model_name,
            model_version=model_version,
            prediction=prediction
        )
        session.add(ml_pred)

    print(f"💾 Saved prediction → {model_name} (trade_id={trade_id}, user_id={user_id})")

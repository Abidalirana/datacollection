# insert_dummy_data.py
from datetime import datetime, timedelta
import random
from faker import Faker
from ai_project.database.models import User, Trade, Emotion, Journal, ResetChallenge, FeatureUsage
from ai_project.database.create_db import SessionLocal

fake = Faker()

# ========================
# CONFIG
# ========================
NUM_USERS = 5          # number of anonymous users
TRADES_PER_USER = 3
JOURNALS_PER_USER = 3
EMOTIONS_PER_USER = 3
FEATURES = ["risk_tracker", "journal_logger", "strategy_helper"]

# ========================
# HELPER FUNCTIONS
# ========================
def random_datetime(start_hour=9, end_hour=16):
    today = datetime.now().date()
    hour = random.randint(start_hour, end_hour)
    minute = random.randint(0, 59)
    return datetime.combine(today, datetime.min.time()) + timedelta(hours=hour, minutes=minute)

def random_outcome():
    return random.choice(["win", "loss"])

def random_emotion():
    return random.choice(["confidence", "fear", "frustration", "excitement"])

# ========================
# GENERATE DATA
# ========================
db = SessionLocal()

for _ in range(NUM_USERS):
    user = User(
        age=random.randint(20, 50),
        location="Anonymous",
        account_type=random.choice(["FTMO", "Demo", "Live"]),
        funded_status=random.choice(["funded", "demo", "not funded"])
    )
    db.add(user)
    db.commit()  # commit to get the user.id
    db.refresh(user)

    # Trades
    for _ in range(TRADES_PER_USER):
        trade = Trade(
            user_id=user.id,
            instrument=random.choice(["US30", "NAS100", "SP500", "EURUSD"]),
            strategy=random.choice(["scalping", "swing", "day trading"]),
            entry_time=random_datetime(),
            exit_time=random_datetime(),
            outcome=random_outcome(),
            risk_reward_ratio=round(random.uniform(1, 3), 2),
            max_drawdown=round(random.uniform(10, 50), 2)
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)

        # Emotions for this trade
        for _ in range(EMOTIONS_PER_USER):
            emotion = Emotion(
                user_id=user.id,
                trade_id=trade.id,
                emotion=random_emotion(),
                timestamp=random_datetime()
            )
            db.add(emotion)

        # Journals for this trade
        for _ in range(JOURNALS_PER_USER):
            journal = Journal(
                user_id=user.id,
                content=fake.sentence(nb_words=10),
                created_at=random_datetime(),
                confidence_score=round(random.uniform(0, 1), 2)
            )
            db.add(journal)

    # Reset challenge
    reset = ResetChallenge(
        user_id=user.id,
        completion_percentage=random.choice([0, 50, 100]),
        start_time=random_datetime(),
        end_time=random_datetime()
    )
    db.add(reset)

    # Feature usage
    for feature in FEATURES:
        usage = FeatureUsage(
            user_id=user.id,
            feature_name=feature,
            usage_count=random.randint(1, 5)
        )
        db.add(usage)

db.commit()
db.close()

print(f"💾 Successfully inserted dummy data for {NUM_USERS} anonymous users!")

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, create_engine
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

# Users Table (anonymous)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    location = Column(String)
    account_type = Column(String)
    funded_status = Column(String)  # funded, not funded, demo

    # relationships
    sessions = relationship("Session", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    journals = relationship("Journal", back_populates="user") 
    emotions = relationship("Emotion", back_populates="user")
    reset_challenges = relationship("ResetChallenge", back_populates="user")
    feature_usages = relationship("FeatureUsage", back_populates="user")
    recovery_plans = relationship("RecoveryPlan", back_populates="user")
    rulebook_votes = relationship("RulebookVote", back_populates="user")
    simulator_logs = relationship("SimulatorLog", back_populates="user")

# Sessions Table
class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    
    user = relationship("User", back_populates="sessions")

# Journal Entries
class Journal(Base):
    __tablename__ = "journals"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float)
    
    user = relationship("User", back_populates="journals")

# Trades
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    instrument = Column(String)
    strategy = Column(String)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    outcome = Column(String)  # win/loss
    risk_reward_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    user = relationship("User", back_populates="trades")

# Emotions
class Emotion(Base):
    __tablename__ = "emotions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    emotion = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="emotions")

# Reset Challenges
class ResetChallenge(Base):
    __tablename__ = "reset_challenges"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    completion_percentage = Column(Float)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    
    user = relationship("User", back_populates="reset_challenges")

# Feature Usage
class FeatureUsage(Base):
    __tablename__ = "feature_usage"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    feature_name = Column(String)
    usage_count = Column(Integer)
    
    user = relationship("User", back_populates="feature_usages")

# Recovery Plans
class RecoveryPlan(Base):
    __tablename__ = "recovery_plans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_details = Column(Text)
    completed = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="recovery_plans")

# Rulebook Votes
class RulebookVote(Base):
    __tablename__ = "rulebook_votes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    rule_name = Column(String)
    vote = Column(Boolean)
    
    user = relationship("User", back_populates="rulebook_votes")

# Simulator Logs
class SimulatorLog(Base):
    __tablename__ = "simulator_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="simulator_logs")

# ML Predictions Table
class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # if prediction tied to user
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    model_name = Column(String)  # e.g., "tilt_predictor"
    model_version = Column(String)
    prediction = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    trade = relationship("Trade")


# Processed Features Table (preprocessing output)
class ProcessedFeature(Base):
    __tablename__ = "processed_features"  # table for storing preprocessed data

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)

    # Core features
    risk_reward_ratio = Column(Float)
    max_drawdown = Column(Float)
    outcome_encoded = Column(Integer)
    journal_length = Column(Integer)

    # Example strategy/instrument flags
    instr_US30 = Column(Boolean)
    strategy_scalping = Column(Boolean)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")
    trade = relationship("Trade")


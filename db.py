from __future__ import annotations
from datetime import datetime
import os
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    Text, String, Integer, Float, Boolean, ForeignKey, DateTime, Numeric, Index
)
from sqlalchemy.dialects.postgresql import JSONB

# ---------- Config ----------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:admin@localhost:5432/fundedflow"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# ---------- Base ----------
class Base(DeclarativeBase):
    pass

# ---------- Core Tables ----------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int]               = mapped_column(primary_key=True)
    name: Mapped[str]             = mapped_column(Text)
    email: Mapped[str]            = mapped_column(Text, unique=True, index=True)

    age: Mapped[Optional[int]]    = mapped_column(Integer, default=None)
    location: Mapped[Optional[str]] = mapped_column(Text, default=None)
    funding_status: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    account_type: Mapped[Optional[str]] = mapped_column(String(64), default=None)

    created_at: Mapped[datetime]  = mapped_column(DateTime, default=datetime.utcnow)

    trades: Mapped[list["Trade"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    journals: Mapped[list["Journal"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    emotions: Mapped[list["Emotion"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    reset_challenges: Mapped[list["ResetChallenge"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    feature_usage: Mapped[list["FeatureUsage"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    recovery_plans: Mapped[list["RecoveryPlan"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    rulebook_votes: Mapped[list["RulebookVote"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    simulator_logs: Mapped[list["SimulatorLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    ai_interactions: Mapped[list["AIInteraction"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_user_symbol_time", "user_id", "symbol", "created_at"),
    )

    id: Mapped[int]            = mapped_column(primary_key=True)
    user_id: Mapped[int]       = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    symbol: Mapped[str]        = mapped_column(Text)
    pnl: Mapped[float]         = mapped_column(Numeric(14, 2))
    emotion_snapshot: Mapped[Optional[str]] = mapped_column(Text, default=None)
    note: Mapped[str]          = mapped_column(Text, default="")

    strategy: Mapped[Optional[str]]       = mapped_column(Text, default=None)
    risk_reward: Mapped[Optional[float]]  = mapped_column(Float, default=None)
    entry_time: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    exit_time: Mapped[Optional[datetime]]  = mapped_column(DateTime, default=None)
    max_drawdown: Mapped[Optional[float]]  = mapped_column(Float, default=None)
    session_length_min: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    outcome: Mapped[Optional[str]] = mapped_column(String(16), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="trades")
    emotions: Mapped[list["Emotion"]] = relationship(back_populates="trade", cascade="all, delete-orphan")

class Journal(Base):
    __tablename__ = "journals"

    id: Mapped[int]           = mapped_column(primary_key=True)
    user_id: Mapped[int]      = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    text: Mapped[str]         = mapped_column(Text)
    confidence: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="journals")

class Emotion(Base):
    __tablename__ = "emotions"
    __table_args__ = (
        Index("ix_emotions_user_time", "user_id", "created_at"),
    )

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    trade_id: Mapped[Optional[int]] = mapped_column(ForeignKey("trades.id", ondelete="CASCADE"), index=True, default=None)

    tag: Mapped[str]             = mapped_column(String(64))
    intensity: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    note: Mapped[Optional[str]]  = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="emotions")
    trade: Mapped[Optional["Trade"]] = relationship(back_populates="emotions")

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    total_trades: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    notes: Mapped[Optional[str]] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="sessions")

class ResetChallenge(Base):
    __tablename__ = "reset_challenges"

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    name: Mapped[str]            = mapped_column(String(128))
    day_target: Mapped[int]      = mapped_column(Integer)
    days_completed: Mapped[int]  = mapped_column(Integer, default=0)
    success: Mapped[Optional[bool]] = mapped_column(Boolean, default=None)

    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="reset_challenges")

class FeatureUsage(Base):
    __tablename__ = "feature_usage"

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    feature_name: Mapped[str]    = mapped_column(String(64))
    action: Mapped[str]          = mapped_column(String(64))
    meta_data: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=None)  # fixed

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="feature_usage")

class RecoveryPlan(Base):
    __tablename__ = "recovery_plans"

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    title: Mapped[str]           = mapped_column(String(128))
    content: Mapped[str]         = mapped_column(Text)
    active: Mapped[bool]         = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="recovery_plans")

class RulebookVote(Base):
    __tablename__ = "rulebook_votes"

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    rule_key: Mapped[str]        = mapped_column(String(128))
    vote_value: Mapped[str]      = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="rulebook_votes")

class SimulatorLog(Base):
    __tablename__ = "simulator_logs"

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    action: Mapped[str]          = mapped_column(String(64))
    meta_data: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=None)  # fixed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="simulator_logs")

class AIInteraction(Base):
    __tablename__ = "ai_interactions"

    id: Mapped[int]              = mapped_column(primary_key=True)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    question: Mapped[str]        = mapped_column(Text)
    response: Mapped[str]        = mapped_column(Text)
    followed: Mapped[Optional[bool]] = mapped_column(Boolean, default=None)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="ai_interactions")

# ---------- Helpers ----------
async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session




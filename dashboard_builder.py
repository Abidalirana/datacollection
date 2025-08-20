# dashboard_builder.py
from __future__ import annotations

import os
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from db import (
    User, Trade, Journal, Emotion, Session,
    ResetChallenge, FeatureUsage
)

DASH_DIR = Path(os.getenv("DASHBOARD_DIR", "dashboards"))
DASH_DIR.mkdir(parents=True, exist_ok=True)

def _to_df(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

async def _fetch_user(db: AsyncSession, user_id: int) -> Optional[User]:
    res = await db.execute(select(User).where(User.id == user_id))
    return res.scalars().first()

async def _fetch_trades(db: AsyncSession, user_id: int) -> List[Dict[str, Any]]:
    res = await db.execute(
        select(Trade).where(Trade.user_id == user_id).order_by(Trade.created_at.asc())
    )
    trades = res.scalars().all()
    out = []
    for t in trades:
        out.append({
            "id": t.id,
            "symbol": t.symbol,
            "pnl": float(t.pnl) if t.pnl is not None else 0.0,
            "emotion_snapshot": t.emotion_snapshot,
            "note": t.note or "",
            "strategy": t.strategy,
            "risk_reward": t.risk_reward,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "max_drawdown": t.max_drawdown,
            "session_length_min": t.session_length_min,
            "outcome": t.outcome,
            "created_at": t.created_at,
        })
    return out

async def _fetch_journals(db: AsyncSession, user_id: int) -> List[Dict[str, Any]]:
    res = await db.execute(
        select(Journal).where(Journal.user_id == user_id).order_by(Journal.created_at.asc())
    )
    journals = res.scalars().all()
    return [{
        "id": j.id,
        "text": j.text,
        "confidence": j.confidence,
        "created_at": j.created_at,
    } for j in journals]

async def _fetch_emotions(db: AsyncSession, user_id: int) -> List[Dict[str, Any]]:
    res = await db.execute(
        select(Emotion).where(Emotion.user_id == user_id).order_by(Emotion.created_at.asc())
    )
    emos = res.scalars().all()
    return [{
        "id": e.id,
        "trade_id": e.trade_id,
        "tag": e.tag,
        "intensity": e.intensity,
        "note": e.note,
        "created_at": e.created_at,
    } for e in emos]

async def _fetch_sessions(db: AsyncSession, user_id: int) -> List[Dict[str, Any]]:
    res = await db.execute(
        select(Session).where(Session.user_id == user_id).order_by(Session.start_time.asc())
    )
    sessions = res.scalars().all()
    return [{
        "id": s.id,
        "start_time": s.start_time,
        "end_time": s.end_time,
        "total_trades": s.total_trades,
        "notes": s.notes,
        "created_at": s.created_at,
    } for s in sessions]

async def _fetch_feature_usage(db: AsyncSession, user_id: int) -> List[Dict[str, Any]]:
    res = await db.execute(
        select(FeatureUsage).where(FeatureUsage.user_id == user_id).order_by(FeatureUsage.created_at.asc())
    )
    fes = res.scalars().all()
    return [{
        "id": f.id,
        "feature_name": f.feature_name,
        "action": f.action,
        "metadata": f.metadata,
        "created_at": f.created_at,
    } for f in fes]

async def _fetch_reset_latest(db: AsyncSession, user_id: int) -> Optional[Dict[str, Any]]:
    res = await db.execute(
        select(ResetChallenge)
        .where(ResetChallenge.user_id == user_id)
        .order_by(ResetChallenge.created_at.desc())
        .limit(1)
    )
    rc = res.scalars().first()
    if not rc:
        return None
    return {
        "name": rc.name,
        "day_target": rc.day_target,
        "days_completed": rc.days_completed,
        "success": rc.success,
        "started_at": rc.started_at,
        "ended_at": rc.ended_at,
        "created_at": rc.created_at,
    }

def _kpis_from_trades(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "trade_count": 0,
            "total_pnl": 0.0,
            "win_rate": None,
            "avg_rr": None,
            "max_drawdown": None,
            "most_traded": [],
        }
    trade_count = len(df)
    total_pnl = float(df["pnl"].sum())

    # win_rate
    if "outcome" in df.columns and not df["outcome"].isna().all():
        wins = (df["outcome"].str.lower() == "win").sum()
        win_rate = round(100.0 * wins / trade_count, 2)
    else:
        # infer wins from pnl > 0
        wins = (df["pnl"] > 0).sum()
        win_rate = round(100.0 * wins / trade_count, 2)

    # avg_rr
    avg_rr = float(df["risk_reward"].dropna().mean()) if "risk_reward" in df.columns else None

    # max_drawdown (on cumulative pnl)
    cum = df["pnl"].cumsum()
    running_max = cum.cummax()
    dd = (cum - running_max)
    max_dd = float(dd.min()) if not dd.empty else 0.0

    # most traded symbols
    most_traded = []
    if "symbol" in df.columns:
        most_traded = (
            df["symbol"].value_counts()
              .head(5)
              .reset_index()
              .rename(columns={"index": "symbol", "symbol": "count"})
              .to_dict("records")
        )

    return {
        "trade_count": trade_count,
        "total_pnl": round(total_pnl, 2),
        "win_rate": win_rate,
        "avg_rr": round(avg_rr, 2) if avg_rr is not None and not math.isnan(avg_rr) else None,
        "max_drawdown": round(max_dd, 2),
        "most_traded": most_traded,
    }

def _fig_pnl_curve(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="No trades yet", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    df2 = df.copy()
    df2["cum_pnl"] = df2["pnl"].cumsum()
    fig.add_trace(go.Scatter(
        x=df2["created_at"], y=df2["cum_pnl"], mode="lines+markers", name="Cumulative PnL"
    ))
    fig.update_layout(title="Cumulative PnL", xaxis_title="Time", yaxis_title="PnL")
    return fig

def _fig_winrate(df: pd.DataFrame, window: int = 20) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="No trades yet", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    df2 = df.copy()
    if "outcome" in df2.columns and not df2["outcome"].isna().all():
        df2["win"] = (df2["outcome"].str.lower() == "win").astype(int)
    else:
        df2["win"] = (df2["pnl"] > 0).astype(int)
    df2["rolling_win_rate"] = df2["win"].rolling(window).mean() * 100.0
    fig.add_trace(go.Scatter(
        x=df2["created_at"], y=df2["rolling_win_rate"], mode="lines", name=f"Rolling Win% ({window})"
    ))
    fig.update_layout(title=f"Rolling Win Rate (window={window})", xaxis_title="Time", yaxis_title="Win %")
    return fig

def _fig_emotion_counts(emo_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if emo_df.empty:
        fig.add_annotation(text="No emotions logged", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    counts = emo_df["tag"].value_counts()
    fig.add_trace(go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), name="Emotions"))
    fig.update_layout(title="Emotion Tags (Counts)", xaxis_title="Emotion", yaxis_title="Count")
    return fig

def _fig_confidence_over_time(j_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if j_df.empty or j_df["confidence"].dropna().empty:
        fig.add_annotation(text="No confidence data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    df2 = j_df.dropna(subset=["confidence"]).copy()
    fig.add_trace(go.Scatter(x=df2["created_at"], y=df2["confidence"], mode="lines+markers", name="Confidence"))
    fig.update_layout(title="Journal Confidence Over Time", xaxis_title="Time", yaxis_title="Confidence (0-100)")
    return fig

def _fig_feature_usage(f_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if f_df.empty:
        fig.add_annotation(text="No feature usage", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    top = (f_df.groupby("feature_name")["id"].count()
           .sort_values(ascending=False).head(10))
    fig.add_trace(go.Bar(x=top.index.tolist(), y=top.values.tolist(), name="Feature Usage"))
    fig.update_layout(title="Top Feature Usage", xaxis_title="Feature", yaxis_title="Events")
    return fig

def _fig_session_heatmap(s_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if s_df.empty or s_df["start_time"].dropna().empty:
        fig.add_annotation(text="No session data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    df = s_df.copy()
    df["weekday"] = df["start_time"].dt.day_name()
    df["hour"] = df["start_time"].dt.hour
    pivot = df.pivot_table(index="weekday", columns="hour", values="id", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.astype(str), y=pivot.index, coloraxis="coloraxis"
    ))
    fig.update_layout(title="Trading Sessions Heatmap (Start Times)", xaxis_title="Hour", yaxis_title="Weekday",
                      coloraxis={"colorscale": "Blues"})
    return fig

def _fig_reset_gauge(reset: Optional[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not reset:
        fig.add_annotation(text="No reset challenge", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(title="Reset Challenge Progress")
        return fig
    target = max(1, int(reset["day_target"]))
    done = int(reset["days_completed"])
    pct = max(0, min(100, int(round(100.0 * done / target))))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": f"{reset['name']} (% Days Completed)"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    fig.update_layout(height=300)
    return fig

def _html_section(title: str, fig: go.Figure) -> str:
    return f"""
<section style="margin:24px 0;">
  <h2 style="font-family:Inter,system-ui,Arial;margin:8px 0;">{title}</h2>
  {fig.to_html(full_html=False, include_plotlyjs='cdn')}
</section>
"""

def _html_kpis(k: Dict[str, Any]) -> str:
    items = [
        ("Trades", k["trade_count"]),
        ("Total PnL", k["total_pnl"]),
        ("Win Rate %", k["win_rate"] if k["win_rate"] is not None else "—"),
        ("Avg R:R", k["avg_rr"] if k["avg_rr"] is not None else "—"),
        ("Max Drawdown", k["max_drawdown"] if k["max_drawdown"] is not None else "—"),
    ]
    most = ", ".join([f"{m['symbol']} ({m['count']})" for m in k.get("most_traded", [])]) or "—"
    grid = "".join([
        f"""<div style="padding:16px;border:1px solid #eee;border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
              <div style="font-size:12px;color:#666;">{label}</div>
              <div style="font-size:22px;font-weight:600;">{value}</div>
            </div>"""
        for (label, value) in items
    ])
    return f"""
<section style="margin:24px 0;">
  <h2 style="font-family:Inter,system-ui,Arial;margin:8px 0;">Key Metrics</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;">{grid}</div>
  <div style="margin-top:12px;font-size:14px;color:#333;"><b>Most Traded:</b> {most}</div>
</section>
"""

async def build_dashboard(db: AsyncSession, user_id: int) -> Path:
    """Builds an HTML dashboard file for the given user and returns its path."""
    user = await _fetch_user(db, user_id)
    trades = await _fetch_trades(db, user_id)
    journals = await _fetch_journals(db, user_id)
    emotions = await _fetch_emotions(db, user_id)
    sessions = await _fetch_sessions(db, user_id)
    feat = await _fetch_feature_usage(db, user_id)
    reset = await _fetch_reset_latest(db, user_id)

    # DataFrames
    tdf = _to_df(trades)
    jdf = _to_df(journals)
    edf = _to_df(emotions)
    sdf = _to_df(sessions)
    fdf = _to_df(feat)

    # Ensure datetime dtype
    for d in [tdf, jdf, edf, sdf]:
        for col in ["created_at", "entry_time", "exit_time", "start_time", "end_time"]:
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], errors="coerce", utc=False)

    # KPIs
    kpis = _kpis_from_trades(tdf)

    # Figures
    fig_pnl = _fig_pnl_curve(tdf)
    fig_wr  = _fig_winrate(tdf)
    fig_emo = _fig_emotion_counts(edf)
    fig_conf= _fig_confidence_over_time(jdf)
    fig_feat= _fig_feature_usage(fdf)
    fig_ses = _fig_session_heatmap(sdf)
    fig_gau = _fig_reset_gauge(reset)

    title = f"FundedFlow — Trader Dashboard (User #{user_id}{' • ' + user.name if user else ''})"
    head = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
</head>
<body style="margin:24px;font-family:Inter,system-ui,Arial;background:#fafafa;color:#111;">
  <header style="margin-bottom:8px;">
    <h1 style="margin:0;font-size:28px;">{title}</h1>
    <div style="color:#666;font-size:13px;">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
  </header>
  <hr style="border:none;border-top:1px solid #eee;margin:16px 0;" />
"""

    body = ""
    body += _html_kpis(kpis)
    body += _html_section("Reset Challenge Progress", fig_gau)
    body += _html_section("Cumulative PnL", fig_pnl)
    body += _html_section("Rolling Win Rate", fig_wr)
    body += _html_section("Emotion Tags (Counts)", fig_emo)
    body += _html_section("Journal Confidence Over Time", fig_conf)
    body += _html_section("Top Feature Usage", fig_feat)
    body += _html_section("Trading Sessions Heatmap", fig_ses)

    tail = """
  <footer style="margin-top:32px;color:#777;font-size:12px;">
    FundedFlow • Local Dashboard (Phase 1)
  </footer>
</body>
</html>
"""
    html = head + body + tail
    out_path = DASH_DIR / f"dashboard_user_{user_id}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path

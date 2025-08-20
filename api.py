# api.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db import get_db, create_tables, User, Trade
from agent import run_datacollector_agent
from dashboard_builder import build_dashboard

app = FastAPI(title="FundedFlow API")

# (Optional) CORS for local dev UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    # Auto-create tables (dev only)
    await create_tables()

@app.get("/", response_class=HTMLResponse)
async def root():
    idx = Path("index.html")
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>FundedFlow API is running</h3>")

# ------------------ Agent Tip ------------------
@app.get("/coach/{user_id}")
async def coach(user_id: int, db: AsyncSession = Depends(get_db)):
    tip = await run_datacollector_agent(user_id, db)
    return {"tip": tip}

# ------------------ Trades (for index.html) ------------------
@app.get("/trades/{user_id}")
async def trades(user_id: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(
        select(Trade).where(Trade.user_id == user_id).order_by(Trade.created_at.asc())
    )
    rows = res.scalars().all()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "pnl": float(t.pnl) if t.pnl is not None else 0.0,
            "emotion": t.emotion_snapshot,
            "note": t.note,
            "created_at": t.created_at.isoformat(),
        }
        for t in rows
    ]

@app.post("/save_trade")
async def save_trade(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
):
    required = ["user_id", "symbol", "pnl"]
    for k in required:
        if k not in payload:
            raise HTTPException(400, f"Missing field: {k}")

    user_id = int(payload["user_id"])
    symbol  = str(payload["symbol"]).strip()
    pnl     = float(payload["pnl"])
    emotion = str(payload.get("emotion") or "").strip() or None
    note    = str(payload.get("note") or "")

    # Make sure user exists (auto-create lightweight user for dev)
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalars().first()
    if not user:
        user = User(
            id=user_id,
            name=f"User {user_id}",
            email=f"user{user_id}@example.com",
        )
        db.add(user)
        await db.flush()

    t = Trade(
        user_id=user.id,
        symbol=symbol,
        pnl=pnl,
        emotion_snapshot=emotion,
        note=note,
    )
    db.add(t)
    await db.commit()
    return {"ok": True, "trade_id": t.id}

# ------------------ Dashboard ------------------
@app.get("/dashboard/{user_id}")
async def dashboard(user_id: int, db: AsyncSession = Depends(get_db)):
    path = await build_dashboard(db, user_id)
    if not path.exists():
        raise HTTPException(500, "Failed to build dashboard")
    return FileResponse(str(path), media_type="text/html", filename=path.name)

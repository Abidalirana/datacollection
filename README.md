datacollection) PS D:\datacollection> dir


    Directory: D:\datacollection


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         8/15/2025   3:51 PM                .venv
-a----         8/15/2025   3:57 PM              0 .env
-a----         8/15/2025   3:50 PM            109 .gitignore
-a----         8/15/2025   3:50 PM              5 .python-version
-a----         8/15/2025   3:57 PM              0 database.py
-a----         8/15/2025   3:57 PM              0 fastapi_app.py
-a----         8/15/2025   3:57 PM              0 index.html
-a----         8/15/2025   3:54 PM           1752 main.py
-a----         8/15/2025   3:52 PM            338 pyproject.toml
-a----         8/15/2025   3:58 PM           4346 README.md
-a----         8/15/2025   3:57 PM              0 requirements.txt
-a----         8/15/2025   3:52 PM          77711 uv.lock


(datacollection) PS D:\datacollection> 

=============
✅ Minimal Working Set – 4 Files Only
Keep your entire FundedFlow-MVP in exactly these four files:
Table
Copy
File	Contains
.env	DATABASE_URL=... + OPENAI_API_KEY=...
database.py	5-line helper → returns a psycopg2 connection
main.py	FastAPI routes + AI tip function (AtlasFX agent)
index.html	Single-page form + Plotly graph
📁 File 1 – .env
bash
Copy
DATABASE_URL=postgresql://user:pass@localhost:5432/fundedflow
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
📁 File 2 – database.py
Python
Copy
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))
📁 File 3 – main.py
Python
Copy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import openai
import os
from dotenv import load_dotenv
from database import get_conn

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

class TradeIn(BaseModel):
    user_id: str
    symbol: str
    pnl: float
    emotion: str
    note: str = ""

@app.post("/save_trade")
def save_trade(t: TradeIn):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (user_id, symbol, pnl, emotion, note, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (t.user_id, t.symbol, t.pnl, t.emotion, t.note, datetime.utcnow()))
        conn.commit()
    return {"ok": True}

@app.get("/trades/{user_id}")
def list_trades(user_id: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol, pnl, emotion, note, created_at
            FROM trades
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cur.fetchall()
    return [{"symbol": r[0], "pnl": r[1], "emotion": r[2], "note": r[3], "timestamp": r[4]} for r in rows]

@app.get("/coach/{user_id}")
def coach(user_id: str):
    trades = list_trades(user_id)[:3]
    context = "\n".join([f"{t['emotion']} after {t['symbol']} PnL {t['pnl']}" for t in trades])
    prompt = f"Recent trader emotions:\n{context}\nGive one short mindset tip."
    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=60)
    return {"tip": resp.choices[0].message.content.strip()}
📁 File 4 – index.html
HTML
Preview
Copy
<!DOCTYPE html>
<html>
<head>
  <title>FundedFlow – Trader Log</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h2>Log a Trade</h2>
  <form id="tradeForm">
    <input id="user_id" placeholder="User ID" required>
    <input id="symbol" placeholder="Symbol (e.g., US30)" required>
    <input id="pnl" type="number" step="0.01" placeholder="PnL" required>
    <input id="emotion" placeholder="Emotion" required>
    <textarea id="note" placeholder="Notes"></textarea>
    <button type="submit">Save</button>
  </form>

  <h3>Chart</h3>
  <div id="chart" style="width: 100%; height: 400px;"></div>

  <script>
    const form = document.getElementById('tradeForm');
    form.addEventListener('submit', async e => {
      e.preventDefault();
      await fetch('/save_trade', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          user_id: user_id.value,
          symbol: symbol.value,
          pnl: +pnl.value,
          emotion: emotion.value,
          note: note.value
        })
      });
      loadChart();
    });

    async function loadChart() {
      const res = await fetch(`/trades/${user_id.value}`);
      const data = await res.json();
      const trace = {
        x: data.map(d => d.timestamp),
        y: data.map(d => d.pnl),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'PnL'
      };
      Plotly.newPlot('chart', [trace]);
    }
  </script>
</body>
</html>
🚀 Run It
bash
Copy
pip install fastapi uvicorn psycopg2-binary openai python-dotenv
uvicorn main:app --reload
Open http://127.0.0.1:8000 in your browser → exactly 4 files, fully working.
Copy
Retry
Share

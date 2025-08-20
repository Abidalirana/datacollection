# main.py
import asyncio
from fastapi import FastAPI
from db import create_tables

app = FastAPI(title="FundedFlow API")

# Import your routers here
# from routes import trades, coach, etc.
# app.include_router(trades.router)
# app.include_router(coach.router)

@app.on_event("startup")
async def startup_event():
    print("Creating database tables if not exist...")
    await create_tables()
    print("✅ Tables are ready!")

@app.get("/")
async def root():
    return {"message": "FundedFlow API running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




#   uvicorn main:app --reload

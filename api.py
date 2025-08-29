from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
import os
from dotenv import load_dotenv
from ai_project.data_collector.orchestrator_collector import run_data_collection

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Async engine and session
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI(title="Data Collection API")

# ✅ Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for DB session
async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

# Endpoint to log user data
@app.post("/log-data/")
async def log_user_data(data: dict, session: AsyncSession = Depends(get_session)):
    user_id = await run_data_collection(data, session)
    return {"message": "Data logged successfully", "user_id": user_id}


# =========================
# ✅ New endpoint: Ask RAG AI
# =========================
from run_pipeline import run_agent_with_query
@app.post("/ask/")
async def ask_ai(query: dict):
    """
    Example JSON payload: {"query": "show my last 10 trades"}
    """
    user_query = query.get("query")
    if not user_query:
        return {"error": "Query field is required"}
    
    response = await run_agent_with_query(user_query)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
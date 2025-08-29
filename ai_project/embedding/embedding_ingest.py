# embedding_ingest.py (fixed for persistence)

import sys
import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Load .env variables
load_dotenv()

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.models import User, Journal, Trade
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient  # ✅ use PersistentClient

# -------------------------------
# Initialize HF embedding model
# -------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# Initialize Chroma DB (persistent)
# -------------------------------
chroma_client = PersistentClient(path="./chroma_db")  # ✅ persistent path
collection = chroma_client.get_or_create_collection(name="embeddings")

# -------------------------------
# Async SQLAlchemy setup
# -------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# -------------------------------
# Ingest embedding function
# -------------------------------
async def ingest_embedding(user_id: int, source_type: str, source_id: int, content: str):
    embedding = model.encode(content).tolist()
    collection.add(
        ids=[f"{source_type}_{source_id}"],
        metadatas=[{
            "user_id": user_id,
            "source_type": source_type,
            "source_id": source_id,
            "content": content
        }],
        embeddings=[embedding]
    )
    print(f"✅ Saved embedding: {source_type} {source_id}")

# -------------------------------
# Ingest all journals
# -------------------------------
async def ingest_all_journals():
    async with SessionLocal() as session:
        result = await session.execute(text("SELECT * FROM journals"))
        journals = result.mappings().all()
        for j in journals:
            await ingest_embedding(
                user_id=j["user_id"],
                source_type="journal",
                source_id=j["id"],
                content=j["content"]
            )

# -------------------------------
# Ingest all trades
# -------------------------------
async def ingest_all_trades():
    async with SessionLocal() as session:
        result = await session.execute(text("SELECT * FROM trades"))
        trades = result.mappings().all()
        for t in trades:
            text_content = f"{t['strategy']} {t['instrument']}"
            await ingest_embedding(
                user_id=t["user_id"],
                source_type="trade",
                source_id=t["id"],
                content=text_content
            )

# -------------------------------
# Main runner
# -------------------------------
async def main():
    print("Starting ingestion...")
    await ingest_all_journals()
    await ingest_all_trades()
    print("✅ All embeddings ingested successfully!")

if __name__ == "__main__":
    asyncio.run(main())

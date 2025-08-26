import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from .models import Base

load_dotenv()  # load DATABASE_URL from .env

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable is not set.")

# Remove +asyncpg for sync engine
sync_db_url = DATABASE_URL.replace("+asyncpg", "")

# Create synchronous engine
engine = create_engine(sync_db_url, echo=True)

def create_db():
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully!")

# Session factory
SessionLocal = sessionmaker(bind=engine)

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    create_db()

from sqlalchemy import create_engine
from models import Base

# SQLite local DB (you can replace with PostgreSQL)
DATABASE_URL = "sqlite:///./fundedflow.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create all tables
Base.metadata.create_all(bind=engine)

print("✅ All tables created successfully!")

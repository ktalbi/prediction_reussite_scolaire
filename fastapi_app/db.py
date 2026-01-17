import os
from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = os.getenv("INFER_DB_URL", "sqlite:///./inference_logs.db")

engine = create_engine(
    DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.now(UTC), index=True)
    inputs_json = Column(JSON)
    predicted_G3 = Column(Float)
    decision = Column(String)
    model_version = Column(String, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)

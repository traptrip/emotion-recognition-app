import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# check_same_thread needed only for SQLite database
engine = create_engine(
    os.environ.get("SQLALCHEMY_DATABASE_URL", "sqlite:///db/emotion_app.db"),
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

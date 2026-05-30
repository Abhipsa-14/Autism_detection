import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, func,
)
from sqlalchemy.orm import relationship
from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    full_name     = Column(String(120), nullable=False)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    predictions = relationship("Prediction", back_populates="user", lazy="select")


class Prediction(Base):
    __tablename__ = "predictions"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # AQ-10 questionnaire answers (0 or 1)
    a1  = Column(Integer, nullable=False)
    a2  = Column(Integer, nullable=False)
    a3  = Column(Integer, nullable=False)
    a4  = Column(Integer, nullable=False)
    a5  = Column(Integer, nullable=False)
    a6  = Column(Integer, nullable=False)
    a7  = Column(Integer, nullable=False)
    a8  = Column(Integer, nullable=False)
    a9  = Column(Integer, nullable=False)
    a10 = Column(Integer, nullable=False)

    # Personal info
    age           = Column(Float, nullable=False)
    gender        = Column(String(1), nullable=False)   # 'm' or 'f'
    jaundice      = Column(Boolean, nullable=False)
    family_autism = Column(Boolean, nullable=False)

    # Model output
    prediction    = Column(Integer, nullable=False)     # 0 or 1
    confidence    = Column(Float, nullable=False)       # 0.0 – 1.0
    risk_level    = Column(String(20), nullable=False)  # Low / Moderate / High

    # AQ raw score (sum of answers, informational)
    aq_score      = Column(Integer, nullable=False)

    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="predictions")

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# ─── Auth ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ─── Questionnaire / Prediction ───────────────────────────────────────────────

class QuestionnaireInput(BaseModel):
    # AQ-10 answers (0 = no, 1 = yes)
    a1:  Literal[0, 1]
    a2:  Literal[0, 1]
    a3:  Literal[0, 1]
    a4:  Literal[0, 1]
    a5:  Literal[0, 1]
    a6:  Literal[0, 1]
    a7:  Literal[0, 1]
    a8:  Literal[0, 1]
    a9:  Literal[0, 1]
    a10: Literal[0, 1]

    # Personal info
    age:           float  = Field(..., gt=0, lt=120, description="Age in years")
    gender:        Literal["m", "f"]
    jaundice:      bool   = Field(..., description="History of jaundice at birth")
    family_autism: bool   = Field(..., description="Family member has autism")


class PredictionResult(BaseModel):
    id:            int
    prediction:    int          # 0 or 1
    confidence:    float        # probability 0.0 – 1.0
    risk_level:    str          # Low / Moderate / High
    aq_score:      int          # raw sum of A1-A10
    created_at:    datetime

    # Echo back the inputs for history display
    a1: int; a2: int; a3: int; a4: int; a5: int
    a6: int; a7: int; a8: int; a9: int; a10: int
    age: float
    gender: str
    jaundice: bool
    family_autism: bool

    model_config = {"from_attributes": True}


class PredictionHistory(BaseModel):
    total: int
    results: list[PredictionResult]

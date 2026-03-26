from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.db.models import User, Prediction
from app.schemas import QuestionnaireInput, PredictionResult, PredictionHistory
from app.core.security import get_current_user
from app.services.ml_service import model_service

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResult, status_code=status.HTTP_201_CREATED)
async def predict(
    payload: QuestionnaireInput,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await model_service.predict(
        a1=payload.a1, a2=payload.a2, a3=payload.a3, a4=payload.a4, a5=payload.a5,
        a6=payload.a6, a7=payload.a7, a8=payload.a8, a9=payload.a9, a10=payload.a10,
        age=payload.age,
        gender=payload.gender,
        jaundice=payload.jaundice,
        family_autism=payload.family_autism,
    )

    pred = Prediction(
        user_id=current_user.id,
        a1=payload.a1,   a2=payload.a2,   a3=payload.a3,
        a4=payload.a4,   a5=payload.a5,   a6=payload.a6,
        a7=payload.a7,   a8=payload.a8,   a9=payload.a9,
        a10=payload.a10,
        age=payload.age,
        gender=payload.gender,
        jaundice=payload.jaundice,
        family_autism=payload.family_autism,
        prediction=result["prediction"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        aq_score=result["aq_score"],
    )
    db.add(pred)
    await db.flush()
    await db.refresh(pred)

    return PredictionResult.model_validate(pred)


@router.get("/history", response_model=PredictionHistory)
async def history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
    )
    records = result.scalars().all()

    return PredictionHistory(
        total=len(records),
        results=[PredictionResult.model_validate(r) for r in records],
    )

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models import LearningPractice, LearningReviewItem, LearningWeakness
from app.schemas.learning import (
    LearningAssessmentCreate,
    LearningAssessmentResponse,
    LearningDashboardResponse,
    LearningPracticeResponse,
    LearningProfileResponse,
    LearningProfileUpsert,
    LearningReviewItemResponse,
    LearningWeaknessResponse,
)
from app.services.learning_service import (
    assessment_to_dict,
    build_dashboard,
    create_assessment,
    get_or_create_profile,
    practice_to_dict,
    profile_to_dict,
    resolve_user_id,
    review_to_dict,
    upsert_profile,
    weakness_to_dict,
)


router = APIRouter(prefix="/learning", tags=["Learning"])


@router.get("/profile", response_model=LearningProfileResponse)
def get_learning_profile(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)) -> LearningProfileResponse:
    user_id = resolve_user_id(db, current_user)
    return LearningProfileResponse(**profile_to_dict(get_or_create_profile(db, user_id)))


@router.put("/profile", response_model=LearningProfileResponse)
def update_learning_profile(
    body: LearningProfileUpsert,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LearningProfileResponse:
    user_id = resolve_user_id(db, current_user)
    profile = upsert_profile(db, user_id, body.model_dump())
    return LearningProfileResponse(**profile_to_dict(profile))


@router.get("/weaknesses", response_model=list[LearningWeaknessResponse])
def list_learning_weaknesses(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)) -> list[LearningWeaknessResponse]:
    user_id = resolve_user_id(db, current_user)
    rows = db.query(LearningWeakness).filter(LearningWeakness.user_id == user_id).order_by(LearningWeakness.severity.desc()).limit(50).all()
    return [LearningWeaknessResponse(**weakness_to_dict(item)) for item in rows]


@router.get("/practices", response_model=list[LearningPracticeResponse])
def list_learning_practices(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)) -> list[LearningPracticeResponse]:
    user_id = resolve_user_id(db, current_user)
    rows = db.query(LearningPractice).filter(LearningPractice.user_id == user_id).order_by(LearningPractice.created_at.desc()).limit(50).all()
    return [LearningPracticeResponse(**practice_to_dict(item)) for item in rows]


@router.post("/assessments", response_model=LearningAssessmentResponse)
def create_learning_assessment(
    body: LearningAssessmentCreate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LearningAssessmentResponse:
    user_id = resolve_user_id(db, current_user)
    item = create_assessment(db, user_id, body.model_dump())
    return LearningAssessmentResponse(**assessment_to_dict(item))


@router.get("/reviews", response_model=list[LearningReviewItemResponse])
def list_learning_reviews(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)) -> list[LearningReviewItemResponse]:
    user_id = resolve_user_id(db, current_user)
    rows = db.query(LearningReviewItem).filter(LearningReviewItem.user_id == user_id).order_by(LearningReviewItem.due_at.asc()).limit(50).all()
    return [LearningReviewItemResponse(**review_to_dict(item)) for item in rows]


@router.get("/dashboard", response_model=LearningDashboardResponse)
def get_learning_dashboard(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)) -> LearningDashboardResponse:
    user_id = resolve_user_id(db, current_user)
    return LearningDashboardResponse(**build_dashboard(db, user_id))

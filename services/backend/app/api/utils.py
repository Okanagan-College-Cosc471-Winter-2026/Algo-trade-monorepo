from fastapi import APIRouter

router = APIRouter()


@router.get("/health-check/")
def health_check() -> bool:
    return True

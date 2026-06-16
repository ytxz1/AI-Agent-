from fastapi import APIRouter

from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.app_env,
        embedding_provider=settings.embedding_provider,
        chat_provider=settings.chat_provider,
        vector_store_dir=str(settings.chroma_dir),
    )


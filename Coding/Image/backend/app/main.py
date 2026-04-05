from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.api.routes import router as api_router
from app.api.feedback import router as feedback_router
from app.security.middleware import SecurityHeadersMiddleware
from app.security.rate_limiter import limiter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.ml.vit_classifier import warmup_classifier
        from app.ml.face_detector import warmup_detector

        vit_ready = warmup_classifier()
        detector_ready = warmup_detector()
        logger.info(
            "Model warmup complete: vit_ready=%s detector_ready=%s",
            vit_ready,
            detector_ready,
        )
    except Exception as e:
        logger.warning(f"Model warmup skipped due to initialization error: {e}")

    logger.info("Axiom-I Image Forensics Backend started successfully.")
    yield
    logger.info("Axiom-I Image Forensics Backend shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(feedback_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "health": f"{settings.API_V1_STR}/health",
        "docs": "/docs",
    }

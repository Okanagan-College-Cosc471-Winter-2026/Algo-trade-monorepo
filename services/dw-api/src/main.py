from fastapi import FastAPI
from src.config import get_settings
from src.routers.ingestion import router as ingestion_router
from src.routers.historical_ingestion import router as historical_router

settings = get_settings()

app = FastAPI(
    title="DW Ingestion API",
    description="API for staging financial market data from DC team.",
    version="0.1.0",
)

app.include_router(ingestion_router)
app.include_router(historical_router)

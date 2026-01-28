import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from services.api.app.database import create_db_and_tables

# A static version for now, can be replaced with git hash later
APP_VERSION = "0.1.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at application startup
    create_db_and_tables()
    yield
    # Runs once at application shutdown (nothing to do yet)


app = FastAPI(
    title="Aegis-AI API",
    version=APP_VERSION,
    description="API for deepfake detection and analysis.",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/version")
def version():
    """Returns the application version."""
    return {"version": APP_VERSION}

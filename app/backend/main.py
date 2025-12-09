"""
SliceWise Backend API - Modular Architecture

This is the NEW main entry point for the SliceWise Backend API.
It replaces the monolithic main_v2.py with a clean, modular architecture.

Features:
- Modular service layer with dependency injection
- Organized routers for each endpoint group
- Centralized configuration management
- Global error handling
- Clean separation of concerns

Author: SliceWise Team
Version: 2.0.0 (Modular)
Date: December 8, 2025
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import configuration
from app.backend.config.settings import settings

# Import routers
from app.backend.routers import (
    health,
    classification,
    segmentation,
    multitask,
    patient
)

# Import middleware
from app.backend.middleware import setup_error_handlers

# Import model manager
from app.backend.services.model_loader import get_model_manager


# ============================================================================
# Lifespan Event Handler (Modern FastAPI Pattern)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    
    This replaces the deprecated @app.on_event("startup") and @app.on_event("shutdown").
    """
    # Startup
    print("\n" + "=" * 80)
    print("SliceWise API - Starting up...")
    print("=" * 80)
    
    # Get model manager and load all models
    model_manager = get_model_manager()
    model_manager.load_all_models()
    
    print("=" * 80)
    print("SliceWise API - Ready!")
    print(f"API Version: {settings.api.version}")
    print(f"Docs: http://{settings.api.host}:{settings.api.port}/docs")
    print("=" * 80 + "\n")
    
    yield  # Application runs here
    
    # Shutdown
    print("\n" + "=" * 80)
    print("SliceWise API - Shutting down...")
    print("=" * 80 + "\n")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=settings.api.cors_credentials,
    allow_methods=settings.api.cors_methods,
    allow_headers=settings.api.cors_headers,
)


# ============================================================================
# Error Handlers
# ============================================================================

setup_error_handlers(app)


# ============================================================================
# Include Routers
# ============================================================================

# Health and info endpoints
app.include_router(health.router)

# Classification endpoints
app.include_router(classification.router)

# Segmentation endpoints
app.include_router(segmentation.router)

# Multi-task endpoint
app.include_router(multitask.router)

# Patient analysis endpoint
app.include_router(patient.router)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Run the application with uvicorn.
    
    This is used for development. In production, use:
        uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
    """
    uvicorn.run(
        "app.backend.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.api.log_level
    )

"""
Error handling middleware for SliceWise Backend.

This module provides global error handlers for the FastAPI application,
including HTTP exceptions, validation errors, and general exceptions.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
from datetime import datetime
from typing import Union


# ============================================================================
# Error Response Formatting
# ============================================================================

def format_error_response(
    status_code: int,
    error_type: str,
    message: str,
    details: Union[str, dict, None] = None
) -> dict:
    """
    Format error response with consistent structure.
    
    Args:
        status_code: HTTP status code
        error_type: Type of error (e.g., "ValidationError", "HTTPException")
        message: Error message
        details: Optional additional details
    
    Returns:
        Dictionary with formatted error response
    """
    response = {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    return response


# ============================================================================
# Exception Handlers
# ============================================================================

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Args:
        request: FastAPI request object
        exc: HTTP exception
    
    Returns:
        JSON response with formatted error
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(
            status_code=exc.status_code,
            error_type="HTTPException",
            message=str(exc.detail),
            details={
                "path": str(request.url),
                "method": request.method
            }
        )
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors.
    
    Args:
        request: FastAPI request object
        exc: Validation error
    
    Returns:
        JSON response with formatted validation errors
    """
    return JSONResponse(
        status_code=422,
        content=format_error_response(
            status_code=422,
            error_type="ValidationError",
            message="Request validation failed",
            details={
                "path": str(request.url),
                "method": request.method,
                "errors": exc.errors()
            }
        )
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions.
    
    Args:
        request: FastAPI request object
        exc: General exception
    
    Returns:
        JSON response with formatted error
    """
    # Log the full traceback for debugging
    error_traceback = traceback.format_exc()
    print(f"\n{'='*80}")
    print(f"UNHANDLED EXCEPTION")
    print(f"{'='*80}")
    print(f"Path: {request.url}")
    print(f"Method: {request.method}")
    print(f"Exception: {type(exc).__name__}")
    print(f"Message: {str(exc)}")
    print(f"\nTraceback:")
    print(error_traceback)
    print(f"{'='*80}\n")
    
    return JSONResponse(
        status_code=500,
        content=format_error_response(
            status_code=500,
            error_type=type(exc).__name__,
            message="Internal server error",
            details={
                "path": str(request.url),
                "method": request.method,
                "exception": str(exc)
            }
        )
    )


# ============================================================================
# Setup Function
# ============================================================================

def setup_error_handlers(app: FastAPI) -> None:
    """
    Setup all error handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # General exceptions (catch-all)
    app.add_exception_handler(Exception, general_exception_handler)
    
    print("[OK] Error handlers configured")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Error handler middleware:")
    print("  - HTTP exception handler")
    print("  - Validation exception handler")
    print("  - General exception handler")
    print("  - Structured error responses")

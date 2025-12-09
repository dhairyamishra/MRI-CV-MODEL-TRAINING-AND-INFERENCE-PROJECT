"""
Middleware for error handling and request processing.

This module contains middleware components for the FastAPI application.
"""

from .error_handler import setup_error_handlers

__all__ = ["setup_error_handlers"]

"""
mdms_simulator.api module.
Exposes FastAPI application and route definitions.
"""

from mdms_simulator.api.app import app, create_app
from mdms_simulator.api.routes import router

__all__ = ["app", "create_app", "router"]

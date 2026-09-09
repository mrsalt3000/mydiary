"""HTTP API layer: routers, Pydantic schemas and view adapters.

Routers are thin: HTTP validation → domain service call → response mapping.
No business logic and no direct SQLAlchemy access here (see the
``backend-fastapi`` skill and the architecture brief §4).
"""

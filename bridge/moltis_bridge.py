"""
bridge/moltis_bridge.py
=======================
Thin compatibility shim and uvicorn entry point.

The bridge has been split into focused modules:
  bridge/client.py   — MoltisConfig + MoltisClient (HTTP/GraphQL)
  bridge/schemas.py  — Pydantic request/response models
  bridge/service.py  — AndrewMoltisBridge orchestration
  bridge/hitl.py     — Human-in-the-loop gate
  bridge/api.py      — FastAPI app, endpoints, rate limiting, utils, CLI

Existing code that imports directly from bridge.moltis_bridge will continue
to work via the re-exports below.
"""

# Re-exports for backward compatibility
from bridge.client import MoltisConfig, MoltisClient  # noqa: F401
from bridge.schemas import (  # noqa: F401
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    ScheduleRequest,
)
from bridge.service import AndrewMoltisBridge  # noqa: F401
from bridge.api import (  # noqa: F401
    app,
    get_bridge,
    generate_moltis_hook_config,
    generate_docker_compose,
    main,
)

if __name__ == "__main__":
    main()

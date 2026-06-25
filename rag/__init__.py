"""
PersonaPlex RAG research framework.

Importing this top-level package only pulls in lightweight, dependency-free submodules
(`config`, `turn_detector`, `injection_manager`). Heavier pieces (vector stores, embedding
models) live in their own submodules and are imported lazily by whatever wires them up, so
that `ENABLE_RAG=False` never pays for, or risks breaking on, those dependencies.
"""

from .config import InjectionMode, RAGConfig
from .injection_manager import InjectionRequest, InjectionStats, InjectionJob, TokenInjector
from .turn_detector import TurnBoundaryDetector, TurnDetectorConfig

__all__ = [
    "InjectionMode",
    "RAGConfig",
    "InjectionRequest",
    "InjectionStats",
    "InjectionJob",
    "TokenInjector",
    "TurnBoundaryDetector",
    "TurnDetectorConfig",
]

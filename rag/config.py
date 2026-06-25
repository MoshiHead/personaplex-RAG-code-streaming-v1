"""
Central configuration for the PersonaPlex RAG research framework.

Importing this module has zero side effects and zero heavy dependencies (no torch, faiss,
sentence-transformers, etc.), so it is always safe to import even when ENABLE_RAG=False --
this is what lets the RunPod notebook expose RAG configuration variables unconditionally
without affecting baseline PersonaPlex startup.

See docs/ARCHITECTURE_REPORT.md (Section 6) and docs/STREAMING_AND_INJECTION_DESIGN.md for the
reasoning behind each mode.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import os


class InjectionMode(str, Enum):
    """Injection strategies under research. String-valued so notebook widgets / env vars / JSON
    logs can use plain strings without an extra encode/decode step."""

    BASELINE = "baseline"               # Mode A -- no RAG, pure PersonaPlex. Always supported.
    PROMPT_RAG = "prompt_rag"           # Mode B -- negative-control baseline: naive "Relevant
                                         # Knowledge: ... User Question: ..." block. Expected to
                                         # underperform Mode C; kept to *measure* that gap, not to
                                         # be tuned into working well.
    PERSONA_RAG = "persona_rag"         # Mode C -- knowledge folded into the same <system>...<system>
                                         # mechanism PersonaPlex uses for its own persona prompt.
    TURN_INJECTION = "turn_injection"   # Mode D -- inject once per detected end-of-user-turn.
    DYNAMIC_RUNTIME = "dynamic_runtime" # Mode E -- inject repeatedly on a fixed interval throughout
                                         # the call.
    CACHE_AWARE = "cache_aware"         # Mode F -- same TokenInjector primitive as C/D/E, benchmarked
                                         # against a naive "reset and replay" baseline to quantify the
                                         # cost of *not* preserving the live RingKVCache.


# Modes whose intended policy is "inject right after the user stops talking" -- these are the
# modes that benefit from (but, per RAGConfig.validate(), do not strictly require) turn-boundary
# detection from rag.turn_detector.
_MODES_USING_TURN_DETECTION = {InjectionMode.TURN_INJECTION}

_KNOWN_VECTOR_DBS = ("faiss", "chroma")
_KNOWN_EMBEDDING_MODELS = ("bge-small", "bge-base", "bge-large", "e5-small", "e5-base", "e5-large",
                           "sentence-transformers")


@dataclass
class RAGConfig:
    """One object capturing every knob the notebook / benchmark harness needs to set.

    Defaults reproduce baseline PersonaPlex behavior (enable_rag=False), so
    `RAGConfig()` is always a safe, no-op default.
    """

    enable_rag: bool = False
    injection_mode: InjectionMode = InjectionMode.BASELINE
    top_k: int = 5
    embedding_model: str = "bge-small"
    vector_db: str = "faiss"
    benchmark_mode: bool = False

    # Modes D/E specific knobs.
    vad_enabled: bool = False
    dynamic_injection_interval_s: float = 30.0  # only consulted by DYNAMIC_RUNTIME
    # Mode D re-injects on every detected turn boundary, so its per-injection token count must
    # stay small relative to `top_k` -- Mode C's own benchmark showed ~25ms/injected token, so a
    # 5-document block (340 tokens, as used by B/C) costs ~8.5s per injection, far too slow to
    # repeat every time the user pauses. Deliberately defaults much smaller than `top_k`.
    turn_injection_top_k: int = 2

    # Retrieval-layer knobs (consumed by rag.retriever once implemented).
    score_threshold: float | None = None

    # Where per-request logs (Phase 9) and benchmark reports (Phase 8) get written.
    log_dir: str = "rag_logs"

    def validate(self) -> list[str]:
        """Returns human-readable warnings; never raises. Designed to be called from a notebook
        cell and printed, rather than crashing a `Run All` over a config typo."""
        warnings: list[str] = []

        if not self.enable_rag and self.injection_mode != InjectionMode.BASELINE:
            warnings.append(
                "ENABLE_RAG is False but INJECTION_MODE is "
                f"'{self.injection_mode.value}' (!= 'baseline'); INJECTION_MODE will be ignored "
                "until ENABLE_RAG=True."
            )

        if self.injection_mode in _MODES_USING_TURN_DETECTION and not self.vad_enabled:
            warnings.append(
                f"INJECTION_MODE='{self.injection_mode.value}' is designed to trigger on detected "
                "end-of-turn boundaries, but VAD_ENABLED=False. With no boundary signal, this mode "
                "will never fire -- either set VAD_ENABLED=True or switch to 'dynamic_runtime' "
                "(fixed-interval injection)."
            )

        if self.top_k <= 0:
            warnings.append(f"TOP_K should be a positive integer, got {self.top_k}.")

        if self.turn_injection_top_k <= 0:
            warnings.append(
                f"turn_injection_top_k should be a positive integer, got {self.turn_injection_top_k}."
            )
        if self.injection_mode == InjectionMode.TURN_INJECTION and self.turn_injection_top_k > 3:
            warnings.append(
                f"turn_injection_top_k={self.turn_injection_top_k} is large for a per-turn "
                "re-injection -- Mode C's benchmark measured ~25ms per injected token, so "
                "5 documents (~340 tokens) cost ~8.5s per injection. Consider keeping this small "
                "(1-2) so repeated mid-conversation injections don't stall the live audio."
            )

        if self.vector_db not in _KNOWN_VECTOR_DBS:
            warnings.append(
                f"Unknown VECTOR_DB '{self.vector_db}'; expected one of {_KNOWN_VECTOR_DBS}."
            )

        if self.embedding_model not in _KNOWN_EMBEDDING_MODELS:
            warnings.append(
                f"Unrecognized EMBEDDING_MODEL '{self.embedding_model}'; expected one of "
                f"{_KNOWN_EMBEDDING_MODELS}. It may still work if it's a valid model name/path for "
                "the configured embedding backend, but double-check for typos."
            )

        if self.dynamic_injection_interval_s <= 0:
            warnings.append(
                "DYNAMIC_INJECTION_INTERVAL_S should be positive, got "
                f"{self.dynamic_injection_interval_s}."
            )

        return warnings

    def as_dict(self) -> dict:
        d = asdict(self)
        d["injection_mode"] = self.injection_mode.value
        return d

    def describe(self) -> str:
        """Human-readable summary, e.g. for printing at the top of a notebook cell or a log file."""
        lines = [f"{key} = {value!r}" for key, value in self.as_dict().items()]
        warnings = self.validate()
        if warnings:
            lines.append("")
            lines.append("WARNINGS:")
            lines.extend(f"  - {w}" for w in warnings)
        return "\n".join(lines)

    @classmethod
    def from_env(cls, prefix: str = "PERSONAPLEX_RAG_") -> "RAGConfig":
        """Build a config from environment variables, e.g. PERSONAPLEX_RAG_ENABLE_RAG=1. Useful for
        driving the same config from a notebook cell (via os.environ) or a shell script identically."""

        def _get(name: str, default, cast=str):
            raw = os.environ.get(prefix + name)
            if raw is None:
                return default
            if cast is bool:
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return cast(raw)

        return cls(
            enable_rag=_get("ENABLE_RAG", False, bool),
            injection_mode=InjectionMode(_get("INJECTION_MODE", InjectionMode.BASELINE.value)),
            top_k=_get("TOP_K", 5, int),
            embedding_model=_get("EMBEDDING_MODEL", "bge-small"),
            vector_db=_get("VECTOR_DB", "faiss"),
            benchmark_mode=_get("BENCHMARK_MODE", False, bool),
            vad_enabled=_get("VAD_ENABLED", False, bool),
            dynamic_injection_interval_s=_get("DYNAMIC_INJECTION_INTERVAL_S", 30.0, float),
            log_dir=_get("LOG_DIR", "rag_logs"),
        )

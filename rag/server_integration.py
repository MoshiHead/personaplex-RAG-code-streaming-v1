"""
Glue layer between the moshi-agnostic rag/ primitives (TokenInjector, Retriever, RequestLogger)
and a live PersonaPlex connection (moshi.server.ServerState.handle_chat) or a single offline run
(moshi.offline.run_inference).

This is the *only* rag/ module that is written with moshi's concrete runtime objects in mind
(a real `LMGen`, a real SentencePiece tokenizer, the zero/sine frame factories) -- every other
module in this package (config, retriever, embeddings, vector_store, injection_manager,
turn_detector) stays moshi-agnostic and importable/testable without moshi installed. This module
has no import-time dependency on moshi either (it only needs *objects* satisfying the protocols
`TokenInjector` already defines), so it remains independently unit-testable with plain stand-ins
-- see rag/tests/test_server_integration.py.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .config import InjectionMode, RAGConfig
from .injection_manager import InjectionJob, InjectionRequest, TokenInjector, wrap_with_system_tags
from .logging_utils import RequestLogger, RequestLogRecord, inspect_kv_cache
from .retriever import Retriever


class RAGSession:
    """One instance per live connection (server.py) or per single run (offline.py). Owns:

      - the `Retriever` (built once at construction, reused across every call on this session)
      - the `TokenInjector` (built from this connection's live `lm_gen`/tokenizer)
      - the `RequestLogger` (writes to `config.log_dir`)
      - any pending incremental `InjectionJob` (for Modes D/E/F, consumed from `opus_loop`, one
        forced step per tick -- see `consume_one_tick()`)

    Concurrency contract: identical to `TokenInjector`'s (docs/STREAMING_AND_INJECTION_DESIGN.md,
    Section 3.1) -- every method on this class must only be called from the single coroutine that
    already owns `lm_gen.step()` for this connection. This class adds no locking of its own for
    the same reason `TokenInjector` doesn't: a lock here would wrongly imply concurrent use from a
    second task is safe to wait for, when it is actually a correctness bug to attempt at all.
    """

    def __init__(
        self,
        config: RAGConfig,
        lm_gen: Any,
        text_tokenizer: Any,
        make_zero_audio_frame,
        make_silence_audio_frame,
        index_path: Optional[str] = None,
    ):
        self.config = config
        self._lm_gen = lm_gen
        self.injector = TokenInjector(lm_gen, text_tokenizer, make_zero_audio_frame, make_silence_audio_frame)
        self.logger = RequestLogger(config.log_dir)

        self.retriever: Optional[Retriever] = None
        if config.enable_rag and index_path:
            self.retriever = Retriever(embedding_model=config.embedding_model, vector_db=config.vector_db)
            self.retriever.load_index(index_path)

        self.pending_job: Optional[InjectionJob] = None

    # ---- Mode C: connection-start / session-start injection --------------------------------
    def inject_persona_compatible_knowledge(self, query: str) -> dict:
        """Mode C. Retrieves context for `query`, folds it into the *same* `<system>...<system>`
        mechanism PersonaPlex's own persona prompt uses, and pushes it through the live model via
        one blocking `TokenInjector` burst.

        Intended call site: right after `lm_gen.step_system_prompts_async(...)` completes, while
        still inside the connection's `async with self.lock:` block -- i.e. before
        `opus_loop`/`recv_loop`/`send_loop` start. See
        docs/STREAMING_AND_INJECTION_DESIGN.md Section 3 for why that is the only safe place, and
        Phase 1's architecture report Section 6 for why Mode C is, by definition, a
        connection-start mechanism rather than a per-utterance one (PersonaPlex has no live
        speech-to-text of the user's audio to retrieve against mid-call -- see the Phase 2
        implementation report for this finding in full).

        Returns the full log record as a dict (also appended to the JSONL log), so the caller can
        print/inspect it immediately.
        """
        record = RequestLogRecord(mode=InjectionMode.PERSONA_RAG.value, user_query=query)

        if not self.config.enable_rag or self.retriever is None:
            record.injection_strategy = "skipped (RAG disabled or no index loaded)"
            self.logger.log(record)
            return record.to_dict()

        t0 = time.monotonic()
        retrieval = self.retriever.retrieve_context(
            query, top_k=self.config.top_k, score_threshold=self.config.score_threshold
        )
        retrieval_latency_s = time.monotonic() - t0
        record.retrieval_latency_s = retrieval_latency_s
        record.retrieved_contexts = retrieval["contexts"]
        record.retrieved_scores = retrieval["scores"]

        if not retrieval["contexts"]:
            record.injection_strategy = "skipped (no contexts above score_threshold)"
            self.logger.log(record)
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        request = InjectionRequest(
            text=knowledge_block, mode=InjectionMode.PERSONA_RAG.value, wrap_system_tags=True
        )

        t1 = time.monotonic()
        stats = self.injector.run_to_completion(request)
        injection_latency_s = time.monotonic() - t1

        record.injection_strategy = "persona_rag (blocking burst, same <system> mechanism as persona prompt)"
        record.prompt_length_chars = len(wrap_with_system_tags(knowledge_block))
        record.context_length_chars = len(knowledge_block)
        record.injected_token_count = stats.token_count
        record.injection_latency_s = injection_latency_s
        record.total_latency_s = retrieval_latency_s + injection_latency_s
        record.kv_cache_status = inspect_kv_cache(self._lm_gen)

        self.logger.log(record)
        return record.to_dict()

    # ---- Modes D/E/F: incremental, per-tick consumption from opus_loop ---------------------
    def queue_injection(self, request: InjectionRequest) -> None:
        """Queue an `InjectionRequest` for incremental consumption. Must be called from the same
        coroutine that will later call `consume_one_tick()` (i.e. `opus_loop`'s own execution
        context) -- see `TokenInjector`'s class docstring. Not used by Mode C (which injects in one
        blocking burst before `opus_loop` even starts); reserved for Modes D/E/F."""
        self.pending_job = self.injector.start(request)

    def consume_one_tick(self) -> bool:
        """Call once per `opus_loop` iteration, BEFORE draining real audio for that tick (see
        docs/STREAMING_AND_INJECTION_DESIGN.md Section 3.2). Executes at most one forced step.

        Returns True if a step was executed. Safe to call even when there is no pending job
        (no-op, returns False) -- this is what makes the `opus_loop` hook inert for
        `ENABLE_RAG=False` and for Mode C (which never populates `pending_job`).
        """
        if self.pending_job is None:
            return False

        executed = self.pending_job.step_once()
        if self.pending_job.done:
            self.logger.log(
                RequestLogRecord(
                    mode=self.pending_job.request.mode,
                    injection_strategy="incremental (per-tick, opus_loop)",
                    injected_token_count=self.pending_job.stats.token_count,
                    injection_latency_s=self.pending_job.stats.wall_time_s,
                    kv_cache_status=inspect_kv_cache(self._lm_gen),
                )
            )
            self.pending_job = None
        return executed

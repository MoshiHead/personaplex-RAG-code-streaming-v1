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
from .turn_detector import TurnBoundaryDetector


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

        # Mode D (turn injection) state. `turn_detector` is only constructed when both
        # TURN_INJECTION is selected and VAD is explicitly enabled (RAGConfig.validate() warns
        # otherwise) -- with no detector, `observe_user_frame()` below is a guaranteed no-op, so
        # this never affects any other mode. `_turn_injection_request` holds the knowledge
        # prepared once by `prepare_turn_injection_knowledge()`, re-injected on every detected
        # turn boundary -- see that method's docstring for why this is a *prepared, fixed* block
        # rather than a fresh per-turn retrieval (PersonaPlex has no ASR; there is no new query
        # text to retrieve against mid-call -- see docs/MODE_C_IMPLEMENTATION_REPORT.md Section 2).
        self.turn_detector: Optional[TurnBoundaryDetector] = None
        if config.enable_rag and config.injection_mode == InjectionMode.TURN_INJECTION and config.vad_enabled:
            self.turn_detector = TurnBoundaryDetector()
        self._turn_injection_request: Optional[InjectionRequest] = None
        self._turn_injection_query: Optional[str] = None

        # Mode E (dynamic/periodic injection) state. Unlike Mode D, this needs no turn-boundary
        # signal at all -- it re-fires on a fixed wall-clock interval regardless of conversational
        # state, so `_dynamic_injection_last_fire_time` (set once by
        # `prepare_dynamic_injection_knowledge`, reset by `tick_dynamic_injection`) is the only
        # piece of state it needs. See the Mode E section below for why its prepared knowledge is
        # deliberately NOT wrapped in `<system>...<system>` tags, unlike Mode C/D.
        self._dynamic_injection_request: Optional[InjectionRequest] = None
        self._dynamic_injection_query: Optional[str] = None
        self._dynamic_injection_last_fire_time: Optional[float] = None

    # ---- Shared retrieval step for any connection-start mode (B or C) -----------------------
    def _retrieve_for_injection(self, query: str, mode: str) -> tuple[RequestLogRecord, Optional[dict]]:
        """Runs retrieval and builds the common prefix of a log record for any connection-start
        injection mode. Returns `(record, None)` if the caller should stop immediately (RAG
        disabled, no index, or nothing retrieved above `score_threshold`) -- `record` already
        explains why via `injection_strategy`. Returns `(record, retrieval_dict)` otherwise, where
        `retrieval_dict` is `Retriever.retrieve_context`'s `{"query", "contexts", "scores"}`.

        Both Mode B and Mode C retrieve identically -- the *only* thing that should differ between
        them is how the retrieved text gets formatted before injection. Sharing this step is what
        guarantees that property rather than relying on two copy-pasted implementations staying in
        sync by hand.
        """
        record = RequestLogRecord(mode=mode, user_query=query)

        if not self.config.enable_rag or self.retriever is None:
            record.injection_strategy = "skipped (RAG disabled or no index loaded)"
            return record, None

        t0 = time.monotonic()
        retrieval = self.retriever.retrieve_context(
            query, top_k=self.config.top_k, score_threshold=self.config.score_threshold
        )
        record.retrieval_latency_s = time.monotonic() - t0
        record.retrieved_contexts = retrieval["contexts"]
        record.retrieved_scores = retrieval["scores"]

        if not retrieval["contexts"]:
            record.injection_strategy = "skipped (no contexts above score_threshold)"
            return record, None

        return record, retrieval

    def _run_injection(
        self, record: RequestLogRecord, request: InjectionRequest, strategy_label: str,
        prompt_text: str, context_text: str,
    ) -> None:
        """Shared injection-and-measurement step: pushes `request` through the live model via one
        blocking `TokenInjector` burst and fills in `record`'s injection-phase fields in place."""
        t1 = time.monotonic()
        stats = self.injector.run_to_completion(request)
        injection_latency_s = time.monotonic() - t1

        record.injection_strategy = strategy_label
        record.prompt_length_chars = len(prompt_text)
        record.context_length_chars = len(context_text)
        record.injected_token_count = stats.token_count
        record.injection_latency_s = injection_latency_s
        record.total_latency_s = (record.retrieval_latency_s or 0.0) + injection_latency_s
        record.kv_cache_status = inspect_kv_cache(self._lm_gen)

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

        Returns the record as a dict, **not yet written to the log**. The retrieval/injection
        phases are the only thing this method can time -- whatever happens next (the live
        server's open-ended duplex conversation, or `offline.py`'s bounded generation loop) is the
        caller's to measure. Call `finalize_and_log(record, ...)` exactly once, when the caller
        knows what (if anything) to add for the generation phase, to actually persist the row.
        Splitting it this way keeps one JSONL row per logical turn instead of two partial ones.
        """
        record, retrieval = self._retrieve_for_injection(query, InjectionMode.PERSONA_RAG.value)
        if retrieval is None:
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        request = InjectionRequest(
            text=knowledge_block, mode=InjectionMode.PERSONA_RAG.value, wrap_system_tags=True
        )
        self._run_injection(
            record, request,
            strategy_label="persona_rag (blocking burst, same <system> mechanism as persona prompt)",
            prompt_text=wrap_with_system_tags(knowledge_block),
            context_text=knowledge_block,
        )
        return record.to_dict()

    # ---- Mode B: connection-start injection, naive prompt template (negative control) ------
    def inject_standard_prompt_rag(self, query: str) -> dict:
        """Mode B -- the deliberate negative-control baseline (see
        docs/ARCHITECTURE_REPORT.md Section 6). Retrieves context identically to Mode C (same
        `_retrieve_for_injection` call, same top_k/score_threshold), but formats it as a generic
        chatbot-style instruction block instead of PersonaPlex's own `<system>...<system>`
        convention, and does NOT wrap it in `<system>` tags:

            Relevant Knowledge:
            <retrieved facts>

            User Question:
            <query>

            Use the knowledge above when answering.

        This is intentionally the "obvious" thing someone might try if they treated PersonaPlex
        like an ordinary text-prompted chat LLM, without accounting for the fact that it has no
        prompt string and was never trained on this template's phrasing. Expected (and the point
        of running this experiment) to retrieve the same facts as Mode C but ground the response
        less reliably. Same connection-start-only call-site constraint as Mode C applies (no ASR
        to retrieve against mid-call -- see the Phase 2 implementation report).

        Returns the record as a dict, not yet logged -- call `finalize_and_log(...)`, same as
        Mode C.
        """
        record, retrieval = self._retrieve_for_injection(query, InjectionMode.PROMPT_RAG.value)
        if retrieval is None:
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        naive_prompt = (
            f"Relevant Knowledge:\n{knowledge_block}\n\n"
            f"User Question:\n{query}\n\n"
            "Use the knowledge above when answering."
        )
        request = InjectionRequest(
            text=naive_prompt, mode=InjectionMode.PROMPT_RAG.value, wrap_system_tags=False
        )
        self._run_injection(
            record, request,
            strategy_label="prompt_rag (naive 'Relevant Knowledge' block, no <system> wrapping -- negative control)",
            prompt_text=naive_prompt,
            context_text=knowledge_block,
        )
        return record.to_dict()

    # ---- Mode D: turn-boundary-triggered burst injection -----------------------------------
    #
    # IMPORTANT (see docs/MODE_D_REDESIGN.md for the full investigation): an earlier version of
    # this mode queued the prepared knowledge for *incremental*, one-forced-step-per-real-tick
    # injection via queue_injection()/consume_one_tick(). A real run showed this corrupts both the
    # visible transcript and the spoken audio -- forcing `text_token=X` always means "the model's
    # output at this position IS X right now," not "X is new context the model may react to
    # later," and the audio depformer conditions on whichever text token is active each frame.
    # Interleaving forced steps with a real generation loop that is actively decoding/forwarding
    # output therefore leaks the raw injected text (verbatim, `<system>` tags included) into both.
    # The fix is to always inject as one self-contained burst -- identical in kind to Mode C's
    # connection-start burst, just triggered later, by a detected pause instead of by the
    # connection starting.
    def prepare_turn_injection_knowledge(self, query: str) -> dict:
        """Mode D setup. Retrieves context for `query` **once**, using the deliberately small
        `config.turn_injection_top_k` (not `config.top_k` -- see `RAGConfig.turn_injection_top_k`'s
        docstring on why per-turn re-injection must stay short), and holds the resulting
        `<system>...<system>`-wrapped knowledge block ready to be re-fired as a burst on every
        detected turn boundary (see `fire_turn_injection_burst`/`_async`).

        This method itself does **not** push anything through the model -- no tokens are forced
        here, so `injected_token_count`/`injection_latency_s` stay at their defaults in the
        returned record. Intended call site: same as Mode C/B (right after
        `step_system_prompts_async` completes, still inside the connection's lock, before
        opus_loop starts). Call `finalize_and_log(record)` on the result the same way as the
        other modes.
        """
        record = RequestLogRecord(mode=InjectionMode.TURN_INJECTION.value, user_query=query)

        if not self.config.enable_rag or self.retriever is None:
            record.injection_strategy = "skipped (RAG disabled or no index loaded)"
            return record.to_dict()

        # Deliberately uses turn_injection_top_k, NOT config.top_k -- this is the one retrieval
        # call for Mode D, sized for repeated mid-conversation re-injection (see
        # RAGConfig.turn_injection_top_k's docstring), unlike Mode B/C's single larger retrieval.
        t0 = time.monotonic()
        retrieval = self.retriever.retrieve_context(
            query, top_k=self.config.turn_injection_top_k, score_threshold=self.config.score_threshold
        )
        record.retrieval_latency_s = time.monotonic() - t0
        record.retrieved_contexts = retrieval["contexts"]
        record.retrieved_scores = retrieval["scores"]
        if not retrieval["contexts"]:
            record.injection_strategy = "skipped (no contexts above score_threshold at turn_injection_top_k)"
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        self._turn_injection_request = InjectionRequest(
            text=knowledge_block, mode=InjectionMode.TURN_INJECTION.value, wrap_system_tags=True
        )
        self._turn_injection_query = query
        record.injection_strategy = (
            "turn_injection (prepared; fired as a burst on each detected turn boundary)"
        )
        record.context_length_chars = len(knowledge_block)
        record.prompt_length_chars = len(wrap_with_system_tags(knowledge_block))
        return record.to_dict()

    def observe_user_frame(self, pcm_frame) -> bool:
        """Feed one frame of raw user-audio PCM to the turn-boundary detector. No-op (returns
        False) unless Mode D is active with VAD enabled and `prepare_turn_injection_knowledge()`
        has already armed a knowledge block -- safe to call unconditionally every frame in any
        mode.

        Returns True iff a turn boundary was just detected. This method only *detects* -- it does
        not inject anything itself. The caller must then fire the burst (`fire_turn_injection_burst()`
        synchronously, or `await fire_turn_injection_burst_async()`) **before** processing the
        current real audio frame any further, so the burst always completes as a self-contained
        unit rather than being interleaved with real generation (see the module-level note above).

        Must be called from the same coroutine that owns `lm_gen.step()` for this connection (the
        same constraint as everything else in this class) -- in the reference server, that's
        `opus_loop`, right where it already slices each raw PCM frame off the incoming buffer; in
        `offline.py`, the equivalent point in its single encode/step loop.
        """
        if self.turn_detector is None or self._turn_injection_request is None:
            return False
        return self.turn_detector.push_frame(pcm_frame)

    # ---- Shared burst-fire-and-log step for any mid-conversation mode (D, E, ...) -----------
    def _fire_prepared_burst(self, request: InjectionRequest, query: Optional[str], strategy_label: str) -> dict:
        """Forces `request`'s tokens through the live model as ONE synchronous blocking burst
        and logs a single completed record immediately. No `finalize_and_log` step needed here
        (unlike Mode B/C's connection-start injection) -- there is no separate bounded "generation
        phase" to wait for; the burst itself is the entire unit of work for this call. For
        `moshi.offline`, which has no event loop to share; use `_fire_prepared_burst_async` for
        the live server."""
        record = RequestLogRecord(mode=request.mode, user_query=query)
        t0 = time.monotonic()
        stats = self.injector.run_to_completion(request)
        injection_latency_s = time.monotonic() - t0

        record.injection_strategy = strategy_label
        record.injected_token_count = stats.token_count
        record.injection_latency_s = injection_latency_s
        record.total_latency_s = injection_latency_s
        record.kv_cache_status = inspect_kv_cache(self._lm_gen)

        self.logger.log(record)
        return record.to_dict()

    async def _fire_prepared_burst_async(self, request: InjectionRequest, query: Optional[str], strategy_label: str) -> dict:
        """Async-checkpointed equivalent of `_fire_prepared_burst`, for `moshi.server`'s
        `opus_loop` -- see `TokenInjector.run_to_completion_async`. Identical fields/logging."""
        record = RequestLogRecord(mode=request.mode, user_query=query)
        t0 = time.monotonic()
        stats = await self.injector.run_to_completion_async(request)
        injection_latency_s = time.monotonic() - t0

        record.injection_strategy = strategy_label
        record.injected_token_count = stats.token_count
        record.injection_latency_s = injection_latency_s
        record.total_latency_s = injection_latency_s
        record.kv_cache_status = inspect_kv_cache(self._lm_gen)

        self.logger.log(record)
        return record.to_dict()

    def fire_turn_injection_burst(self) -> dict:
        """Mode D: fires the prepared turn-injection knowledge as ONE synchronous blocking burst
        (same underlying mechanism as Mode C's `inject_persona_compatible_knowledge`), triggered
        by a detected turn boundary instead of connection start. For `moshi.offline`; use
        `fire_turn_injection_burst_async()` for the live server."""
        return self._fire_prepared_burst(
            self._turn_injection_request, self._turn_injection_query,
            "turn_injection (burst, fired on detected turn boundary)",
        )

    async def fire_turn_injection_burst_async(self) -> dict:
        """Async-checkpointed equivalent of `fire_turn_injection_burst`, for `moshi.server`'s
        `opus_loop`."""
        return await self._fire_prepared_burst_async(
            self._turn_injection_request, self._turn_injection_query,
            "turn_injection (burst, fired on detected turn boundary)",
        )

    # ---- Mode E: fixed-interval (time-based) burst injection, independent of turn boundaries -
    #
    # Per the Mode D finding (docs/MODE_C_IMPLEMENTATION_REPORT.md Section 8, real run on the RTX
    # 5090 pod): wrapping a mid-call burst in `<system>...<system>` tags -- the same format used
    # exactly once, at connection start, by `step_system_prompts` -- appears to read to the model
    # as "a call is starting," causing it to abandon its sentence and re-greet instead of
    # grounding in the injected facts. Mode E deliberately tests the same proven-safe burst
    # mechanism WITHOUT that wrapping: a plain knowledge block, no `<system>` tags, no Mode-B-style
    # "Relevant Knowledge:/User Question:" framing either (there is no specific question to frame
    # against a periodic, conversation-state-independent re-fire). This isolates whether the
    # `<system>` tag itself was the cause of Mode D's derailment.
    def prepare_dynamic_injection_knowledge(self, query: str) -> dict:
        """Mode E setup. Retrieves context for `query` once, using `config.dynamic_injection_top_k`
        (same "keep it small" reasoning as Mode D's `turn_injection_top_k` -- repeated
        mid-conversation injection must stay cheap; ~25ms/injected token measured for Mode B/C),
        and holds a PLAIN (not `<system>`-wrapped) knowledge block ready to be re-fired as a burst
        every `config.dynamic_injection_interval_s` seconds via `tick_dynamic_injection()` /
        `fire_dynamic_injection_burst()`/`_async()`.

        Starts the fixed-interval clock now -- the first re-fire happens
        `dynamic_injection_interval_s` seconds after this call, not after the first tick. Does
        not push anything through the model itself; call `finalize_and_log(record)` on the result
        the same way as Mode C/B/D's setup methods.
        """
        record = RequestLogRecord(mode=InjectionMode.DYNAMIC_RUNTIME.value, user_query=query)

        if not self.config.enable_rag or self.retriever is None:
            record.injection_strategy = "skipped (RAG disabled or no index loaded)"
            return record.to_dict()

        t0 = time.monotonic()
        retrieval = self.retriever.retrieve_context(
            query, top_k=self.config.dynamic_injection_top_k, score_threshold=self.config.score_threshold
        )
        record.retrieval_latency_s = time.monotonic() - t0
        record.retrieved_contexts = retrieval["contexts"]
        record.retrieved_scores = retrieval["scores"]
        if not retrieval["contexts"]:
            record.injection_strategy = "skipped (no contexts above score_threshold at dynamic_injection_top_k)"
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        self._dynamic_injection_request = InjectionRequest(
            text=knowledge_block, mode=InjectionMode.DYNAMIC_RUNTIME.value, wrap_system_tags=False
        )
        self._dynamic_injection_query = query
        self._dynamic_injection_last_fire_time = time.monotonic()
        record.injection_strategy = (
            "dynamic_runtime (prepared, no <system> wrapping; re-fired as a burst every "
            f"{self.config.dynamic_injection_interval_s}s)"
        )
        record.context_length_chars = len(knowledge_block)
        record.prompt_length_chars = len(knowledge_block)  # no wrapping added for this mode
        return record.to_dict()

    def tick_dynamic_injection(self) -> bool:
        """Call once per real audio frame -- no-op (returns False) unless Mode E is active and
        `prepare_dynamic_injection_knowledge()` has already armed a knowledge block, safe to call
        unconditionally in any mode (same contract as `observe_user_frame()`).

        Returns True iff `config.dynamic_injection_interval_s` has elapsed since the last fire (or
        since `prepare_dynamic_injection_knowledge`, for the first one), and resets the internal
        clock. Like `observe_user_frame()`, this only *detects* -- the caller must fire the burst
        (`fire_dynamic_injection_burst()`/`_async()`) before processing the current frame any
        further, so it always completes as a self-contained unit (the same interleaving hazard
        documented for Mode D applies to any forced-token mechanism, not just D).
        """
        if self._dynamic_injection_request is None or self._dynamic_injection_last_fire_time is None:
            return False
        if time.monotonic() - self._dynamic_injection_last_fire_time < self.config.dynamic_injection_interval_s:
            return False
        self._dynamic_injection_last_fire_time = time.monotonic()
        return True

    def fire_dynamic_injection_burst(self) -> dict:
        """Mode E: fires the prepared dynamic-injection knowledge as ONE synchronous blocking
        burst (same mechanism as Mode D's `fire_turn_injection_burst`), triggered by a fixed
        wall-clock interval instead of a detected pause. For `moshi.offline`; use
        `fire_dynamic_injection_burst_async()` for the live server."""
        return self._fire_prepared_burst(
            self._dynamic_injection_request, self._dynamic_injection_query,
            f"dynamic_runtime (burst, fired on {self.config.dynamic_injection_interval_s}s interval, no <system> wrapping)",
        )

    async def fire_dynamic_injection_burst_async(self) -> dict:
        """Async-checkpointed equivalent of `fire_dynamic_injection_burst`, for `moshi.server`'s
        `opus_loop`."""
        return await self._fire_prepared_burst_async(
            self._dynamic_injection_request, self._dynamic_injection_query,
            f"dynamic_runtime (burst, fired on {self.config.dynamic_injection_interval_s}s interval, no <system> wrapping)",
        )

    # ---- Mode F: cache-aware burst vs. a naive reset_and_replay baseline -- a benchmark, not a --
    # new injection strategy. Modes C/D/E already established the only mechanism this project
    # found that works for connection-start grounding (the <system>-wrapped burst); Mode F exists
    # to quantify *why preserving the live RingKVCache matters at all* by measuring what it would
    # cost an implementation that did NOT have this project's live-injection capability and had to
    # fall back to the obvious alternative: reset_streaming() (wiping the RingKVCache) and replay
    # the entire persona/voice prompt setup phase from scratch before it could inject anything.
    def fire_cache_aware_burst(self, query: str) -> dict:
        """Mode F, arm 1 (cache-aware). Retrieves context for `query` and injects it via the same
        connection-preserving burst mechanism as Mode C -- `reset_streaming()` is never called,
        so whatever is already in the live RingKVCache (persona + voice prompt, and in a real
        live call, anything said so far) stays intact. This is the arm representing what this
        project's mechanism makes possible; `benchmark_reset_and_replay_baseline`/`_async` is the
        naive alternative it's being measured against.

        Returns the record as a dict, not yet logged -- call `finalize_and_log(...)`, same as
        Mode C/B's connection-start methods.
        """
        record, retrieval = self._retrieve_for_injection(query, InjectionMode.CACHE_AWARE.value)
        if retrieval is None:
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        request = InjectionRequest(
            text=knowledge_block, mode=InjectionMode.CACHE_AWARE.value, wrap_system_tags=True
        )
        self._run_injection(
            record, request,
            strategy_label="cache_aware (burst, no reset -- preserves the live RingKVCache)",
            prompt_text=wrap_with_system_tags(knowledge_block),
            context_text=knowledge_block,
        )
        return record.to_dict()

    def benchmark_reset_and_replay_baseline(self, query: str, replay_persona_and_voice_prompt_fn) -> dict:
        """Mode F, arm 2 (naive baseline). Retrieves the SAME knowledge as arm 1, but simulates an
        implementation that lacks this project's live-injection mechanism: it must call
        `reset_streaming()` (wiping the RingKVCache) and replay the entire persona/voice prompt
        setup phase from scratch before it can inject anything. `replay_persona_and_voice_prompt_fn`
        is a zero-argument callable supplied by the caller (`moshi.offline`/`moshi.server`) that
        performs that replay -- this class has no handle on the voice prompt path or persona text
        needed to call `LMGen.step_system_prompts` itself, by design (RAGSession otherwise never
        calls `reset_streaming()`, and this is the one mode where that's the deliberate point of
        the comparison, not an accident).

        Times the WHOLE reset + replay + re-injection sequence as one cost, because that total --
        not just the injection step -- is the number that answers "what would this cost without
        this project's mechanism." For `moshi.offline`; use `_async` for the live server.
        """
        record, retrieval = self._retrieve_for_injection(query, InjectionMode.CACHE_AWARE.value)
        if retrieval is None:
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        request = InjectionRequest(
            text=knowledge_block, mode=InjectionMode.CACHE_AWARE.value, wrap_system_tags=True
        )

        t0 = time.monotonic()
        replay_persona_and_voice_prompt_fn()
        stats = self.injector.run_to_completion(request)
        replay_and_injection_latency_s = time.monotonic() - t0

        record.injection_strategy = (
            "cache_aware (naive reset_and_replay baseline -- reset_streaming() + full "
            "persona/voice prompt replay + reinjection)"
        )
        record.injected_token_count = stats.token_count
        record.injection_latency_s = replay_and_injection_latency_s
        record.total_latency_s = (record.retrieval_latency_s or 0.0) + replay_and_injection_latency_s
        record.kv_cache_status = inspect_kv_cache(self._lm_gen)
        self.logger.log(record)
        return record.to_dict()

    async def benchmark_reset_and_replay_baseline_async(self, query: str, replay_persona_and_voice_prompt_fn) -> dict:
        """Async-checkpointed equivalent of `benchmark_reset_and_replay_baseline`, for
        `moshi.server`'s connection-start call site. `replay_persona_and_voice_prompt_fn` must be
        an async callable (it will be awaited) -- the live server replays via
        `LMGen.step_system_prompts_async`, not the sync `step_system_prompts`."""
        record, retrieval = self._retrieve_for_injection(query, InjectionMode.CACHE_AWARE.value)
        if retrieval is None:
            return record.to_dict()

        knowledge_block = "\n".join(retrieval["contexts"])
        request = InjectionRequest(
            text=knowledge_block, mode=InjectionMode.CACHE_AWARE.value, wrap_system_tags=True
        )

        t0 = time.monotonic()
        await replay_persona_and_voice_prompt_fn()
        stats = await self.injector.run_to_completion_async(request)
        replay_and_injection_latency_s = time.monotonic() - t0

        record.injection_strategy = (
            "cache_aware (naive reset_and_replay baseline -- reset_streaming() + full "
            "persona/voice prompt replay + reinjection)"
        )
        record.injected_token_count = stats.token_count
        record.injection_latency_s = replay_and_injection_latency_s
        record.total_latency_s = (record.retrieval_latency_s or 0.0) + replay_and_injection_latency_s
        record.kv_cache_status = inspect_kv_cache(self._lm_gen)
        self.logger.log(record)
        return record.to_dict()

    def finalize_and_log(
        self,
        record: dict,
        generation_latency_s: Optional[float] = None,
        final_answer: Optional[str] = None,
    ) -> dict:
        """Completes a record returned by `inject_persona_compatible_knowledge` with whatever the
        caller learned afterward, recomputes `total_latency_s` to include the generation phase,
        and writes the one complete row to the JSONL log.

        `offline.py` calls this with both `generation_latency_s` (timed around its bounded
        encode/step/decode loop) and `final_answer` (the joined transcript). `server.py`'s
        connection-start call site has no equivalent bounded "generation phase" -- the live duplex
        conversation just keeps going -- so it calls this immediately with neither argument,
        which leaves those two fields `None` in the log, correctly reflecting that they don't
        apply there.

        Call this exactly once per record: the log is append-only, so a second call for the same
        logical turn produces a second, separate row rather than amending the first.
        """
        if generation_latency_s is not None:
            record["generation_latency_s"] = generation_latency_s
        if final_answer is not None:
            record["final_answer"] = final_answer
        record["total_latency_s"] = (
            (record.get("retrieval_latency_s") or 0.0)
            + (record.get("injection_latency_s") or 0.0)
            + (record.get("generation_latency_s") or 0.0)
        )
        self.logger.log(RequestLogRecord(**record))
        return record

    # ---- Generic incremental primitives -- NOT currently used by any mode ------------------
    #
    # CAUTION: do not wire these into a loop that also decodes/forwards real generation output
    # without first suppressing that output for the duration of the drain. Mode D originally used
    # exactly this pattern and it corrupted both the transcript and the spoken audio -- see the
    # warning at the top of the "Mode D" section above and docs/MODE_D_REDESIGN.md. Kept as a
    # tested, lower-level primitive (e.g. for a future mode that has a legitimate reason to spread
    # a burst across ticks *and* correctly suppresses output during the drain), not as the
    # recommended way to do mid-conversation injection.
    def queue_injection(self, request: InjectionRequest) -> None:
        """Queue an `InjectionRequest` for incremental consumption via `consume_one_tick()`. See
        the CAUTION above before reaching for this over a burst
        (`TokenInjector.run_to_completion`/`run_to_completion_async`)."""
        self.pending_job = self.injector.start(request)

    def consume_one_tick(self) -> bool:
        """Executes at most one forced step from a job queued via `queue_injection()`. See the
        CAUTION above before reaching for this over a burst.

        Returns True if a step was executed. Safe to call even when there is no pending job
        (no-op, returns False).
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

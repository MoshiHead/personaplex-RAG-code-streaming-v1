"""
Generic, mode-agnostic token-injection primitive for PersonaPlex's continuous streaming decoder.

Read docs/STREAMING_AND_INJECTION_DESIGN.md (Sections 3-4) for the full reasoning. Summary of the
constraints this module is built around:

  - PersonaPlex has no "prompt string". All conditioning -- the persona/system prompt, the voice
    prompt, and (with this module) RAG context -- enters the live model through
    `LMGen.step(...)`, one ~80ms frame at a time. There is no batched-prefill code path.
  - The real attention KV-cache (`RingKVCache`, in moshi/moshi/modules/transformer.py) is an
    append-only ring buffer, shared across an entire connection, and mutated by exactly one
    coroutine in the reference server: the one running `opus_loop`. It has NO internal locking.
  - Therefore: this module's stepping methods must only ever be invoked from that same single
    coroutine/thread for a given LMGen instance. Calling it concurrently from a second task WILL
    corrupt the shared streaming state. This module does not add a lock itself, because a lock
    would (incorrectly) imply concurrent use is safe if you just wait your turn -- it is not; the
    fix is "only ever call this from the opus_loop-equivalent coroutine," which is a call-site
    discipline, not something a lock here can enforce.
  - Injecting must NEVER call `reset_streaming()` -- that wipes the entire live conversation's
    RingKVCache (persona prompt, voice prompt, and all prior turns), not just "the prompt".

This module intentionally has zero dependency on `moshi`, `torch`, or any vector-store/embedding
library, so it is fully unit-testable with plain Python stand-ins (see
rag/tests/test_injection_manager.py) without a GPU or the real model loaded. It is the one
primitive that Modes C, D, E and F (Phase 1 architecture report, Section 6) all reuse; the
decision of *when* to call it and *what text* to inject is the caller's policy, not this module's
concern.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import time


@runtime_checkable
class StepCapable(Protocol):
    """Minimal protocol satisfied by `moshi.models.lm.LMGen` (and by test stand-ins)."""

    def step(
        self,
        input_tokens: Any = None,
        moshi_tokens: Any = None,
        text_token: Any = None,
        return_embeddings: bool = False,
    ) -> Any: ...


@runtime_checkable
class TextTokenizerLike(Protocol):
    """Minimal protocol satisfied by `sentencepiece.SentencePieceProcessor` (and test stand-ins)."""

    def encode(self, text: str) -> list: ...


def wrap_with_system_tags(text: str) -> str:
    """Wraps text in `<system> ... <system>` tags, exactly matching
    `moshi.server.wrap_with_system_tags` / `moshi.offline.wrap_with_system_tags`.

    Duplicated here on purpose (3 lines) rather than imported, so this module never requires
    `moshi` (and therefore `aiohttp`, `huggingface_hub`, `torch`, ...) to be importable just to
    build or unit-test an injection request. If PersonaPlex's own helper ever changes, this copy
    should be updated to match -- Mode C's entire premise is "use the exact same wrapping the
    persona prompt uses," so the two must stay byte-for-byte identical.
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class InjectionRequest:
    """Describes a block of text to force through the live model as text tokens.

    `mode` is a free-form label (e.g. "persona_rag", "prompt_rag") used only for logging and
    benchmarking (Phase 8/9) -- it has no effect on how the tokens are pushed; every mode shares
    the same stepping mechanics in `TokenInjector`.
    """

    text: str
    mode: str = "unspecified"
    wrap_system_tags: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class InjectionStats:
    """Running/finished statistics for one `InjectionJob`, suitable for the Phase 8 benchmark
    suite and the Phase 9 per-request log to consume directly (e.g. via `dataclasses.asdict`)."""

    mode: str
    token_count: int = 0
    steps_executed: int = 0
    wall_time_s: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    finished: bool = False

    @property
    def tokens_per_second(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.steps_executed / self.wall_time_s


class TokenInjector:
    """Forces a sequence of text tokens through a live `LMGen` stream using the same per-frame
    mechanism PersonaPlex uses for its own persona/system prompt
    (`moshi.models.lm.LMGen._step_text_prompt_core`), without ever calling `reset_streaming()`.

    Two usage patterns (see docs/STREAMING_AND_INJECTION_DESIGN.md Section 3.3 for when to pick
    which):

    Blocking burst -- mirrors how the original system prompt is loaded; simplest, but the full
    cost is paid in one go::

        injector = TokenInjector(lm_gen, text_tokenizer, make_zero_audio_frame, make_silence_audio_frame)
        stats = injector.run_to_completion(InjectionRequest(text="...", mode="persona_rag"))

    Incremental -- spreads the same forced steps across many `opus_loop` ticks so no single tick
    is blocked for the full duration::

        job = injector.start(InjectionRequest(text="...", mode="dynamic_runtime"))
        # from inside opus_loop's existing while-loop, once per tick, before draining real audio:
        if not job.done:
            job.step_once()
    """

    def __init__(
        self,
        lm_gen: StepCapable,
        text_tokenizer: TextTokenizerLike,
        make_zero_audio_frame,
        make_silence_audio_frame,
        zero_text_code: int = 3,
    ):
        """
        Parameters
        ----------
        lm_gen : StepCapable
            The live `LMGen` instance for this connection (or a test stand-in).
        text_tokenizer : TextTokenizerLike
            Tokenizer used to turn injected text into token ids (the same SentencePiece tokenizer
            the server already uses for the persona prompt).
        make_zero_audio_frame : Callable[[], Any]
            Returns the "silence" agent-audio-codebook tensor to force during injection steps.
            In the reference server this is `LMGen._encode_zero_frame`.
        make_silence_audio_frame : Callable[[], Any]
            Returns the "sine" input-audio tensor to force during injection steps. In the
            reference server this is `LMGen._encode_sine_frame`. Using the sine frame (rather than
            real buffered user audio) exactly matches how the persona prompt is loaded; callers
            that want to avoid discarding live user audio during injection should instead feed
            real encoded user-audio frames as `input_tokens` via a custom call to `_force_one_token`
            -- left as a documented extension point, not implemented here, since Mode C's stated
            goal is byte-for-byte parity with the existing persona-prompt mechanism.
        zero_text_code : int
            Token id meaning "no text" on the *other* streams during this step (matches
            `LMGen.zero_text_code`, which is always `3` for the released PersonaPlex checkpoint).
        """
        self._lm_gen = lm_gen
        self._tokenizer = text_tokenizer
        self._make_zero_audio_frame = make_zero_audio_frame
        self._make_silence_audio_frame = make_silence_audio_frame
        self._zero_text_code = zero_text_code

    def encode(self, request: InjectionRequest) -> list:
        """Tokenize `request.text`, applying the `<system>` wrapper iff requested. Pure function,
        safe to call from any thread/coroutine (does not touch `lm_gen`)."""
        text = wrap_with_system_tags(request.text) if request.wrap_system_tags else request.text
        return list(self._tokenizer.encode(text))

    def start(self, request: InjectionRequest) -> "InjectionJob":
        """Tokenize `request` and return a fresh, not-yet-started `InjectionJob`. Must be driven
        (via `step_once()`) from the same coroutine/thread that owns `lm_gen.step()`."""
        token_ids = self.encode(request)
        return InjectionJob(self, request, token_ids)

    def run_to_completion(self, request: InjectionRequest) -> InjectionStats:
        """Convenience wrapper: push every token through in one blocking call. See the class
        docstring for when to prefer the incremental `start()`/`step_once()` pattern instead."""
        job = self.start(request)
        while not job.done:
            job.step_once()
        return job.stats

    def _force_one_token(self, token_id) -> None:
        """The actual unit of work: one forced frame, identical in shape to
        `LMGen._step_text_prompt_core`'s loop body."""
        self._lm_gen.step(
            moshi_tokens=self._make_zero_audio_frame(),
            text_token=token_id,
            input_tokens=self._make_silence_audio_frame(),
        )


class InjectionJob:
    """One in-progress (or completed) injection, driven one forced frame at a time.

    Constructed via `TokenInjector.start()`. Not meant to be instantiated directly.
    """

    def __init__(self, injector: TokenInjector, request: InjectionRequest, token_ids: list):
        self._injector = injector
        self.request = request
        self._token_ids = token_ids
        self._cursor = 0
        self.stats = InjectionStats(mode=request.mode, token_count=len(token_ids))

    @property
    def done(self) -> bool:
        return self._cursor >= len(self._token_ids)

    def step_once(self) -> bool:
        """Force exactly one queued token through the live model.

        Returns True if a step was executed, False if the job was already complete (calling
        `step_once()` on a finished job is a safe no-op, so callers don't need to re-check `done`
        between checking it and calling this).
        """
        if self.done:
            self.stats.finished = True
            return False

        token_id = self._token_ids[self._cursor]
        self._injector._force_one_token(token_id)
        self._cursor += 1

        self.stats.steps_executed += 1
        self.stats.wall_time_s = time.monotonic() - self.stats.started_at
        if self.done:
            self.stats.finished = True
        return True

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
  - CRITICAL (found via a real Mode D run -- see docs/MODE_D_REDESIGN.md): forcing
    `text_token=X` via `step()` does not mean "show the model X as context it can choose to react
    to later." It means "the model's output at this position IS X, right now" -- the audio
    depformer also conditions on whichever text token is active each frame, forced or not. This
    is harmless ONLY when nothing reads/forwards `step()`'s resolved output while forcing happens
    (true for the persona prompt and for `run_to_completion`/`run_to_completion_async` when called
    *before* a real generation loop starts watching output, as Modes B/C do). It actively corrupts
    both the visible transcript and the spoken audio if forced steps are interleaved with a real
    generation loop that IS reading output at the same time -- do not build an "incremental,
    one-forced-step-per-real-tick" injection mode; always inject as one self-contained burst
    (`run_to_completion`/`run_to_completion_async`), even if triggered mid-conversation (Mode D).

This module intentionally has zero dependency on `moshi`, `torch`, or any vector-store/embedding
library, so it is fully unit-testable with plain Python stand-ins (see
rag/tests/test_injection_manager.py) without a GPU or the real model loaded. It is the one
primitive that Modes C, D, E and F (Phase 1 architecture report, Section 6) all reuse; the
decision of *when* to call it and *what text* to inject is the caller's policy, not this module's
concern.
"""

import asyncio
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

    Always inject as a self-contained burst, never interleaved with a real generation loop that
    is concurrently reading/forwarding `step()`'s output (see the CRITICAL note above and
    docs/MODE_D_REDESIGN.md) -- two burst variants, depending on whether the caller has an
    asyncio event loop to share:

    Synchronous burst (e.g. `moshi.offline`, which has no event loop)::

        injector = TokenInjector(lm_gen, text_tokenizer, make_zero_audio_frame, make_silence_audio_frame)
        stats = injector.run_to_completion(InjectionRequest(text="...", mode="persona_rag"))

    Async-checkpointed burst (e.g. `moshi.server`'s `opus_loop`) -- functionally identical, but
    yields control after each forced step so other coroutines (`recv_loop`/`send_loop`) aren't
    starved for the whole burst, mirroring `LMGen._step_text_prompt_async`'s pattern::

        stats = await injector.run_to_completion_async(InjectionRequest(text="...", mode="turn_injection"))

    `start()`/`InjectionJob.step_once()` remain available as lower-level primitives (e.g. for a
    caller that wants fine-grained control over exactly when each step fires), but do NOT use them
    to spread an injection's steps across ticks of a loop that is also decoding/forwarding real
    generation output -- that is exactly the interleaving pattern that caused the corruption
    described above.
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
        """Synchronous burst: push every token through in one blocking call, with no opportunity
        for anything else to run in between. Use this when the caller has no event loop to share
        (e.g. `moshi.offline`) or when blocking briefly is genuinely fine (e.g. Mode C/B's
        connection-start injection, before any real generation loop is watching output)."""
        job = self.start(request)
        while not job.done:
            job.step_once()
        return job.stats

    async def run_to_completion_async(self, request: InjectionRequest) -> InjectionStats:
        """Async-checkpointed burst: identical effect to `run_to_completion` (every token is
        still forced through sequentially, in one self-contained burst, with no real generation
        interleaved in between -- see the class docstring's CRITICAL note on why that matters),
        but yields control via `await asyncio.sleep(0)` after each forced step so other
        coroutines on the same event loop (e.g. `recv_loop`/`send_loop` in `moshi.server`) get a
        chance to run during the burst, mirroring `LMGen._step_text_prompt_async`'s pattern.

        This does NOT make the burst safe to run concurrently with another coroutine that also
        calls `lm_gen.step()` -- the concurrency contract (Section 3.1 of
        docs/STREAMING_AND_INJECTION_DESIGN.md) is unchanged: only one coroutine may ever drive
        this `LMGen` instance. It only prevents that single coroutine's own burst from starving
        *other* coroutines on the same event loop for the burst's full duration.
        """
        job = self.start(request)
        while not job.done:
            job.step_once()
            await asyncio.sleep(0)
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

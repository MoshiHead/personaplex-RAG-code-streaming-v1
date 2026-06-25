# PersonaPlex Streaming Architecture & Token-Injection Design

Companion to [`ARCHITECTURE_REPORT.md`](./ARCHITECTURE_REPORT.md) (read that first). This document goes one
level deeper on exactly three things, as requested before any RAG implementation begins:

1. The complete streaming/connection lifecycle (including a correction/refinement to Phase 1's report).
2. The full `RingKVCache` lifecycle.
3. The safest point to inject live tokens without resetting the connection, and the generic interface design
   built on top of it.

No production code (`moshi/moshi/**`) is modified by this document or by the scaffolding it describes.

---

## 1. Per-connection lifecycle (refines Phase 1, Section 4c)

Phase 1 stated `reset_streaming()` is "only called once, right when a new WebSocket connects." That is true,
but incomplete — there are actually **two different resets with two different scopes**, both inside
`server.py: handle_chat`, and getting this right matters for injection design:

```
ServerState.__init__()                         <- once per SERVER PROCESS (not per connection)
  mimi.streaming_forever(1)                     <- allocates RingKVCache buffers for mimi's transformer
  other_mimi.streaming_forever(1)               <- allocates RingKVCache buffers for other_mimi
  lm_gen.streaming_forever(1)                   <- allocates RingKVCache buffers for all 32+6 attention layers
                                                    (allocation happens ONCE; buffers are reused, never
                                                    reallocated, for the life of the process)

handle_chat(request)                            <- once per WEBSOCKET CONNECTION
  lm_gen.load_voice_prompt(...)                 <- sets which persona/voice this connection will use
  lm_gen.text_prompt_tokens = encode(...)
  async with self.lock:                         <- SINGLE shared lock: only one live connection's model
    │                                              calls are ever in flight at a time, server-wide
    mimi.reset_streaming()                      <- zeroes mimi's conv-overlap state (RESET 1 of 2)
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()                    <- zeroes ALL attention RingKVCache end_offset/offset
    │                                              counters to 0 -- the *only* full wipe of conversational
    │                                              memory for this connection, and it happens exactly once,
    │                                              right here, before anything else
    await lm_gen.step_system_prompts_async(...) <- voice prompt + silence + text/persona prompt + silence,
    │                                              each one a real step() that grows the now-empty
    │                                              RingKVCache by exactly one position
    mimi.reset_streaming()                      <- RESET 2 of 2: only mimi's *codec* conv-overlap state,
    │                                              not lm_gen's RingKVCache. This clears residual signal
    │                                              continuity from encoding the voice-prompt WAV so live
    │                                              mic audio starts clean. lm_gen's attention cache is
    │                                              deliberately left untouched here -- the system prompt's
    │                                              tokens must remain visible to attention for the rest of
    │                                              the call, or the persona/voice conditioning is lost.
    │
    ws.send_bytes(handshake)
    [recv_loop(), opus_loop(), send_loop()]      <- 3 asyncio tasks, but only opus_loop ever touches lm_gen
```

**Consequence**: a single server process handles **one live conversation at a time** (the `self.lock` is a
process-wide singleton, not per-connection). This is an existing limitation of the reference server, not
something we introduce — but it matters for benchmarking (Phase 8): concurrent-load tests are out of scope for
this server as written; benchmarks should be single-session, sequential.

**Consequence for RAG**: `lm_gen.reset_streaming()` must **never** be called again after this point for a given
connection. Any injection strategy that achieves "fresh context" by reconnecting / resetting is, by definition,
throwing away the entire `RingKVCache` (system prompt + voice prompt + all prior conversation) and paying the
full `step_system_prompts` cost again. This is the formal definition of the "naive baseline" that Mode F should
be benchmarked against (see Section 3.4).

---

## 2. `RingKVCache` complete lifecycle

| Stage | What happens | Where | Cost |
|---|---|---|---|
| **Allocate** | `torch.zeros((2, B, H, capacity, D))` + `end_offset = 0`. `capacity` = the layer's `context` (3000 for main transformer layers, `weights_per_step` count for Depformer layers). One `RingKVCache` per `StreamingMultiheadAttention` instance (32 main + 6 Depformer = 38 total). | `StreamingMultiheadAttention._init_streaming_state`, called transitively by `lm_gen.streaming_forever(1)` | One-time, at process start. Memory: `2 × B × H × capacity × D × dtype_size` per layer — for the main transformer (`H=32, D=128, capacity=3000, bf16`) that's ≈ 2×1×32×3000×128×2 bytes ≈ **49 MB per layer × 32 layers ≈ 1.6 GB** just for the main transformer's KV cache at batch size 1. |
| **Reset (zero)** | `end_offset.zero_()`. Buffer *contents* are untouched (stale data from a prior connection), but become unreachable because `complete()`'s validity check is `indexes < end_offset`, not buffer content. | `lm_gen.reset_streaming()` → propagates to every `StreamingModule` child, including each `RingKVCache` | Once per new connection (Section 1). Cheap (one scalar write per layer). |
| **Grow (append)** | `complete(k, v)`: writes the new frame's K/V at `index_copy_(dim=2, index=end_offset % capacity, ...)`, then `end_offset += T` (`T=1` always in this codebase — see Phase 1, Section 4a). Returns `positions` for the causal mask, computed from `end_offset` modulo `capacity`. | `RingKVCache.complete()`, called once per attention layer per `lm_gen.step()` call | One write of `(2, B, H, 1, D)` per layer per step — cheap relative to the attention matmul itself. This is the operation any "injection" must go through; **there is no cheaper path**. |
| **Evict** | Implicit: once `end_offset > capacity`, the next write's `index_copy_` target index (`end_offset % capacity`) overwrites a slot still holding a previous, now too-old frame's K/V. The causal mask's `delta < context` check also independently makes any position older than `capacity` frames mathematically unreachable, even on the rare frame where the literal bytes haven't been overwritten yet. | `RingKVCache.complete()` (same call as Grow — eviction isn't a separate operation, it's a side-effect of being a fixed-size ring buffer) | No extra cost, but a hard **3000-frame / 240-second sliding window** on the main transformer. Anything injected (persona prompt, RAG context, conversation history) older than this is gone, full stop. |
| **Persist / restore (exists, but unused at runtime)** | `StreamingModule.save_streaming_state(...)` / `load_streaming_state(...)` (in `modules/streaming.py`) can serialize *any* streaming state — including every `RingKVCache`'s raw buffer + `end_offset` — to a safetensors file + JSON metadata, and restore it byte-for-byte later. **This utility already exists in the repo and is not currently called anywhere in `server.py`/`offline.py`.** | `modules/streaming.py: save_streaming_state/load_streaming_state`, `StreamingModule.save_streaming_state/set_streaming_state_inplace` | Not used today. This is the *only* mechanism in the codebase that could make a cache "free" to restore (skip replaying the persona/voice prompt by loading a previously-saved post-system-prompt cache snapshot) — flagged here as a candidate optimization for Phase 8 benchmarking (e.g., snapshot the cache immediately after `step_system_prompts` once per persona, and restore it instead of replaying voice+text prompts for every new connection using that persona), but it is explicitly **out of scope for the RAG injection modes themselves** — it doesn't help inject *new, per-request* content, only avoids replaying *static, already-known* content. |
| **Destroy** | Implicit — buffers are just `torch.Tensor`s owned by Python objects; freed on process exit or object garbage-collection. No explicit teardown exists or is needed. | — | — |

---

## 3. Safest insertion point for live token injection

### 3.1 Why it must be inside `opus_loop`, and nowhere else

`handle_chat` spawns three concurrent `asyncio` tasks per connection:

| Task | Touches `lm_gen`? | Touches audio I/O? |
|---|---|---|
| `recv_loop` | No | Appends incoming WebSocket bytes to `opus_reader` |
| `opus_loop` | **Yes — the only task that calls `lm_gen.step()`** | Drains `opus_reader`, encodes via Mimi, calls `lm_gen.step()`, decodes via Mimi, writes to `opus_writer` |
| `send_loop` | No | Drains `opus_writer`, sends WebSocket bytes |

`_LMGenState` (the object every `step()` call mutates: `cache`, `provided`, `offset`, and transitively every
`RingKVCache.end_offset`) has **no internal locking**. The only thing currently guaranteeing exclusive access is
that `opus_loop` is the sole caller of `lm_gen.step()`. **If a RAG injection were implemented as a second
concurrent task/coroutine calling `lm_gen.step()`, it would race with `opus_loop` and silently corrupt
`state.offset`/the ring buffers** (two writers advancing the same monotonic counter and writing to the same
ring positions without coordination). This is not a hypothetical risk to design around abstractly — it is the
single most important constraint on the injection interface.

**Conclusion: injection must execute synchronously, inline, from within the same coroutine that already runs
`opus_loop`'s loop body.** No new lock is needed — we get exclusivity for free by construction, as long as we
never spawn a second task that also calls `step()`.

### 3.2 Exact insertion point within `opus_loop`

```python
async def opus_loop():
    all_pcm_data = None
    while True:
        if close:
            return
        await asyncio.sleep(0.001)

        pcm = opus_reader.read_pcm()
        ... # real-audio handling, with one addition (see Section 3.3): right where a raw PCM
            # frame is sliced off, check the turn-boundary detector; if it just fired, `await` the
            # prepared knowledge as one self-contained burst (run_to_completion_async) BEFORE
            # processing that frame's real lm_gen.step() call -- never interleaved with it. When
            # there's nothing to detect (ENABLE_RAG=False, or no mode is using the detector), this
            # is a single cheap check and behavior is byte-for-byte identical to the current
            # server.
```

Why this exact spot, and not elsewhere:
- It is **before** the real-audio drain, so a detected turn boundary gets handled before that tick's real audio
  is processed.
- It requires **zero changes to `recv_loop`/`send_loop`** and **zero changes to the WebSocket protocol** — the
  client keeps streaming audio in and out exactly as today; any incoming audio that arrives during a burst
  simply queues in the existing buffers and gets processed right after, at the cost of the burst's latency.
- It does not touch `mimi`/`other_mimi` at all for text-only injection — only
  `lm_gen.step(text_token=..., moshi_tokens=<silence>, input_tokens=<silence>)`, exactly mirroring
  `LMGen._step_text_prompt_core`.

### 3.3 Why a self-contained burst, not incremental per-tick interleaving

> **Superseded by a real finding — see [`docs/MODE_D_REDESIGN.md`](./MODE_D_REDESIGN.md) for the full
> investigation.** This section originally argued for spreading an injection's forced steps one-per-tick across
> many `opus_loop` iterations (`TokenInjector.start()` + `InjectionJob.step_once()`), reasoning only about
> *latency smoothness*. A real Mode D run on the RTX 5090 pod showed that reasoning was incomplete: forcing
> `text_token=X` doesn't mean "X is new context the model may react to" — it means "the model's output at this
> position IS X, right now," and the audio depformer also conditions on whichever text token is active each
> frame. Interleaving forced steps with a real generation loop that is actively decoding/forwarding output
> therefore leaks the **raw** injected text (literal `<system>` tags included) into both the transcript and the
> spoken audio — spreading it thinner across more ticks doesn't reduce that corruption, it just scatters it
> across a longer stretch of the response instead of containing it to one moment.
>
> **Corrected guidance**: always inject as one self-contained burst — `TokenInjector.run_to_completion`
> (synchronous, for callers with no event loop, e.g. `moshi.offline`) or
> `TokenInjector.run_to_completion_async` (async-checkpointed — yields control after each forced step via
> `await asyncio.sleep(0)`, mirroring `LMGen._step_text_prompt_async`'s pattern, so `recv_loop`/`send_loop`
> aren't starved for the burst's duration — for `moshi.server`'s `opus_loop`). This applies even to Modes
> triggered mid-conversation (D/E): the burst just fires *later* (on a detected pause, or on a timer) instead of
> at connection start, not differently in kind from Mode C's burst. `start()`/`InjectionJob.step_once()` remain
> available as a lower-level primitive, but must not be wired into a loop that also reads real generation output
> without first ensuring that loop doesn't decode/forward output for the duration of the drain.

### 3.4 What "Cache-Aware RAG" (Mode F) actually means here

Per Phase 1, Section 6: there is no way to grow the `RingKVCache` for new content without a real forward pass.
"Cache-aware" therefore means: **always inject via the mechanism above (live, in-place, no reset), and benchmark
it against the literal naive alternative a less careful implementation might reach for**:

- **Cache-aware (what we build)**: `TokenInjector` steps applied live, inside `opus_loop`, no
  `reset_streaming()` ever called mid-conversation.
- **Naive baseline (what we benchmark against)**: tear down the WebSocket / call `reset_streaming()` and replay
  `step_system_prompts()` (original persona + voice prompt) plus a freshly-built knowledge-augmented text prompt,
  losing all conversation history in the process.

The benchmark in Phase 8 should report: tokens injected, wall time, and (for the naive baseline) the additional
cost of re-running `step_system_prompts` and the qualitative loss of conversational continuity.

---

## 4. Generic token-injection interface (design → implemented as scaffolding)

Implemented in [`rag/injection_manager.py`](../rag/injection_manager.py). Design goals, in priority order:

1. **Zero changes to `moshi/moshi/**`.** The interface only calls already-public methods (`LMGen.step`,
   a tokenizer's `.encode`) — it duplicates the ~3-line `wrap_with_system_tags` helper rather than importing
   `moshi.server` (which would pull in `aiohttp`/`huggingface_hub`/etc. as hard dependencies of the RAG package).
2. **Same mechanism as the persona prompt**, mirroring `LMGen._step_text_prompt_core`'s structure (force
   `text_token`, silence the agent-audio channel, feed a sine "silence" frame on the input-audio channel) so
   that Mode C ("use the exact same mechanism as PersonaPlex persona/system prompts") is satisfied by
   construction, not by convention.
3. **One primitive, reusable by every mode.** `InjectionRequest` carries a free-form `mode` tag purely for
   logging/benchmarking; the actual stepping logic in `TokenInjector`/`InjectionJob` is mode-agnostic.
   Per-mode *policy* (when to build a request, what text to put in it, how often) lives one layer up, outside
   this module — this module only knows how to safely push tokens, not when or why.
4. **Supports both burst variants from the (corrected) Section 3.3** — `run_to_completion`
   (synchronous) and `run_to_completion_async` (async-checkpointed, for `opus_loop`) — so every
   mode injects as one self-contained burst regardless of whether it fires at connection start or
   mid-conversation.
5. **Concurrency contract is explicit and documented in the class docstring** (Section 3.1) rather than enforced
   by a lock — adding a lock here would suggest it's safe to call from a second task, which it is not; the right
   fix for that misuse is at the call site (always drive it from `opus_loop`), not inside this module.
6. **No hard dependency on `torch`, `moshi`, or any vector-store/embedding library.** The injector only needs an
   object satisfying a minimal `step(...)` protocol and an object satisfying a minimal `encode(...)` protocol —
   it is fully unit-testable with plain Python stand-ins (see `rag/tests/test_injection_manager.py`), without a
   GPU or the real 7B model loaded.

The companion lightweight turn-boundary detector for Modes D/E is implemented in
[`rag/turn_detector.py`](../rag/turn_detector.py) — a lightweight energy-threshold-plus-hangover heuristic,
deliberately **not** a learned VAD model (no new ML dependency), built behind a narrow interface
(`TurnBoundaryDetector.push_frame(pcm) -> bool`) so it can be swapped for `webrtcvad`/Silero VAD later without
changing any caller. It is **off by default** (`RAGConfig.vad_enabled = False`); when off, Mode D has no
boundary signal and Mode E should fall back to its fixed-interval policy — `RAGConfig.validate()` surfaces this
as an explicit warning rather than silently doing the wrong thing.

---

## 5. What is *not* yet implemented (by design, per the phased plan)

- No wiring into `moshi/moshi/server.py: opus_loop` yet — Section 3.2 describes exactly where it will go, but
  the actual server edit is deferred to the incremental implementation phase (next), one mode at a time, so each
  mode can be benchmarked in isolation before the next is added.
- No retrieval/vector-store/embeddings (`rag/retriever.py`, `rag/vector_store.py`, `rag/embeddings.py`) yet —
  `TokenInjector` accepts raw text; wiring it to retrieved knowledge is the next increment.
- No `rag/experiments.py`/`rag/benchmark.py` yet.

This keeps every artifact added so far (`rag/config.py`, `rag/turn_detector.py`, `rag/injection_manager.py`)
independently testable and inert when `ENABLE_RAG=False`, satisfying "should not affect baseline PersonaPlex."

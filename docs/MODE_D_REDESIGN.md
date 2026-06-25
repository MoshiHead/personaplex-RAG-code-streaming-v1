# Mode D Redesign: Why Incremental Per-Tick Injection Corrupts Output

This documents a real bug found by actually running Mode D against the live model on the RunPod
RTX 5090 pod, the root cause, and the fix. It supersedes the "incremental vs. blocking burst"
guidance in `docs/STREAMING_AND_INJECTION_DESIGN.md` Section 3.3, which turned out to be based on
an incomplete model of the cost of injection (latency only -- it missed a correctness issue that
matters more).

## What was observed

Mode D was originally implemented as: detect a turn boundary mid-conversation, then queue the
prepared knowledge for **incremental** consumption -- one forced text token per real audio frame,
via `consume_one_tick()`, interleaved with the live `opus_loop`/`offline.py` generation loop. A
real run produced this transcript (PAD/EPAD noise stripped):

> "...Oh, I'm sorry. We **`<system> Drones may not be flown in winds exceeding 20mph, in rain, or
> in temperatures below 0°C. ...two pickup locations: the Downtown depot...`**"

The agent's own sentence ("We...") is cut off mid-word, and the **raw** injected block -- literal
`<system>` tag, literal KB text, even a SentencePiece byte-fallback artifact (`<0x0A>` for the
newline joining the two retrieved documents) -- gets spoken verbatim. The benchmark log also
showed the injection never finished draining (`0 completed incremental injection(s)`) because the
remaining audio ran out before all the queued tokens could be spread across it.

## Root cause

`LMGen.step(text_token=X)` does not mean "show the model X as context it may choose to react to
later." It means **"the model's output at this position IS X, right now"** -- `X` directly
overrides whatever the model would have sampled at that position. The audio depformer also
conditions on whichever text token is active each frame (forced or sampled), via
`next_text_token` in `process_transformer_output` (`moshi/moshi/models/lm.py`), so forcing the
text channel also perturbs the synthesized audio at that exact frame, not just the transcript.

This is harmless for the persona prompt, and for Mode C/B's connection-start burst, for one
specific reason: **nothing reads or forwards `step()`'s resolved output while the forcing
happens.** `step_system_prompts` never decodes its output at all; Mode C/B's
`TokenInjector.run_to_completion` discards every `step()` return value (`_force_one_token` doesn't
even capture it), and this happens entirely *before* the real generation loop starts watching
output. By the time anything starts decoding/forwarding text again, the forced tokens are safely
in the past (folded into the live `RingKVCache` as attention context) rather than being the
*current* output anyone is reading.

(There is a small, mostly cosmetic version of this even in Mode C/B: the resolved output has a
`max_delay`-frame lag, so the very first real step after a burst reads back roughly the last
forced position. This is consistent with the stray `"> "` that prefixed Mode C's own transcript --
one token's worth of tail lag, not the dozens-of-tokens leak Mode D produced.)

Mode D broke this property on purpose: it deliberately interleaved forced steps with the *same*
loop that is actively decoding and forwarding `step()`'s output to the transcript/client. Every
forced token was therefore guaranteed to eventually be read back and treated as if the agent had
said it -- spreading the injection thinner across more ticks doesn't reduce this effect, it just
scatters the corruption across a longer stretch of the response instead of containing it to one
moment.

## The fix

Mode D must inject as **one self-contained burst** -- mechanically identical to Mode C's burst,
just *triggered later*, by a detected pause, instead of at connection start. A burst inherently
doesn't leak, for the same reason Mode C/B's bursts don't: nothing reads `step()`'s return value
during it.

The remaining design problem was purely about *not freezing the live duplex conversation* for the
burst's duration (a real synchronous loop would block the entire asyncio event loop, including
`recv_loop`/`send_loop`, for the ~3-9s a burst takes). The existing codebase already had the
answer for this: `LMGen.step_system_prompts_async` is an async generator that yields a checkpoint
after each forced step so other coroutines can run between steps, while the single coroutine that
owns `lm_gen.step()` still drives every step sequentially (the concurrency contract in
`docs/STREAMING_AND_INJECTION_DESIGN.md` Section 3.1 is unaffected -- this only prevents the host
event loop from starving, not concurrent model access). `TokenInjector.run_to_completion_async`
mirrors that exact pattern.

## What changed

- `rag/injection_manager.py`: added `TokenInjector.run_to_completion_async` (async-checkpointed
  burst); added a prominent warning to the module/class docstrings about forcing-as-dictation.
- `rag/server_integration.py`: `observe_user_frame()` now only *detects* a boundary (returns
  `bool`) -- it no longer auto-queues anything. New `fire_turn_injection_burst()` (sync, for
  `moshi.offline`) and `fire_turn_injection_burst_async()` (for `moshi.server`'s `opus_loop`) fire
  the prepared knowledge as one burst, on demand. `queue_injection`/`consume_one_tick` remain as
  lower-level, explicitly-flagged-risky primitives (not deleted -- still correct in isolation, just
  not safe to wire into a loop that's also reading real generation output without extra care).
- `moshi/moshi/offline.py` / `moshi/moshi/server.py`: call sites updated to fire the burst
  (sync / `await`-ed async, respectively) immediately upon a detected boundary, instead of queuing
  for per-tick draining.
- Tests rewritten: `rag/tests/test_server_integration.py`'s Mode D class now tests
  detect-then-burst instead of detect-then-queue-then-drain; new `TestRAGSessionModeDAsyncBurst`
  (using `unittest.IsolatedAsyncioTestCase`) proves the async burst forces identical tokens to the
  sync version and doesn't starve other coroutines on the event loop.

## Status

Not yet re-run against the real model. Next step: re-run the notebook's Run 4 (Mode D) cell on the
RunPod pod and confirm the transcript no longer contains raw `<system>` tags or verbatim KB text --
the agent should either naturally pause-then-continue around the burst, or (more likely, given
Mode C's behavior) pick up the topic the same way Mode C did, just triggered mid-call instead of
upfront.

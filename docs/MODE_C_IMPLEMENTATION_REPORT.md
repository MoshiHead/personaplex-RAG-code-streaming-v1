# Mode C Implementation Report (Phase 2 increment)

What was built this increment, what was actually validated and how, what's left for you to run on
the real RunPod RTX 5090 pod, and the new architectural finding (no ASR in this pipeline) that
shaped the validation design. Read this before deciding whether to proceed to Modes B/D/E/F.

## 1. What was built

| File | Purpose |
|---|---|
| `rag/embeddings.py` | `build_embeddings()` / `query_embeddings()` over `sentence-transformers`, with correct BGE/E5 query-vs-passage prefixing baked in. |
| `rag/vector_store.py` | `FaissVectorStore`: create/save/load/update/delete over a FAISS `IndexIDMap(IndexFlatIP)`. Chroma recognized but raises `NotImplementedError` (not built yet, per the brief's stated priority). |
| `rag/retriever.py` | `Retriever.retrieve_context(query, top_k, ...)` → `{"query", "contexts", "scores"}`, plus document ingestion (`build_index_from_documents`). |
| `rag/data/aero_rentals_kb.json` | 10-document test knowledge base extending the README's "AeroRentals Pro" persona with facts the bare persona prompt doesn't contain (cancellation policy, deposits, insurance, license requirements, late fees, weather policy, etc). |
| `assets/test/aero_rentals_question_cancellation.wav` (+ `.txt`) | A synthesized (Windows SAPI) spoken question — *"Hi, I need to cancel my drone rental tomorrow morning. What is your cancellation policy?"* — used as the offline.py input for the A/B experiment. Placed under `assets/test/`, not `rag/data/`, because the repo's `.gitignore` blanket-ignores `*.wav` except under `assets/**` — see Section 3c below. |
| `rag/build_index.py` | CLI/function: knowledge-base JSON → embeddings → FAISS index → saved to disk. |
| `rag/logging_utils.py` | `RequestLogRecord`/`RequestLogger` (JSONL per-request log) + `inspect_kv_cache()` (best-effort, defensive, read-only `RingKVCache` introspection for logging). |
| `rag/benchmark.py` | `TurnBenchmark` + `summarize()` (mean/p50/p95 over retrieval/injection/generation/total latency). |
| `rag/server_integration.py` | `RAGSession` — the glue connecting `TokenInjector` + `Retriever` + `RequestLogger` to a live `LMGen`. `inject_persona_compatible_knowledge()` (Mode C, blocking) and `queue_injection()`/`consume_one_tick()` (incremental, reserved for D/E/F). |
| `rag/tests/test_retriever.py`, `test_logging_utils.py`, `test_benchmark.py`, `test_server_integration.py` | New unit tests (51 total across the whole `rag/` suite, all passing). |
| `moshi/moshi/offline.py` (patched) | New optional `--rag-enable/--rag-index/--rag-query/--rag-top-k/--rag-embedding-model/--rag-log-dir` flags. Off by default; behavior is unchanged when `--rag-enable` is not passed. |
| `moshi/moshi/server.py` (patched) | New optional `--rag-enable/--rag-index/...` server flags + a per-connection `rag_query` query-string param. `ServerState.rag_session` stays `None` unless `--rag-enable` is passed; both new call sites (connection-start injection, and an `opus_loop` per-tick hook reserved for D/E/F) are guarded by that `None` check. |
| `docs/PERSONA_CACHE_SNAPSHOT_INVESTIGATION.md` | The requested, separate investigation into `save_streaming_state`/`load_streaming_state` for faster persona startup — concluded **not** worth building yet (see that doc for why). |

## 2. New architectural finding: there is no ASR anywhere in this pipeline

While wiring Mode C's trigger point, it became necessary to pin down exactly what "the user's
query" means in PersonaPlex's runtime. The answer: **nothing**. Tracing every text-producing code
path (`server.py: opus_loop`, `LMGen.step`/`process_transformer_output`) shows the *only* text ever
available is the model's own sampled output tokens — there is no speech-to-text of the user's
incoming audio anywhere. The user's voice is only ever turned into Mimi audio *codes*, never into
words PersonaPlex (or our code) can read.

**Consequence**: a literal reading of "retrieve based on what the user just said" (which the
original Modes D/E descriptions implied) is not implementable today without bolting on a real ASR
component (e.g. faster-whisper) listening to the same PCM stream — a substantial new dependency,
out of scope for this increment and not something to silently build. Mode C's connection-start
design (Phase 1 report, Section 6) already sidesteps this — the query is supplied once, explicitly,
at connection/run start (mirroring how `text_prompt`/`voice_prompt` already work) — which is exactly
why Mode C, not D or E, was the right one to validate first. **Modes D/E's "trigger on user query"
framing will need to be revised** when we get to them: either (a) scope them to a turn-boundary
*signal* (which the VAD-based `rag/turn_detector.py` already supports, no ASR needed) carrying a
fixed/pre-supplied knowledge update rather than a per-utterance retrieval query, or (b) explicitly
add ASR as a new, separately-flagged dependency. Flagging this now rather than discovering it
mid-implementation of D/E.

## 3. What was actually validated, and how (be precise about this)

This work was done on a Windows dev machine with **no GPU, no CUDA, and none of `torch`'s
PersonaPlex-relevant siblings installed (`sentencepiece`, `aiohttp`, `sphn`'s consumers, the gated
HF model weights)** — i.e., the real `LMGen`/`LMModel` cannot run here at all. Everything below is
scoped honestly around that constraint:

### Validated for real, with real libraries, right now (reproducible — see the commands)

- **Retrieval pipeline is genuinely real, not mocked.** Installed `faiss-cpu` + `sentence-transformers`
  locally, downloaded the real `BAAI/bge-small-en-v1.5` model from Hugging Face (no gating, public
  model), built a real FAISS index over the 10-document AeroRentals KB, and ran real queries:

  ```
  Q: How much is the deposit for the premium drone?
    [0.778] A refundable security deposit is required at pickup: $150 for the PhoenixDrone X and $300...
  Q: What is your cancellation policy if I need to cancel last minute?
    [0.639] Cancellations made more than 24 hours before the scheduled pickup time receive a full refu...
  ```
  With `top_k=2` one query ("What happens if I return the drone late?") initially missed the
  intended late-fee document (it ranked 3rd at score 0.645, just behind two 0.662 matches) — a real,
  honest retrieval-quality observation, not hidden. Re-checked with the actual configured default
  `TOP_K=5`: the correct document **is** included. This is a genuine (if small) finding: retrieval
  quality with a 10-document corpus and a "small" embedding model is good but not perfect, and
  `TOP_K` matters more than the score alone might suggest for borderline queries.
- **The injection control-flow contract is proven correct in isolation**: 51 unit tests (`rag/tests/`),
  using plain-Python stand-ins for `LMGen` and the tokenizer, assert that `TokenInjector`/`RAGSession`
  step exactly one forced token per call, that incremental and blocking injection produce identical
  token sequences, that `reset_streaming()` is never called, and that KV-cache introspection
  degrades gracefully (never raises) when the expected internal attributes aren't present.
- **`offline.py`/`server.py` patches are syntax-checked** (`python -m py_compile`) and manually
  re-read line-by-line against the original file to confirm every new code path is gated behind
  `rag_enable`/`self.rag_session is not None`, so `ENABLE_RAG=False` (or omitting `--rag-enable`)
  provably reproduces the original control flow.

### NOT validated here — requires the real RunPod RTX 5090 pod (this is your next step)

The actual claim this whole project hinges on — **"Mode C's injected knowledge changes what
PersonaPlex says, without resetting the connection"** — can only be checked by running the real
7B model. I cannot do that from this machine. The notebook (`PersonaPlex_RunPod_RTX5090.ipynb`,
new Sections 19-21) is built to make this a single, scripted, reproducible A/B experiment once you
run it there:

1. Section 19 installs `faiss-cpu`/`sentence-transformers`, builds the FAISS index from
   `rag/data/aero_rentals_kb.json`, and sanity-checks retrieval (this part will reproduce the local
   results above, just on the pod).
2. Section 20 runs `moshi.offline` **twice**, same seed/voice/persona/input audio both times:
   once with no RAG flags (baseline) and once with `--rag-enable --rag-index ... --rag-query "..."`
   (Mode C). Both transcripts and audio are displayed side by side.
3. Section 21 loads the JSONL log Mode C wrote and reports retrieval/injection latency.

**What to look for**: the baseline transcript has no way to correctly state AeroRentals' actual
cancellation terms (24-hour cutoff, 50% fee) since that fact is absent from the bare persona prompt
in `README.md` — at best it should guess generically or deflect. If the Mode C transcript states
that specific policy (even approximately), that is the experimental proof requested. If it doesn't,
the benchmark/log cells will show whether retrieval found the right document (likely yes, per the
local validation above) or whether the injected tokens simply failed to influence generation
(which would be a real, interesting negative result about how PersonaPlex weighs the persona prompt
vs. an injected mid-prompt knowledge block — worth its own report section if it happens).

## 3b. First live-pod run hit a real bug: GPU contention with the still-running server

Your first run on the RTX 5090 pod failed both the baseline and Mode C cells with
`torch.OutOfMemoryError`, before reaching any RAG-specific code. Root cause: `moshi.offline` loads
its own full copy of the 7B model in a *separate OS process*, and Section 10's live server
(`server_proc`) was still running in the background holding its own full copy (~19 GiB of the
31.36 GiB card, per your traceback) -- two full model copies don't fit on one RTX 5090
simultaneously. This wasn't a RAG bug; it would have failed identically for any offline.py
invocation while the server is up.

**Fixed**: added a new cell ("Free GPU memory before running the offline A/B experiment", just
before Section 20's Run A) that detects and stops `server_proc` if it's still alive, plus an
`nvidia-smi` memory check printed immediately after, so any future OOM at this point is
diagnosable from the cell output directly rather than several cells later. Re-run Section 10 to
restart the live server afterward if you still want the web UI.

## 3c. Second live-pod run hit a real bug: the question WAV never made it to the pod

After fixing 3b, the baseline run got past model loading and the persona/voice prompt phase, then
failed at `lm_load_audio(input_wav, ...)` with `No such file or directory` for
`rag/data/aero_rentals_question_cancellation.wav`. Root cause: this repo's `.gitignore` has a
blanket `*.wav` rule, with only `assets/**` explicitly re-included (`!assets/` / `!assets/**`).
The question WAV was placed under `rag/data/`, outside that exception, so whatever git-based
mechanism moved this repo onto the RunPod pod silently dropped it — every other new file in `rag/`
(`.py`, `.json`) is untouched by `.gitignore` and made it through fine, which is why the failure
only affected this one binary asset.

**Fixed**: moved `aero_rentals_question_cancellation.wav` (and its `.txt` companion) from
`rag/data/` to `assets/test/`, alongside the repo's existing `input_assistant.wav`/
`input_service.wav`/`prompt_service.txt` — the same convention already proven to survive a clone
(Section 12's offline smoke test has used `assets/test/input_assistant.wav` successfully from the
start). Updated the notebook's `AERO_QUESTION_WAV` path and the Section 20 markdown accordingly.
No other binary assets exist under `rag/` today, so this was the only file affected.

## 3d. Padded-WAV re-run: experimental proof obtained, with one fidelity caveat

Full results and analysis are in the conversation record; summary:

- **Baseline** confidently stated *"We don't have a cancellation policy. Just bring it back on
  time..."* — a clean confabulation caused by the fact being absent from the bare persona prompt.
- **Mode C** correctly stated both core numeric facts: *">24 hours before pickup → full refund"*
  and *"within 24 hours → 50% fee"*, matching the KB exactly. This is the proof requested: Mode C
  measurably changes and improves factual correctness, via the live, never-reset connection.
- **Caveat**: Mode C's recitation of the third (compound/contrastive) clause inverted the outcome
  -- it said no-shows lose "the full rental plus the deposit," but the KB says the deposit *is*
  refunded for no-shows. The two simple threshold facts transferred correctly; the one clause with
  a "but not X" structure didn't. Likely cause: PersonaPlex is fine-tuned for natural
  conversation, not extractive recitation, so it paraphrases injected knowledge in its own words --
  which is reliable for simple facts and failure-prone on compound ones. Worth keeping in mind for
  any production use of this mechanism; out of scope to fix in this increment.

**Fixed `generation_latency_s`/`final_answer` always being `null`** (flagged as a known gap after
the first run): `RAGSession.inject_persona_compatible_knowledge` no longer logs immediately --
it returns an unfinalized record, and the new `RAGSession.finalize_and_log(record,
generation_latency_s=..., final_answer=...)` writes the single complete JSONL row once the caller
knows the generation-phase outcome. `offline.py` now times its bounded generation loop and passes
both fields in; `server.py`'s connection-start call site finalizes immediately with neither (there
is no bounded "generation phase" in a live duplex conversation -- both fields correctly stay
`None` there). 55 unit tests now pass (4 new, covering both finalize-immediately and
finalize-with-generation-data paths, plus that the log stays empty until `finalize_and_log` runs).

## 4. Recommendation

Per your instruction, **do not proceed to Modes B/D/E/F yet**. Next action is yours: run Sections
18-21 of the updated notebook on the RunPod RTX 5090 pod and report back the two transcripts (or
just confirm whether Mode C's transcript correctly reflects the cancellation policy). That result
determines what comes next:

- If Mode C clearly works: proceed to Mode B (the negative-control baseline — expected to show the
  same retrieval succeeding but the naive prompt template failing to help, which is the point) and
  start scoping the ASR question for D/E per Section 2 above.
- If Mode C does not change the output: the most likely causes, in order of likelihood given the
  architecture, are (a) the injected knowledge block being too long relative to the model's
  attention to the persona-prompt region specifically (worth testing shorter, single-fact
  injections), or (b) the model weighing newly-injected text lower than the original persona prompt
  because of recency/position effects in training data — both are real research questions worth a
  dedicated debugging pass before concluding the mechanism doesn't work at all.

## 5. Addendum: Mode B implemented (negative-control complete)

Mode C was confirmed working end-to-end (Section 3d). Per instruction, proceeded to Mode B to
complete the A/B/C comparison.

**Implementation**: `RAGSession` was refactored to share a `_retrieve_for_injection()` step between
Mode B and Mode C (same `query`/`top_k`/`score_threshold` call, identical retrieved facts) and a
shared `_run_injection()` measurement step, so the *only* code-level difference between the two
modes is the text template handed to `TokenInjector`:

- Mode C wraps the retrieved facts in `<system>...<system>` (same as the persona prompt).
- Mode B (`RAGSession.inject_standard_prompt_rag`) builds `"Relevant Knowledge:\n<facts>\n\nUser
  Question:\n<query>\n\nUse the knowledge above when answering."` with no `<system>` wrapping.

Both `moshi/moshi/offline.py` and `moshi/moshi/server.py` gained a `--rag-injection-mode
{persona_rag,prompt_rag}` flag (default `persona_rag`, so existing notebook cells/commands are
unaffected) that selects between the two at the same connection-start call site. 8 new unit tests
cover the naive template's exact format, that it's *not* `<system>`-wrapped, and that Mode B/Mode C
retrieve identically while diverging only in injected text (63 tests total, all passing).

**Notebook**: Section 20 now runs all three (Mode A baseline, Mode C, Mode B) against the same
seed/voice/persona/padded-WAV, and Section 21's benchmark report is grouped per mode.

## 6. A/B/C result: retrieval is not the bottleneck, injection format is

Run on the real RTX 5090 pod. Decoded transcripts (stripped of `PAD`/`EPAD` control tokens):

- **Mode A**: *"...We don't have a cancellation policy. Just bring it back on time..."* (confabulated)
- **Mode C**: *"...Cancellations made more than 24 hours before pickup get a full refund. If it's
  within 24 hours, there's a 50% fee. And no shows lose the full rental plus the deposit..."*
  (correct on the two core numeric facts; the no-show clause is still subtly wrong, per Section 3d)
- **Mode B**: *"...Sure, I can help with that. Just to confirm, your reservation is for
  tomorrow?"* (never states the policy at all)

The benchmark log confirms Mode B and Mode C retrieved **identically** -- same 5
`retrieved_contexts`, same scores to the decimal (the shared `_retrieve_for_injection()` refactor
is doing its job: this is a controlled comparison, retrieval is not the variable). The only
difference was the injection template, and the result is unambiguous: Mode B doesn't just answer
incorrectly, it doesn't engage with the retrieved facts at all, defaulting to a generic
clarifying question instead -- a worse outcome than even Mode A's wrong-but-attempted answer.

Two secondary observations:
- Mode B also took noticeably longer to start speaking (a much longer leading silence than A/C),
  suggesting the out-of-distribution prompt structure disrupts conversational *timing*, not just
  content.
- Per-token injection cost was consistent across modes (~25.3ms/token for both 340 and 371
  injected tokens), a good sanity check that the latency measurement methodology is sound and
  that this cost is architectural, not content-dependent.

**This is the headline finding of the project so far**: with retrieval held constant and proven
identical, injection-format compatibility with PersonaPlex's own training distribution -- not
retrieval quality -- is what determines whether retrieved knowledge actually gets used. This
directly confirms the hypothesis from `docs/ARCHITECTURE_REPORT.md` Section 6 and is the strongest
evidence yet for prioritizing persona-compatible injection (and its incremental/cache-aware
variants, Modes D/E/F) over naive prompt-template approaches in any further work.

## 7. Mode D implemented (turn-boundary-triggered incremental injection)

**Design** (per the ASR-gap reframing from Section 2): Mode D does not retrieve a fresh query per
turn -- there is no transcript of what the user said to retrieve against. Instead,
`RAGSession.prepare_turn_injection_knowledge(query)` retrieves **once**, at connection start, using
a new, deliberately small `RAGConfig.turn_injection_top_k` (default 2, vs. `top_k`'s default 5) --
Mode C's own benchmark showed ~25ms per injected token, so a 5-document/340-token block costs ~8.5s
per injection, far too slow to repeat every time the user pauses. The resulting short knowledge
block is held, not injected, until `rag/turn_detector.py`'s `TurnBoundaryDetector` (fed raw PCM via
the new `RAGSession.observe_user_frame()`) detects a pause, at which point it's queued via the
existing `queue_injection()`/`consume_one_tick()` incremental mechanism (already built for this
purpose in the Phase 2 design, now finally exercised by a real mode). `observe_user_frame()`
deliberately refuses to queue a second injection while one is still draining
(`self.pending_job is None` check), so a chatty user pausing repeatedly can't stack unbounded
injections.

Wired into both `moshi/moshi/offline.py` (feeding the *raw* `user_audio` array, sliced in lockstep
with the existing encode/step loop -- `lm_encode_from_sphn` only exposes already-Mimi-encoded
tokens, so a separate `frame_idx` counter re-slices the original array independently) and
`moshi/moshi/server.py` (feeding `opus_loop`'s raw `chunk` before its conversion to a torch
tensor). New `--rag-injection-mode=turn_injection`, `--rag-vad-enable`,
`--rag-turn-injection-top-k` flags on both, all additive.

**Real calibration finding**: tested the default `TurnDetectorConfig` against the actual
synthesized-speech WAV used in the notebook (not just synthetic test tones) and found the original
~480ms silence-hangover default fired **twice during the spoken question itself** (at 4.08s and
7.04s, before the question even finished at ~7.42s) -- a natural pause after the comma in "Hi, I
need to..." was long enough to trigger a premature boundary. Swept hangover values against the
real file and found 1.2s (15 frames) clears that pause while still firing reliably (once, at 7.76s)
once the speaker actually stops. Updated `TurnDetectorConfig`'s default from 6 to 15 frames with
this measurement documented in the code comment. This is exactly the failure mode the "lightweight
heuristic, not a learned VAD" design choice flagged as a risk -- now empirically confirmed and
tuned against one real recording, not just asserted.

70 unit tests now pass (7 new, covering: VAD-disabled no-op, pre-preparation no-op, boundary
queuing, `turn_injection_top_k` actually being used instead of `top_k`, no-stacking while a job
drains, full incremental drain + completion logging, and re-firing after a previous injection
finishes).

**Notebook**: Section 20 gained "Run 4 -- Mode D", reusing the same padded WAV (its 10s trailing
silence is exactly the kind of pause Mode D is designed to react to) and the same
persona/voice/seed as the other three runs. Section 21's benchmark report now also reports Mode
D's two distinct log-row types: a `turn_injection` setup row (retrieval only, no tokens forced) and
one or more `incremental (per-tick, opus_loop)` completion rows (one per turn boundary that
finished draining).

**Not yet run against the real model** -- next step is the same pattern as B/C: run Section 20's
new Run 4 cell on the RunPod pod and compare Mode D's transcript to Mode C's. The interesting
question this time isn't just "does it state the policy correctly" but "does injecting mid-stream,
while the agent has already started speaking, work as well as injecting before it starts at all."

## 8. Mode D real run: corruption bug fixed, but a new failure mode found (concluded, not pursued further)

The incremental design above hit a critical bug on its first real run -- forced text leaked
verbatim into the spoken transcript (`<system>` tags, raw KB text, a SentencePiece artifact), and
the injection never finished draining. Full root-cause analysis and the fix (replace per-tick
interleaving with a self-contained burst -- mechanically identical to Mode C's burst, just
triggered later, by a detected pause instead of by connection start) are in
`docs/MODE_D_REDESIGN.md`. `TokenInjector.run_to_completion_async` added for the live server so the
burst doesn't freeze `recv_loop`/`send_loop` for its ~3-9s duration; `offline.py`/`server.py`
updated to fire it on a detected boundary instead of queuing for per-tick consumption; 75 unit
tests passing.

**Re-run on the real RTX 5090 pod confirmed the burst fix works as designed**: no raw `<system>`
tags or verbatim KB text appeared anywhere in Mode D's transcript this time, for any of the four
runs. The forced burst tokens are correctly invisible to the transcript (nothing decodes/forwards
`step()`'s output during the burst, exactly like Mode C/B's connection-start burst), and
`injected_token_count=149` for the 542-char/2-document block matches the expected ~0.27
tokens/char ratio seen in Mode B/C -- the right content was forced, not corrupted.

**But a second, distinct problem surfaced once the literal leak was out of the way.** Mode D and
Mode A are byte-identical up to `"...Oh, I'm sorry. We"` -- strong evidence of the same run
diverging at exactly the burst point. Right there, instead of continuing its sentence or grounding
in the injected facts, Mode D's transcript jumps to *"We> Hello, thank you for calling AeroRentals
Pro. This is Tomaz. How can I help you today?"* -- the model abandons what it was saying and
re-samples a **fresh greeting**, as if a new call had just started. The injected cancellation/
weather/pickup facts never appear in any form in the visible response.

**Working hypothesis**: `prepare_turn_injection_knowledge` wraps the burst in `<system>...<system>`
tags (`wrap_system_tags=True`, [rag/server_integration.py:263](../rag/server_integration.py#L263)) --
the exact same format `step_system_prompts` uses exactly once, at connection start, before the
model has ever spoken. That is the only context in which the model ever saw a `<system>` block
during training/persona setup. Forcing that identical pattern again mid-call is plausibly
interpreted by the model as "a call is starting" rather than "background knowledge to fold into
the current sentence" -- which would explain a re-greet specifically, not just generic
disruption. This reframes Section 6's headline finding: Mode C/B's outcomes don't only depend on
format vs. no-format, but also on `<system>` tags being used at the one position in the
conversation where they are actually in-distribution (the very start). Mid-call injection may need
a different convention entirely (e.g. no wrapping at all, or a different marker the model was
never trained to associate with call-start).

**Decision (per instruction): record this as a documented limitation and do not pursue a fix
within Mode D.** Mode D is concluded for this project at: *corruption-free, but does not ground --
it derails the conversation instead.* This is a real, useful negative result, not a dead end to
hide -- it sharpens what Modes E/F need to get right (any per-turn or periodic injection scheme
will need to avoid the same out-of-distribution `<system>`-block-mid-call pattern, or explicitly
test whether the same derailment occurs with a different wrapping convention).

## 9. Mode E implemented (fixed-interval injection, deliberately without `<system>` wrapping)

**Design**: mechanically identical to Mode D's fix -- a self-contained burst via the same
`TokenInjector.run_to_completion`/`run_to_completion_async`, never interleaved with the real
generation loop -- but triggered by a fixed wall-clock interval (`RAGConfig.
dynamic_injection_interval_s`, default 30s) instead of a detected pause, and with one deliberate
change: the burst is built as a **plain** knowledge block (`wrap_system_tags=False`, no Mode-B-style
"Relevant Knowledge:/User Question:" framing either -- there's no specific question to frame against
a periodic, conversation-state-independent re-fire). This directly tests Section 8's hypothesis:
was Mode D's re-greet derailment caused specifically by the `<system>...<system>` tag reading as
"a call is starting," or is mid-call forced injection fragile regardless of format?

`RAGSession.prepare_dynamic_injection_knowledge(query)` retrieves once (using the new
`RAGConfig.dynamic_injection_top_k`, default 2, same "keep it small" reasoning as
`turn_injection_top_k`) and starts a wall-clock timer. `RAGSession.tick_dynamic_injection()` --
called once per real audio frame, a no-op in any other mode -- returns `True` once
`dynamic_injection_interval_s` has elapsed and resets the timer; like Mode D's `observe_user_frame`,
it only *detects*, the caller fires `fire_dynamic_injection_burst()`/`_async()` before processing
that frame any further. The actual burst-fire-and-log body is now shared between Modes D and E via
a new private `RAGSession._fire_prepared_burst`/`_fire_prepared_burst_async` helper (both modes'
public methods became one-line wrappers around it -- no behavior change to Mode D, confirmed by the
existing Mode D tests still passing unmodified).

Wired into both `moshi/moshi/offline.py` (a second `tick_dynamic_injection()` check alongside
Mode D's `observe_user_frame()` check, both no-ops unless their respective mode is active) and
`moshi/moshi/server.py`'s `opus_loop` (same pattern, `await`-ed). New `--rag-dynamic-injection-interval-s`/
`--rag-dynamic-injection-top-k` flags on both, and `dynamic_runtime` added to both files'
`--rag-injection-mode` choices. 86 unit tests now pass (11 new: tick-before-prepare no-op,
tick-before-interval no-op, tick-after-interval detects without injecting, `dynamic_injection_top_k`
actually used instead of `top_k`, burst forces tokens with no `<system>` tag present in the forced
text, re-firing after a second elapsed interval, plus async-burst equivalents of each).

**Notebook**: Section 20 gained "Run 5 -- Mode E", reusing the same persona/voice/seed/padded-WAV
as the other four runs, with a short demo-only interval (`MODE_E_DEMO_INTERVAL_S = 5.0`, overriding
Section 18's production-realistic 30s default) so the ~17.4s clip actually exercises at least one
fixed-interval re-injection. Section 21's benchmark report gained the same two-row-type handling
for `dynamic_runtime` that Mode D already had.

## 10. Mode E real run: confirms the `<system>`-tag hypothesis, but exposes a deeper limitation

Run on the real RTX 5090 pod, same persona/voice/seed/padded-WAV as A-D, `MODE_E_DEMO_INTERVAL_S =
5.0` (two bursts fired, at ~37.7s and ~52.6s elapsed, 143 tokens each, no `<system>` wrapping).

**The `<system>`-tag hypothesis is confirmed.** Mode E's transcript does **not** re-greet -- it is
byte-for-byte identical to Mode A's confabulated answer (*"...Oh, I'm sorry. We don't have a
cancellation policy. Just bring it back on time and the rental is all good."*) except for two
isolated, harmless `.` tokens appearing in the trailing silence, right around the two burst
points. No leaked text, no derailment, no re-greet. Removing the `<system>` wrapping eliminated
the specific failure mode Mode D had.

**But Mode E still does not ground.** The model doesn't acknowledge the injected facts in any way
-- it produces the *exact same wrong answer* as the unmodified baseline, as if the two 143-token
bursts (fired squarely in the middle of, and after, its response) had no effect on what it chose
to say. This is a different failure mode from Mode D's (silent non-engagement vs. visible
derailment), but it is still a failure to achieve the project's actual goal: grounding a live,
already-started response in newly injected knowledge.

**This sharpens the project's central finding beyond "format matters" (Section 6) and "`<system>`
tags are call-start-coded" (Section 8).** Looking at what actually differs between Mode C (grounds
correctly) and Modes D/E (don't), the deeper variable isn't the `<system>` tag at all -- it's
*timing relative to generation*:

- **Mode C** injects *before* the model has sampled a single token of its response. The entire
  response is generated fresh, with the injected facts already part of the context the response is
  built from.
- **Modes D and E** inject *after* the model has already started generating (and, per these runs,
  already committed to its wrong answer -- "Oh, I'm sorry, we don't have a cancellation policy..."
  is well underway by the time any pause/interval trigger can plausibly fire). Forcing background
  tokens into an in-flight response doesn't give the model a mechanism to revise what it already
  decided to say; the `<system>` tag only changed *how* that fixed trajectory got perturbed at the
  forcing point (derail into a new greeting vs. no visible effect), not *whether* the underlying
  facts got used.

**Implication for any future revisit of D/E**: faster reaction time (firing within the first
fraction of a second after the user's question, before the model has sampled enough of a response
to commit to a wrong answer) is a more promising lever than injection format. Out of scope for this
increment -- recorded here as the reframed conclusion for both Mode D and Mode E, both of which are
considered concluded, informative negative results.

## 11. Mode F implemented (cache-aware benchmark: burst vs. naive reset_and_replay)

Per instruction, with D and E concluded, proceeded to Mode F -- not a new injection mechanism, but
a benchmark of the one mechanism in this project that reliably grounds (Mode C's connection-start
burst) against the obvious alternative an implementation without it would have to fall back to.

**Design**: two arms, both retrieving and injecting the *same* knowledge for the *same* query, so
the only variable is the path taken to get there:

- **Arm 1 (`RAGSession.fire_cache_aware_burst`)** -- identical to Mode C: one `<system>`-wrapped
  burst via `TokenInjector.run_to_completion`, `reset_streaming()` never called. This is what this
  project's mechanism makes possible.
- **Arm 2 (`RAGSession.benchmark_reset_and_replay_baseline`/`_async`)** -- simulates an
  implementation that does *not* have a live-injection mechanism and must, on receiving new
  knowledge mid-call, call `reset_streaming()` (wiping the RingKVCache) and replay the entire
  persona/voice prompt setup from scratch (`LMGen.step_system_prompts`/`_async`, supplied by the
  caller as a closure -- `RAGSession` has no handle on the voice prompt path or persona text)
  before it can inject anything. The whole reset+replay+reinject sequence is timed as one cost,
  since that total is the number that answers "what would this cost without this project's
  mechanism."

Both arms share `_retrieve_for_injection`-equivalent retrieval (using `config.top_k`, same as Mode
C) and the same `_run_injection`/burst-logging plumbing already proven correct by Modes C-E. Arm 2
deliberately runs second, after arm 1, so the live cache going into the rest of the run reflects
arm 2's replay + reinjection -- this is intentional, not an oversight: it means Mode F's run also
produces a transcript that should ground similarly to Mode C's, in addition to the latency
numbers, since arm 2 leaves the same knowledge live by the time the main generation loop runs.

Wired into `moshi/moshi/offline.py` (both arms run back-to-back right after `step_system_prompts`,
the same insertion point as every other connection-start mode) and `moshi/moshi/server.py`'s
connection-start block (arm 2's replay closure uses `step_system_prompts_async`, matching how the
server already does its initial persona/voice prompt). No new config fields needed -- Mode F
reuses `RAGConfig.top_k` like Mode C. 93 unit tests now pass (7 new: arm 1 never calls
`reset_streaming`, arm 1 returns unfinalized like Mode C, arm 2 calls the replay closure exactly
once before injecting, arm 2 self-logs immediately (no `finalize_and_log` needed, unlike arm 1),
both arms retrieve with the same `top_k`, plus async equivalents of the replay-ordering and
non-starving checks).

**Notebook**: Section 20 gained "Run 6 -- Mode F", same persona/voice/seed/padded-WAV as A-E.
Section 21's benchmark report treats Mode F's two rows as a *paired comparison* rather than a
setup/completion pair -- it prints arm 1's and arm 2's `injection_latency_s` side by side and
computes the ratio between them directly (`reset_and_replay costs Nx as much as the burst`).

**Not yet run against the real model** -- next step is the same pattern as B/C/D/E: run Section
20's new Run 6 cell on the RunPod pod. The expected result, given Mode C's own earlier benchmark
(~25ms/injected token plus the full persona/voice prompt forcing cost), is that arm 2 costs
noticeably more than arm 1 -- arm 2 pays for everything arm 1 pays for, plus a full prompt replay.
The interesting number is *how much* more, since that's the concrete, quantified answer to "why
does this project's live-injection mechanism matter at all" that Modes A-E's qualitative
grounding/derailment findings didn't directly measure.

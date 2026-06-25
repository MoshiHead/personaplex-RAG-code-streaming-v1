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

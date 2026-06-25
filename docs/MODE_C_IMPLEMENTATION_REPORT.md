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
| `rag/data/aero_rentals_question_cancellation.wav` (+ `.txt`) | A synthesized (Windows SAPI) spoken question — *"Hi, I need to cancel my drone rental tomorrow morning. What is your cancellation policy?"* — used as the offline.py input for the A/B experiment. |
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

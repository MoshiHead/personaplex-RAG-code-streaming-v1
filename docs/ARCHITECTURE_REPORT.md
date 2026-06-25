# PersonaPlex Architecture Report (Phase 1 Deliverable)

Audience: engineering team planning RAG integration.
Scope: read-only analysis of `moshi/moshi/**`. No code was modified to produce this report.

Primary files inspected: `moshi/moshi/server.py`, `moshi/moshi/offline.py`,
`moshi/moshi/models/lm.py`, `moshi/moshi/models/loaders.py`, `moshi/moshi/models/compression.py`,
`moshi/moshi/modules/streaming.py`, `moshi/moshi/modules/transformer.py`, `moshi/moshi/modules/conv.py`,
`moshi/moshi/utils/compile.py`, `moshi/moshi/quantization/*`.

---

## 0. The single most important finding

**PersonaPlex is not a turn-based chat LLM with a prompt string. It is a continuous, fixed-frame-rate
(12.5 Hz / 80 ms-per-step) full-duplex autoregressive decoder over multiple parallel token streams
(1 text stream + 16 audio codebook streams). There is no point in the runtime code path where a "prompt"
exists as a string or a token tensor that gets prepended to a context window and forward-passed once.**

Every single thing that ever influences the model — the persona/system instruction, the voice prompt, the
user's live speech, and the model's own replies — enters the model through the exact same mechanism:
`LMGen.step(...)`, called once per 80 ms frame, forever, for the lifetime of the WebSocket connection.

This has direct consequences for every RAG injection mode requested in the project brief, documented in
Section 6 below. Read Section 3-5 first if the conclusion in Section 6 looks surprising.

---

## 1. Model architecture report

| Aspect | Value | Source |
|---|---|---|
| Model family | Moshi/Helium-derived full-duplex speech LM, fine-tuned by NVIDIA as "PersonaPlex" | `moshi/moshi/models/lm.py` (`LMModel`), `README.md` |
| Parametrization | Main transformer: `dim=4096`, `num_layers=32`, `num_heads=32`, RMSNorm (f32), RoPE, SiLU gating FFN (`hidden_scale=4.125`). "Depformer" (small per-codebook transformer): `depformer_dim=1024`, `num_layers=6`, `num_heads=16`, weights-per-step. | `moshi/moshi/models/loaders.py: _lm_kwargs` |
| Tokenizer (text) | SentencePiece, 32k vocab, `tokenizer_spm_32k_3.model`, downloaded from the HF repo | `loaders.py: TEXT_TOKENIZER_NAME`, `server.py` |
| Audio tokenizer | "Mimi" neural codec (SEANet encoder/decoder + RVQ), 24kHz, 12.5 Hz frame rate, 8 codebooks used at inference (`set_num_codebooks(8)`) | `loaders.py: get_mimi`, `models/compression.py` |
| Token streams | 1 text stream (`n_q` slot 0) + 16 audio codebook streams total (`n_q=16` declared, `dep_q=16` set at load time) — conceptually: 8 codebooks for "self" (agent) audio + 8 for "other" (user) audio, each with its own per-codebook delay | `loaders.py: _lm_kwargs`, `lm.py: AUDIO_TOKENS_PER_STREAM=8` |
| Inference backend | Plain PyTorch eager execution. Attention is `torch.nn.functional.scaled_dot_product_attention` (SDPA backend selection is whatever PyTorch picks at runtime — no FlashAttention/xFormers package is a declared dependency). Single-step decode is wrapped in `CUDAGraphed` for low dispatch overhead. | `modules/transformer.py: StreamingMultiheadAttention.forward`, `utils/compile.py` |
| Quantization | **None on the language model weights** — loaded as plain `bf16` safetensors (`get_moshi_lm(..., dtype=torch.bfloat16)`). The `quantization/` package implements **Residual Vector Quantization (RVQ)** for the Mimi *audio codec* (`bins=2048`, `n_q=32` codebook entries) — this is unrelated to LLM weight quantization; do not conflate the two when reporting "quantization method" upward. | `loaders.py`, `quantization/core_vq.py`, `quantization/vq.py` |
| "Context window" | **Two different, easily-confused numbers**: main transformer attention `context=3000` *frames* (= 3000 / 12.5 Hz = **240 seconds, 4 minutes**, sliding window, see Section 4); Depformer attention `context=8` (irrelevant for long-range memory — it only looks across the 8 audio codebooks of the *current* frame). | `loaders.py: _transformer_kwargs["context"]`/`_lm_kwargs` (main=3000), `modules/transformer.py: StreamingMultiheadAttention` |
| Frame rate | 12.5 Hz → one full model step (all streams) every **80 ms** | `loaders.py: FRAME_RATE`, `lm.py: FRAME_RATE_HZ` |

---

## 2. Prompt pipeline — where prompts enter, and how they're "tokenized" and "merged"

There is **no system-prompt text block ever concatenated into model input**. Instead:

1. **Text/persona prompt** (`text_prompt`, e.g. *"You work for CitySan Services..."*): the server wraps it with
   `<system> ... <system>` tags (`server.py: wrap_with_system_tags`) and SentencePiece-encodes it into a list of
   token ids **once**, at connection start (`request.query["text_prompt"]` → `lm_gen.text_prompt_tokens`).
   It is then fed **one token per `step()` call** by `LMGen._step_text_prompt_core` (`models/lm.py:1096`),
   with the audio channels forced to silence/sine tokens during those same steps. Each of those steps is a
   completely normal autoregressive transformer step — it grows the real attention KV-cache by one position,
   exactly like a frame of live conversation does.
2. **Voice prompt** (e.g. `NATF2.pt`/a reference WAV): either (a) a WAV is loaded, normalized to -24 LUFS, encoded
   frame-by-frame through the **Mimi** audio encoder, and the resulting audio-codebook tokens are forced into the
   *agent* audio channels via the same `step()` loop (`LMGen._step_voice_prompt_core`), or (b) if a precomputed
   `.pt` file is supplied, the **already-computed transformer hidden-state embeddings** for each of those frames
   are replayed directly into the transformer via `step_embeddings()` (skips Mimi + the codebook embedding lookup,
   but still pays for a full transformer forward pass per frame — see Section 4).
3. **Conversation state**: there is no chat history object, no list of turns, no role-tagged message array.
   `step_system_prompts()`/`step_system_prompts_async()` (`models/lm.py:1117`) runs exactly once per WebSocket
   connection, in this fixed order: `voice prompt frames → ~0.5s silence → text/persona prompt tokens → ~0.5s
   silence`. After that, `handle_chat`'s `opus_loop()` (`server.py:204`) just keeps calling `lm_gen.step()` once
   per incoming 80ms audio frame, forever, until the socket closes. **Nothing is ever rebuilt.** The "prompt" is
   never revisited after these first few hundred milliseconds of the connection.

**Implication:** "inject the persona prompt" and "inject RAG context" are, mechanically, the *same kind of
operation* — feeding a sequence of forced text tokens through `step()`. The only question is *when* (at
connection start vs. mid-stream) and *how many tokens* (latency cost, Section 4) you can afford to inject.

---

## 3. Conversation pipeline (request flow)

```
 Browser mic
     │  PCM audio (WebAudio)
     ▼
 client (Opus-encoded) ── WebSocket binary frames, kind=1 ──► server.py: handle_chat()
                                                                   │
                                                  opus_reader.append_bytes(payload)
                                                                   │
                                                          recv_loop() (asyncio task)
                                                                   │
                                                     opus_reader.read_pcm() [Opus → PCM]
                                                                   │
                                                          opus_loop() (asyncio task, runs forever)
                                                                   │
                                            chunk = next 80ms of PCM  (self.frame_size samples)
                                                                   │
                                            mimi.encode(chunk)  ──► user audio codes  [Mimi encoder = Speech-to-Tokens]
                                                                   │
                                            lm_gen.step(input_tokens=user_codes)
                                                                   │            (this single call IS "Prompt
                                                                   │             Construction + Model Inference":
                                                                   │             there is no separate prompt step)
                                                          ┌────────┴─────────┐
                                                          │ main transformer │  (32 layers, attends over RingKVCache,
                                                          │ (LMModel.forward_│   capacity=3000 frames, see Sec.4)
                                                          │  codes, graphed) │
                                                          └────────┬─────────┘
                                                                   │ transformer_out, text_logits
                                                          ┌────────┴─────────┐
                                                          │ sample text token│  (this token IS the literal output
                                                          │ (top-k/temp)     │   text — there is no separate
                                                          └────────┬─────────┘   "Text Output" stage; sampling
                                                                   │             happens inline, per frame)
                                                          ┌────────┴─────────┐
                                                          │ Depformer (6     │  samples 8 agent audio codebook
                                                          │ layers, per      │  tokens conditioned on the sampled
                                                          │ codebook step)   │  text token + transformer_out
                                                          └────────┬─────────┘
                                                                   │ agent audio codes
                                                          mimi.decode(agent_codes) ──► PCM  [Mimi decoder = Tokens-to-Speech]
                                                                   │
                                                          opus_writer.append_pcm(pcm)
                                                                   │
                                                          send_loop() (asyncio task)
                                                                   │
                                            ws.send_bytes(b"\x01" + opus_bytes)   (audio out, kind=1)
                                            ws.send_bytes(b"\x02" + text_bytes)   (text token out, kind=2, only
                                                                                   when not PAD/BOS/EOS, sent from
                                                                                   inside opus_loop right after
                                                                                   sampling — i.e. text and audio
                                                                                   are emitted from the SAME frame
                                                                                   step, not a separate pipeline stage)
     ▲
     │  Opus-encoded audio + raw text-token bytes, interleaved over one WebSocket
 Browser <── decodes Opus to PCM, plays audio; displays streamed text tokens as a live transcript
```

Key takeaways that don't match the brief's assumed `STT → Prompt Construction → Inference → TTS` pipeline:
- **STT and TTS are not separate models** — Mimi is one codec used bidirectionally (`mimi.encode` for the user's
  audio in, `mimi.decode` for the agent's audio out). There is no Whisper-style transcription step; the "text"
  you see is the LM's own sampled text-codebook output, generated *in lockstep* with the audio.
- **"Prompt Construction" and "Model Inference" are the same call** (`lm_gen.step`). There is no intermediate
  Python-level prompt string at any point during live conversation.
- **Full duplex**: the model is simultaneously "listening" (its `input_tokens`/`other_mimi` channels are fed
  every frame regardless of whether the user is speaking) and "speaking" (its `moshi_tokens` channel is sampled
  every frame regardless of whether it has anything new to say — silence is itself a learned, sampled output,
  not the absence of a step). There is no discrete request/response cycle to hook a "per-turn" RAG call into
  without building a turn-boundary heuristic on top (see Section 6, Mode D/E).

---

## 4. KV cache analysis

There are, confusingly, **two different "caches"** in this codebase. Getting this distinction right is
essential before touching anything RAG-related:

### 4a. `RingKVCache` (`modules/transformer.py:232`) — the *real* attention KV-cache
- One instance per `StreamingMultiheadAttention` layer (so 32 of them in the main transformer + 6 in the
  Depformer, all created in `_init_streaming_state` when `mimi.streaming_forever(1)` / `lm_gen.streaming_forever(1)`
  is called once at server startup, per `ServerState.__init__`).
- **Fixed-capacity ring buffer**: `capacity = context` (3000 for the main transformer). Implemented as a
  `(2, B, H, capacity, D)` tensor (`cache[0]`=keys, `cache[1]`=values) plus a monotonically increasing
  `end_offset` counter. New K/V vectors are written via `index_copy_` at `end_offset % capacity` — i.e., **once
  more than 3000 frames (240 seconds) have been generated, the oldest entries are silently overwritten.**
- Causal masking is `delta = pos_q - pos_k; valid = (pos_k >= 0) & (delta >= 0) & (delta < context)` —
  a genuine **sliding-window attention**, not "infinite history, just slow." Anything older than the window is
  mechanically *unreachable* by attention, full stop, regardless of what's stored in any prompt/text-token list.
- **Updated**: every single call to `state.graphed_main(...)` (i.e., every `lm_gen.step()` call) appends exactly
  one new position to every layer's `RingKVCache`, whether that step's content is real user audio, the system
  prompt, the voice prompt, or (would-be) injected RAG context. There is no "prefill many tokens at once" code
  path — `prepare_step_input` asserts `S == 1` everywhere. Filling N tokens of context costs exactly N sequential
  `step()` calls; there is no batched/parallel prefill kernel available in this implementation.
- **Reused**: continuously, for the entire lifetime of the connection (`reset_streaming()` is only called once,
  right when a new WebSocket connects, in `handle_chat`).
- **Can it be modified?** Mutating tensor *contents* in place (e.g. `state.cache.copy_(...)`, exactly as
  `load_voice_prompt_embeddings` already does for the *other* cache, see 4b) is safe and CUDA-graph-compatible
  (the graph only cares about tensor shape/identity, not values — confirmed by `CUDAGraphed.reset()`'s own
  docstring: "Useful if some shapes have changed, **or external state (e.g. KVCache) has changed**"). But there
  is **no API to splice in K/V vectors for content that was never run through the model** — every entry in this
  cache is the literal output of this exact attention layer for some specific position, with RoPE phase baked
  into the K projection at that position. You cannot fabricate a "knowledge" K/V pair out of an embedding lookup;
  it must come from an actual forward pass at the correct position.

### 4b. `_LMGenState.cache` (`models/lm.py:556`) — **not the attention cache** — the multi-stream delay buffer
- A small circular buffer (`capacity = max_delay + 3`, typically just a handful of frames) holding the *raw token
  ids* for every stream, used purely to implement the per-codebook **delay pattern** (`lm_model.delays`) that lets
  audio codebooks be teacher-forced/sampled slightly out of phase with the text stream. This is what
  `load_voice_prompt_embeddings` restores via `state.cache.copy_(self.voice_prompt_cache)` to skip re-deriving
  the delay bookkeeping after an embedding replay. **This is unrelated to attention and has nothing to do with
  "memory" of conversation content** — don't conflate it with 4a in any report or log message; the project brief
  conflates these ("KV cache appears to be used during streaming inference") and the team should be explicitly
  told they are different things with different lifetimes and different eviction rules.

### 4c. Cost / latency implications (directly answers Phase 7's questions)
- **Cache rebuild cost**: there is no "rebuild" operation cheaper than replaying every frame through
  `step()`. `reset_streaming()` zeroes `end_offset`/`offset` everywhere; repopulating requires re-running
  `step_system_prompts()` (voice + text prompt) from scratch — i.e., exactly the same cost as the original
  connection-start sequence. **Never call `reset_streaming()` mid-conversation just to inject new context** —
  that throws away the entire live `RingKVCache` (all prior conversational context, evicted instantly) and is
  almost certainly the actual mechanism behind the team's "rebuilding prompts every turn introduces latency"
  observation if it's been used to splice in retrieval results.
- **Per-frame latency**: bounded below by one full forward pass through 32 main-transformer layers + 6
  Depformer layers (CUDA-graphed for dispatch efficiency, but still real compute), and must complete within the
  80 ms frame budget to keep the duplex stream real-time. Injecting `N` tokens of RAG context (whether via the
  persona-prompt mechanism or any new mechanism) costs **`N × (per-step latency)`**, contended against the live
  audio frame budget — this is the real, unavoidable origin of "latency growth" in Phase 6/7, not a fixable
  inefficiency.
- **Context retention**: any content injected as forced steps becomes unreachable to attention after 3000 frames
  (4 minutes) of subsequent activity, *by architecture*, independent of injection strategy. A RAG mode that
  injects context once and expects it to stay influential 10 minutes later cannot work without **periodic
  re-injection** — which then competes with the same latency budget repeatedly.

---

## 5. Streaming generation loop

- `LMGen.step()` (`models/lm.py:814`) is the unit of generation: write provided tokens into `_LMGenState.cache`
  at the right delayed position → run the main transformer on the previous position's tokens
  (`state.graphed_main`, CUDA-graphed) → sample/force the text token → run the Depformer once per audio codebook
  (`state.graphed_depth`, also CUDA-graphed) → write sampled tokens back into the cache → once `state.offset`
  has advanced past `max_delay`, emit the fully-resolved frame for the position `max_delay` steps ago.
- This is why output is delayed by `max_delay` frames relative to input — a fixed, small (sub-second) pipeline
  latency inherent to the delay-pattern architecture, separate from any RAG-induced latency.
- `state.warmup()` (`server.py:119`) primes the CUDA graphs with 4 dummy steps right after model load, before
  any real connection — this is one-time per-process cost, not per-turn.

---

## 6. Direct implications for the requested RAG injection modes

This is the most actionable part of the report — it tells us which of the originally proposed Modes (A–F) are
implementable as specified, and which need to be reframed given what the code actually does.

| Mode | As originally specified | Reality check | Recommendation |
|---|---|---|---|
| **A — Baseline** | No RAG | Fully compatible, no changes needed. | Implement as a no-op pass-through; this is also our regression safety net. |
| **B — Standard Prompt RAG** ("Append `Relevant Knowledge: ... User Question: ...`") | Implies a single text string handed to "the model" before inference, like a normal chat LLM. | There is no single inference call to append a string to. The only place a block of text can plausibly go is the **one-time `text_prompt`** at connection start (Section 2) — i.e. this mode can only ever be a *static, connection-start* injection, not a per-question one, unless the connection is torn down and restarted per question (which destroys all live conversational `RingKVCache` state and the persona's own voice/behavior priming — almost certainly why the team observed "the model often ignores dynamically injected context": it was likely being added to a string that the live decode loop never reads, or it required a disruptive reconnect). Also: PersonaPlex was fine-tuned on prompts wrapped in `<system> ... <system>` with natural-language persona instructions — a `"Relevant Knowledge:\n...\nUser Question:\n...\nUse the knowledge above..."` template is out-of-distribution phrasing the model has no learned tendency to attend to. | Keep as a **research baseline that we *expect* to underperform**, exactly to let the experiment prove the hypothesis. Implement it as literally appending the knowledge block, `<system>`-wrapped, into the *connection-start* `text_prompt`. Document expected failure mode in the benchmark report rather than trying to make it work harder than the architecture allows. |
| **C — Persona Compatible RAG** | Inject knowledge "using the exact same mechanism used by PersonaPlex persona/system prompts" | This is actually *correct and implementable as stated* — Section 2 shows the persona prompt mechanism is just "SentencePiece-encode text, force it token-by-token through `step()` before/around the live persona prompt." Folding retrieved knowledge into the *same* `<system>...<system>`-wrapped text block (rather than a separate "Relevant Knowledge" template) is the structurally faithful version of this mode. | **Implement as specified.** This is the most promising mode architecturally and should be the first one benchmarked against B. |
| **D — Conversation Turn Injection** ("immediately before the latest user message") | Assumes a discrete, detectable "user message" boundary. | PersonaPlex has no built-in turn/utterance boundary — speech is continuous duplex audio with the model itself deciding when to talk. "Before the latest user message" must be defined by us (e.g. VAD-based silence detection on the incoming audio, or watching the model's own sampled text stream for a sentence-final token). | **Reframe**: implement turn-boundary detection (simple energy/VAD-based heuristic to start) as an explicit, documented component of this mode, not an assumption. Inject via the same forced-`step()` mechanism as Mode C, at the detected boundary. |
| **E — Dynamic Runtime Context** ("inject at every turn", "evaluate context retention") | Implies repeated injection is "free" if cache is preserved. | Per Section 4c, each injection still costs `N × per-step latency` against the live frame budget every single time, and per the 3000-frame (4-min) sliding window, anything from more than ~4 minutes ago needs re-injection anyway to remain attendable. "Context retention" should be measured as: does the model reference re-injected knowledge correctly, and does perceived audio latency/stutter increase with injection frequency. | **Implement as specified**, but the benchmark must report latency growth as expected/inherent (not a bug to fix), and explicitly test the 4-minute eviction boundary as a quality cliff. |
| **F — Cache-Aware RAG** ("preserve KV cache, append only delta context") | Implies an alternative to "rebuilding the prompt" that's somehow cheaper. | Per Section 4a/4c: **the live `RingKVCache` is *already* append-only/preserved** for anything injected via `step()` without calling `reset_streaming()` — this isn't a new technique to invent, it's the default behavior of the architecture as long as we never tear down the connection. There is no way to skip the per-token forward-pass cost (no K/V-without-forward-pass API exists). | **Reframe the deliverable**: Mode F's prototype should be "inject delta context via forced `step()` calls without ever calling `reset_streaming()`" (which the other modes should also follow), with the experiment measuring that this is strictly cheaper than a naive "tear down and reconnect with a rebuilt prompt" baseline — which IS something a less careful implementation might do, and is the legitimate alternative this mode should be benchmarked against. |

**Bottom line for Phase 6 design**: Modes A, C, D, E, F all reduce to "decide *what* text/context to force-feed
and *when*", built on one shared primitive: *append N forced text-token steps via the existing
`step()`/`_step_text_prompt_core` mechanism, without ever resetting the stream*. Mode B is the deliberate
"naive" baseline expected to underperform C. This sharply simplifies the `injection_manager.py` design in
Phase 2 — it should be one well-tested primitive ("inject token sequence at point X without disrupting the
live cache") with thin per-mode policies on top, not five independent code paths.

---

## 7. What does *not* need to be touched

To satisfy "do not break existing PersonaPlex functionality" concretely:
- `moshi/moshi/server.py`, `offline.py`, `models/lm.py`, `models/loaders.py`, `modules/*` should not require any
  edits for an MVP of Modes A–F. The injection primitive identified in Section 6 can be built as a thin wrapper
  that calls existing public methods (`LMGen.step`, `text_tokenizer.encode`, `wrap_with_system_tags`-equivalent)
  from a *new* `rag/injection_manager.py`, without modifying `LMGen` itself, as long as the server is willing to
  call into it. The only unavoidable touch point is `server.py: handle_chat` (to plug in retrieval + injection
  hooks at connection-start and at runtime) — and even that can be done so that `ENABLE_RAG=False` reproduces the
  current code path byte-for-byte.

---

## 8. Open assumptions (flagged per project constraints)

1. **Turn-boundary detection** (Mode D/E) is not provided by the repo; we will need to add a lightweight
   heuristic (e.g., VAD on the user audio stream, or watching for end-of-utterance in the model's own sampled
   text). This is new functionality, clearly outside "preserve existing pipeline," and will be implemented as an
   isolated, optional module.
2. We assume "RAG knowledge text" will be injected through the **text** channel (same mechanism as the persona
   prompt), not the audio channel — audio-channel injection (like the voice prompt) doesn't make sense for
   factual knowledge.
3. We assume RunPod RTX 5090 deployment (per the existing notebook) remains the target environment; benchmark
   numbers (Phase 8) will be reported for that specific GPU/driver/CUDA combination and won't generalize to other
   hardware without re-running.
4. `gradio`/`accelerate` are already installed per the existing notebook; the RAG stack adds `faiss-cpu` (or
   `faiss-gpu`, TBD in Phase 3) + `chromadb` + `sentence-transformers` as new dependencies — these are additive
   and never imported unless `ENABLE_RAG=True`, satisfying the "should not affect baseline PersonaPlex" requirement.

---

## 9. Recommendation before proceeding to Phase 2

Proceed with the `rag/` package as scoped, with these adjustments to the brief based on the findings above:
- Build the **one shared injection primitive** first (Section 6 bottom line) and unit-test it in isolation
  against a `LMGen` instance before wiring any of Modes B–F on top.
- Treat **Mode B as an intentional negative-control baseline**, not a target to optimize.
- Add a small **VAD/turn-boundary module** as an explicit, separately-documented piece of new functionality for
  Modes D/E (not implied by the existing repo).
- Frame all "KV cache" logging/benchmarking (Phase 7/8) using the 4a/4b vocabulary above so reports don't
  conflate the attention `RingKVCache` with the `_LMGenState` delay buffer.

Awaiting confirmation to proceed to Phase 2 (scaffold `rag/` package) with these adjustments incorporated.

# Investigation: Persona/Voice Cache Snapshots via `save_streaming_state`/`load_streaming_state`

Requested as a side investigation, explicitly **separate from the RAG injection modes** — this is
about speeding up *connection startup* for a persona PersonaPlex already knows how to load, not
about injecting new knowledge. Nothing in this document is implemented; it is a feasibility
analysis only, and a recommendation on whether it's worth building later.

## What exists today

`moshi/moshi/modules/streaming.py` already implements generic save/restore for *any* `StreamingModule`
tree:

- `StreamingModule.save_streaming_state(save_path, metadata_save_path)` — walks every streaming
  child (every `RingKVCache` in the 32 main-transformer + 6 Depformer attention layers, plus the
  `_LMGenState` delay buffer, the `_TransformerState`/`_LayerState` offsets, etc.), flattens them
  into a `safetensors` file + a JSON metadata sidecar.
- `StreamingModule.set_streaming_state_inplace(state_dict)` / `load_streaming_state(path,
  metadata_path)` — the inverse: loads those files and copies the tensors back in place.

This is real, working, generic infrastructure already in the repo. **It is not called anywhere in
`server.py` or `offline.py` today** — `step_system_prompts()` is always replayed from scratch, every
connection.

## Is the post-system-prompt cache state actually deterministic (i.e., snapshot-able)?

Yes. Tracing `LMGen.step()` → `process_transformer_output()`: every step *always* computes a sampled
token from the logits, but for every position during `step_system_prompts()` (voice prompt frames,
both silence gaps, and text prompt tokens), `provided_` is `True` and the actual value written to the
cache is selected via `torch.where(provided_, target_, sampled_...)` — i.e., the **forced** value, not
the sampled one. The model is in `eval()` mode (no dropout), so the forward pass is otherwise
deterministic. Therefore: for a fixed `(voice_prompt, text_prompt)` pair, the resulting
`RingKVCache` contents after `step_system_prompts()` are bit-for-bit deterministic and **do** depend
only on that pair — not on the RNG seed, not on anything from a previous connection (since
`reset_streaming()` zeroes everything first). This is exactly the property you need for snapshotting
to be sound: compute once per distinct `(voice_prompt, text_prompt)`, reuse forever.

## The catch: snapshot size is governed by ring *capacity*, not by how much was actually used

`RingKVCache.asdict()` returns `{"cache": self.cache, "end_offset": self.end_offset}` — and
`self.cache` is the **full** `(2, B, H, capacity, D)` tensor, always, regardless of how many frames
are actually filled (`end_offset`). For the main transformer, `capacity=3000` per layer × 32 layers
≈ **1.6 GB** (see `docs/STREAMING_AND_INJECTION_DESIGN.md` Section 2's memory estimate) — and a
typical persona+voice prompt fills only a tiny fraction of that (a voice-prompt WAV is usually a few
seconds = tens of frames; a one-sentence persona instruction is a few dozen text tokens; plus two
~0.5s silence gaps). Using `save_streaming_state`/`load_streaming_state` **as they exist today**
would serialize/deserialize the entire ~1.6 GB+ buffer per saved persona, almost all of it unused
zero-padding.

**This means the off-the-shelf utility, used naively, would likely not be a net win**: copying
~1.6 GB of tensor data from disk (or even from a warm OS page cache) to GPU memory is not obviously
faster than just replaying perhaps 100-300 forced `step()` calls at native (CUDA-graphed,
sub-10ms-per-step once warmed) speed — a rough back-of-envelope puts the *current* `step_system_prompts`
cost at roughly 1-3 seconds wall-clock for a typical persona+voice prompt, which a multi-GB
snapshot load could easily match or exceed depending on storage speed.

## What a worthwhile version would require (not built, scoped only)

A genuinely faster approach would need new code (outside `moshi/moshi/**`, e.g. a small dedicated
helper module) that:

1. On save: for each `RingKVCache`, slice only `cache[:, :, :end_offset]` (and `end_offset` itself)
   rather than calling the generic `save_streaming_state`, cutting snapshot size roughly
   proportional to actual prompt length instead of fixed capacity.
2. On load: zero a freshly-allocated full-capacity buffer, copy the saved slice into the first
   `end_offset` positions, and set `end_offset` accordingly, plus restore `_LMGenState`'s
   `cache`/`provided`/`offset` and per-layer `_LayerState.offset_cpu`/`_TransformerState.offset` the
   same way.
3. Key the cache by a hash of `(voice_prompt_identity, text_prompt_string)` so a persona swap
   correctly invalidates/selects a different snapshot.

This is a contained, well-defined piece of work, but it duplicates/bypasses the existing generic
utility rather than reusing it as-is, and its payoff is bounded by how slow `step_system_prompts`
actually turns out to be in practice on the real RTX 5090 deployment (not yet measured against the
real model in this increment — see `docs/MODE_C_IMPLEMENTATION_REPORT.md`).

## Recommendation

- **Do not build this now.** It optimizes connection *startup* latency, which is orthogonal to the
  RAG injection modes (which optimize/measure *mid-conversation* knowledge injection) and the
  current project priority is proving Mode C works at all.
- **Worth revisiting later if**: the live RunPod measurement of `step_system_prompts`'s actual
  wall-clock cost (to be captured once Mode C's benchmark cells are run for real, per
  `docs/MODE_C_IMPLEMENTATION_REPORT.md`) turns out to be large enough (e.g. several seconds or
  more, perhaps because of large voice-prompt WAVs or slower hardware) that the engineering cost of
  the slice-based snapshot approach above would clearly pay for itself — and even then, only for
  deployments that reuse a small, fixed set of personas repeatedly (e.g. a kiosk/IVR-style product),
  not for this research notebook's varied-persona experimentation.

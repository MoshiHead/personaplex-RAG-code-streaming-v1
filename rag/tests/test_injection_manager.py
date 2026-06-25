"""
Unit tests for rag.injection_manager, using plain Python stand-ins for `LMGen` and the
SentencePiece tokenizer -- no torch, no GPU, no real model required. These tests exist to prove
the *control flow contract* (token-by-token stepping, never resetting state, incremental vs.
blocking usage) is correct in isolation, which is the whole point of designing the interface with
no hard dependency on `moshi`/`torch` in the first place.
"""

import unittest

from rag.injection_manager import (
    InjectionRequest,
    TokenInjector,
    wrap_with_system_tags,
)


class FakeLMGen:
    """Records every call to `step()` so tests can assert on exactly what was forced through,
    without needing a real LMModel/transformer."""

    def __init__(self):
        self.calls = []
        self.reset_streaming_called = False

    def step(self, input_tokens=None, moshi_tokens=None, text_token=None, return_embeddings=False):
        self.calls.append(
            {
                "input_tokens": input_tokens,
                "moshi_tokens": moshi_tokens,
                "text_token": text_token,
            }
        )
        return None

    def reset_streaming(self):
        # Real LMGen.reset_streaming() wipes the whole RingKVCache -- TokenInjector must never
        # call this; tests assert on `reset_streaming_called` staying False throughout.
        self.reset_streaming_called = True


class FakeTokenizer:
    """Deterministic stand-in for sentencepiece.SentencePieceProcessor: maps each character to
    its ordinal value, so token sequences are trivially predictable in assertions."""

    def encode(self, text: str) -> list:
        return [ord(ch) for ch in text]


class TestWrapWithSystemTags(unittest.TestCase):
    def test_wraps_plain_text(self):
        self.assertEqual(
            wrap_with_system_tags("You are a helpful teacher."),
            "<system> You are a helpful teacher. <system>",
        )

    def test_idempotent_on_already_wrapped_text(self):
        already = "<system> already wrapped <system>"
        self.assertEqual(wrap_with_system_tags(already), already)

    def test_strips_surrounding_whitespace_before_wrapping(self):
        self.assertEqual(
            wrap_with_system_tags("  spaced out  "),
            "<system> spaced out <system>",
        )


class TestTokenInjectorBlocking(unittest.TestCase):
    def setUp(self):
        self.lm_gen = FakeLMGen()
        self.injector = TokenInjector(
            lm_gen=self.lm_gen,
            text_tokenizer=FakeTokenizer(),
            make_zero_audio_frame=lambda: "ZERO_AUDIO",
            make_silence_audio_frame=lambda: "SINE_AUDIO",
            zero_text_code=3,
        )

    def test_run_to_completion_forces_one_step_per_token(self):
        request = InjectionRequest(text="hi", mode="persona_rag", wrap_system_tags=False)
        stats = self.injector.run_to_completion(request)

        expected_tokens = [ord("h"), ord("i")]
        self.assertEqual(len(self.lm_gen.calls), len(expected_tokens))
        for call, expected_token in zip(self.lm_gen.calls, expected_tokens):
            self.assertEqual(call["text_token"], expected_token)
            self.assertEqual(call["moshi_tokens"], "ZERO_AUDIO")
            self.assertEqual(call["input_tokens"], "SINE_AUDIO")

        self.assertEqual(stats.mode, "persona_rag")
        self.assertEqual(stats.token_count, len(expected_tokens))
        self.assertEqual(stats.steps_executed, len(expected_tokens))
        self.assertTrue(stats.finished)
        self.assertGreaterEqual(stats.wall_time_s, 0.0)

    def test_never_calls_reset_streaming(self):
        self.injector.run_to_completion(InjectionRequest(text="some knowledge", mode="dynamic_runtime"))
        self.assertFalse(self.lm_gen.reset_streaming_called)

    def test_system_tag_wrapping_changes_token_count(self):
        raw = InjectionRequest(text="hi", mode="prompt_rag", wrap_system_tags=False)
        wrapped = InjectionRequest(text="hi", mode="prompt_rag", wrap_system_tags=True)

        raw_tokens = self.injector.encode(raw)
        wrapped_tokens = self.injector.encode(wrapped)

        self.assertEqual(raw_tokens, [ord("h"), ord("i")])
        self.assertGreater(len(wrapped_tokens), len(raw_tokens))


class TestTokenInjectorIncremental(unittest.TestCase):
    def setUp(self):
        self.lm_gen = FakeLMGen()
        self.injector = TokenInjector(
            lm_gen=self.lm_gen,
            text_tokenizer=FakeTokenizer(),
            make_zero_audio_frame=lambda: "ZERO_AUDIO",
            make_silence_audio_frame=lambda: "SINE_AUDIO",
        )

    def test_step_once_executes_exactly_one_token_at_a_time(self):
        job = self.injector.start(InjectionRequest(text="abc", mode="turn_injection", wrap_system_tags=False))

        self.assertFalse(job.done)
        self.assertEqual(len(self.lm_gen.calls), 0)

        for expected_count in (1, 2, 3):
            executed = job.step_once()
            self.assertTrue(executed)
            self.assertEqual(len(self.lm_gen.calls), expected_count)

        self.assertTrue(job.done)
        self.assertTrue(job.stats.finished)

    def test_step_once_on_finished_job_is_a_safe_noop(self):
        job = self.injector.start(InjectionRequest(text="x", mode="turn_injection", wrap_system_tags=False))
        self.assertTrue(job.step_once())
        self.assertTrue(job.done)

        # Calling step_once again after completion must not raise and must not force another step.
        executed_again = job.step_once()
        self.assertFalse(executed_again)
        self.assertEqual(len(self.lm_gen.calls), 1)

    def test_incremental_and_blocking_paths_force_identical_tokens(self):
        text = "identical content"

        incremental_lm_gen = FakeLMGen()
        incremental_injector = TokenInjector(
            incremental_lm_gen, FakeTokenizer(), lambda: "Z", lambda: "S"
        )
        job = incremental_injector.start(InjectionRequest(text=text, wrap_system_tags=False))
        while not job.done:
            job.step_once()

        blocking_lm_gen = FakeLMGen()
        blocking_injector = TokenInjector(blocking_lm_gen, FakeTokenizer(), lambda: "Z", lambda: "S")
        blocking_injector.run_to_completion(InjectionRequest(text=text, wrap_system_tags=False))

        incremental_tokens = [c["text_token"] for c in incremental_lm_gen.calls]
        blocking_tokens = [c["text_token"] for c in blocking_lm_gen.calls]
        self.assertEqual(incremental_tokens, blocking_tokens)


if __name__ == "__main__":
    unittest.main()

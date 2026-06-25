"""Unit tests for rag.turn_detector. Pure numpy -- no torch/GPU required."""

import unittest

import numpy as np

from rag.turn_detector import TurnBoundaryDetector, TurnDetectorConfig


def _silence(n=1920):
    return np.zeros(n, dtype=np.float32)


def _speech(n=1920, amplitude=0.5):
    rng = np.random.default_rng(42)
    return (rng.standard_normal(n).astype(np.float32)) * amplitude


class TestTurnBoundaryDetector(unittest.TestCase):
    def setUp(self):
        self.config = TurnDetectorConfig(silence_hangover_frames=3, energy_threshold=0.01)
        self.detector = TurnBoundaryDetector(self.config)

    def test_continuous_silence_never_fires(self):
        for _ in range(20):
            boundary = self.detector.push_frame(_silence())
            self.assertFalse(boundary)

    def test_continuous_speech_never_fires(self):
        for _ in range(20):
            boundary = self.detector.push_frame(_speech())
            self.assertFalse(boundary)

    def test_speech_then_sustained_silence_fires_once(self):
        boundaries = []

        for _ in range(5):
            boundaries.append(self.detector.push_frame(_speech()))

        for _ in range(self.config.silence_hangover_frames):
            boundaries.append(self.detector.push_frame(_silence()))

        # Should fire exactly once, on the frame that completes the hangover window.
        self.assertEqual(sum(boundaries), 1)
        self.assertTrue(boundaries[-1])

    def test_does_not_refire_during_continued_silence(self):
        for _ in range(3):
            self.detector.push_frame(_speech())
        fired = [self.detector.push_frame(_silence()) for _ in range(15)]
        self.assertEqual(sum(fired), 1)

    def test_fires_again_after_a_new_speech_segment(self):
        def speak_then_go_silent():
            for _ in range(3):
                self.detector.push_frame(_speech())
            results = [self.detector.push_frame(_silence()) for _ in range(self.config.silence_hangover_frames)]
            return sum(results)

        self.assertEqual(speak_then_go_silent(), 1)
        self.assertEqual(speak_then_go_silent(), 1)

    def test_reset_clears_state(self):
        for _ in range(3):
            self.detector.push_frame(_speech())
        self.detector.reset()
        # After reset, silence alone should not fire (no preceding speech in the new "session").
        fired = [self.detector.push_frame(_silence()) for _ in range(10)]
        self.assertEqual(sum(fired), 0)

    def test_empty_frame_is_treated_as_silence(self):
        boundary = self.detector.push_frame(np.zeros(0, dtype=np.float32))
        self.assertFalse(boundary)


if __name__ == "__main__":
    unittest.main()

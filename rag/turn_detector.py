"""
Lightweight, dependency-free turn-boundary ("user appears to have stopped talking") detector.

This exists only for Mode D (TURN_INJECTION), which needs *some* signal for "inject now" that
isn't a fixed timer. It is deliberately a simple short-term-energy-plus-hangover heuristic rather
than a learned VAD model, so the RAG package adds zero new ML dependencies for this piece. It is
disabled by default (`RAGConfig.vad_enabled = False`) and is fully optional: nothing else in this
package or in baseline PersonaPlex depends on it.

If this heuristic proves too noisy in practice, it is designed to be swapped 1:1 for a real VAD
(e.g. `webrtcvad`, Silero VAD) without touching any caller -- both would implement the same
`push_frame(pcm) -> bool` contract.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TurnDetectorConfig:
    """Defaults are tuned for Mimi's native frame geometry (24kHz audio, ~80ms/frame, i.e. 1920
    samples/frame at the model's 12.5 Hz frame rate) so the detector can be fed the exact same
    chunks `moshi.server.opus_loop` already slices off the incoming PCM stream."""

    sample_rate: int = 24000
    frame_size: int = 1920
    energy_threshold: float = 0.01     # RMS amplitude considered "speech present"
    # ~15 * 80ms = ~1.2s of continuous silence => turn boundary. Empirically calibrated (not just
    # guessed) against a real synthesized-speech WAV
    # (assets/test/aero_rentals_question_cancellation_padded.wav): a shorter hangover (e.g. the
    # previous default of 6 frames / ~480ms) fired on a natural mid-sentence pause (the comma
    # after "Hi,") *before* the question even finished, queuing a turn-injection at the wrong
    # moment. 1.2s clears ordinary intra-sentence pauses while still firing reliably once the
    # speaker actually stops talking.
    silence_hangover_frames: int = 15


class TurnBoundaryDetector:
    """Stateful, single-stream detector. Feed it consecutive user-audio frames in order; it
    returns `True` exactly once per utterance, on the frame that completes a sustained silence
    gap following speech -- a heuristic proxy for "the user just finished a turn."

    Not thread/coroutine-safe for concurrent callers (same constraint as `TokenInjector` -- see
    docs/STREAMING_AND_INJECTION_DESIGN.md, Section 3.1). Intended to be driven from the same
    `opus_loop` execution context that already owns the live audio frames.
    """

    def __init__(self, config: TurnDetectorConfig | None = None):
        self.config = config or TurnDetectorConfig()
        self._was_speaking = False
        self._silence_run = 0

    def reset(self) -> None:
        """Clear all state, e.g. when a new connection/session starts."""
        self._was_speaking = False
        self._silence_run = 0

    def push_frame(self, pcm_frame: np.ndarray) -> bool:
        """Feed one frame of mono PCM audio (float32, roughly `config.frame_size` samples).

        Returns True iff this frame completes a detected end-of-turn boundary.
        """
        if pcm_frame.size == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(np.square(pcm_frame))))

        speaking_now = rms >= self.config.energy_threshold

        boundary = False
        if speaking_now:
            self._was_speaking = True
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._was_speaking and self._silence_run == self.config.silence_hangover_frames:
                boundary = True
                # Require a fresh stretch of speech before the next boundary can fire again.
                self._was_speaking = False

        return boundary

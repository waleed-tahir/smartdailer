import numpy as np
from typing import Generator, List, Tuple
from smartdailer.config import AudioConfig


class AudioChunkBuffer:
    """
    Audio buffer that extracts streaming chunks with a sliding window:
    - Default chunk duration: 2.5 seconds (e.g., 40,000 samples at 16kHz)
    - Default overlap duration: 0.5 seconds (e.g., 8,000 samples at 16kHz)
    - Step / Hop duration: 2.0 seconds (e.g., 32,000 samples at 16kHz)
    """

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.sample_rate = self.config.sample_rate
        self.chunk_samples = self.config.chunk_samples
        self.overlap_samples = self.config.overlap_samples
        self.step_samples = self.config.step_samples

        self._buffer = np.array([], dtype=np.float32)
        self._total_samples_processed = 0
        self._chunk_index = 0

    def add_audio(self, audio: np.ndarray) -> Generator[Tuple[int, float, float, np.ndarray], None, None]:
        """
        Append audio samples to the buffer and yield available chunks:
        Yields tuples of (chunk_index, start_sec, end_sec, chunk_audio_array)
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert multi-channel to mono

        audio = audio.astype(np.float32)
        self._buffer = np.concatenate([self._buffer, audio]) if len(self._buffer) > 0 else audio

        while len(self._buffer) >= self.chunk_samples:
            chunk = self._buffer[:self.chunk_samples].copy()
            start_sec = self._total_samples_processed / self.sample_rate
            end_sec = start_sec + self.config.chunk_duration_sec

            yield (self._chunk_index, start_sec, end_sec, chunk)

            self._chunk_index += 1
            self._total_samples_processed += self.step_samples
            # Advance by step_samples, keeping overlap_samples for the next chunk
            self._buffer = self._buffer[self.step_samples:]

    def flush(self, min_duration_sec: float = 0.2) -> Generator[Tuple[int, float, float, np.ndarray], None, None]:
        """
        Flush any remaining audio in the buffer when utterance ends.
        """
        min_samples = int(min_duration_sec * self.sample_rate)
        if len(self._buffer) >= min_samples:
            chunk = self._buffer.copy()
            start_sec = self._total_samples_processed / self.sample_rate
            end_sec = start_sec + (len(chunk) / self.sample_rate)

            yield (self._chunk_index, start_sec, end_sec, chunk)

            self._chunk_index += 1
            self._total_samples_processed += len(chunk)
            self._buffer = np.array([], dtype=np.float32)
        else:
            self._buffer = np.array([], dtype=np.float32)

    def reset(self):
        """Reset the buffer state for a new turn or conversation."""
        self._buffer = np.array([], dtype=np.float32)
        self._total_samples_processed = 0
        self._chunk_index = 0

    @classmethod
    def slice_audio(
        cls, audio: np.ndarray, config: AudioConfig = None
    ) -> List[Tuple[int, float, float, np.ndarray]]:
        """
        Convenience method to slice a complete audio array into
        2.5s chunks with 0.5s overlap and final remainder.
        """
        buffer = cls(config or AudioConfig())
        chunks = list(buffer.add_audio(audio))
        remainder = list(buffer.flush())
        return chunks + remainder

import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from smartdailer.asr.base import ASRBase
from smartdailer.audio.buffer import AudioChunkBuffer
from smartdailer.brain.base import BrainBase
from smartdailer.config import AudioConfig
from smartdailer.tts.base import TTSBase


class VoiceAgent:
    """
    A Voice Agent encapsulates:
    - Audio ingestion with 2.5s chunk / 0.5s overlap buffer
    - ASR transcription (Whisper or Parakeet)
    - Dialogue Brain (LLM or Scenario)
    - TTS Speech Synthesis (Local Windows SAPI5 or Edge-TTS)
    """

    def __init__(
        self,
        name: str,
        asr: ASRBase,
        tts: TTSBase,
        brain: BrainBase,
        voice_id: str,
        audio_config: Optional[AudioConfig] = None,
    ):
        self.name = name
        self.asr = asr
        self.tts = tts
        self.brain = brain
        self.voice_id = voice_id
        self.audio_config = audio_config or AudioConfig()
        self.chunk_buffer = AudioChunkBuffer(self.audio_config)

    def listen_and_transcribe(
        self,
        incoming_audio: np.ndarray,
        on_chunk_transcribed: Optional[Callable[[int, float, float, str], None]] = None,
        realtime: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """
        Processes incoming speech audio through the 2.5s chunk / 0.5s overlap sliding window.
        If realtime=True, streams frames at wall-clock speed synchronized with speaker audio.
        """
        self.chunk_buffer.reset()
        chunk_results = []
        raw_chunk_texts = []

        if realtime:
            # Stream frames in real-time (e.g. 200ms frame slices)
            frame_size = int(0.20 * self.audio_config.sample_rate)
            samples_fed = 0
            start_wall_time = time.time()

            for i in range(0, len(incoming_audio), frame_size):
                frame = incoming_audio[i : i + frame_size]
                samples_fed += len(frame)

                for chunk_idx, start_sec, end_sec, chunk_audio in self.chunk_buffer.add_audio(frame):
                    text = self.asr.transcribe_chunk(chunk_audio, sample_rate=self.audio_config.sample_rate)
                    chunk_results.append({
                        "chunk_index": chunk_idx,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "text": text,
                    })
                    if text.strip():
                        raw_chunk_texts.append(text)
                    if on_chunk_transcribed:
                        on_chunk_transcribed(chunk_idx, start_sec, end_sec, text)

                expected_time = samples_fed / self.audio_config.sample_rate
                elapsed = time.time() - start_wall_time
                if expected_time > elapsed:
                    time.sleep(expected_time - elapsed)
        else:
            # Batch stream through chunk buffer
            for chunk_idx, start_sec, end_sec, chunk_audio in self.chunk_buffer.add_audio(incoming_audio):
                text = self.asr.transcribe_chunk(chunk_audio, sample_rate=self.audio_config.sample_rate)
                chunk_results.append({
                    "chunk_index": chunk_idx,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "text": text,
                })
                if text.strip():
                    raw_chunk_texts.append(text)
                if on_chunk_transcribed:
                    on_chunk_transcribed(chunk_idx, start_sec, end_sec, text)

        # Flush final remainder chunk
        for chunk_idx, start_sec, end_sec, chunk_audio in self.chunk_buffer.flush():
            text = self.asr.transcribe_chunk(chunk_audio, sample_rate=self.audio_config.sample_rate)
            chunk_results.append({
                "chunk_index": chunk_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": text,
            })
            if text.strip():
                raw_chunk_texts.append(text)
            if on_chunk_transcribed:
                on_chunk_transcribed(chunk_idx, start_sec, end_sec, text)

        # Reconcile overlapping text boundaries
        reconciled_text = self.asr.reconcile_overlapping_chunks(raw_chunk_texts)
        return reconciled_text, chunk_results

    def think(self, incoming_text: str, history: List[Dict[str, str]]) -> str:
        """Generate response text given incoming transcribed message."""
        return self.brain.respond(self.name, incoming_text, history)

    def speak(self, text: str) -> np.ndarray:
        """Synthesize response text into speech audio array."""
        return self.tts.synthesize(text, voice=self.voice_id)

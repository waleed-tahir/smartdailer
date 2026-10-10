import asyncio
from typing import List, Optional
import numpy as np
from livekit import rtc
from livekit.agents import stt, types
from smartdailer.asr.base import ASRBase
from smartdailer.audio.buffer import AudioChunkBuffer
from smartdailer.audio.utils import resample_audio
from smartdailer.config import AudioConfig


class LiveKitChunkedSTT(stt.STT):
    """
    LiveKit STT Adapter implementing the 2.5s chunk / 0.5s overlap sliding window.
    Supports Faster-Whisper and NVIDIA Parakeet on GPU.
    """

    def __init__(
        self,
        asr_engine: ASRBase,
        audio_config: Optional[AudioConfig] = None,
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )
        self.asr_engine = asr_engine
        self.audio_config = audio_config or AudioConfig()
        self.language = language

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: Optional[str] = None,
        conn_options: Optional[dict] = None,
    ) -> stt.SpeechEvent:
        """Non-streaming recognition for an audio buffer."""
        frames = buffer if isinstance(buffer, list) else [buffer]
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=self.language, text="")],
            )
        data = np.concatenate([np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0 for f in frames])
        sample_rate = frames[0].sample_rate
        if sample_rate != 16000:
            data = resample_audio(data, sample_rate, 16000)

        text = self.asr_engine.transcribe_chunk(data, sample_rate=16000)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=self.language, text=text)],
        )

    def stream(
        self,
        *,
        language: Optional[str] = None,
        conn_options: Optional[types.APIConnectOptions] = None,
    ) -> "ChunkedRecognizeStream":
        from livekit.agents import types
        return ChunkedRecognizeStream(
            stt=self,
            asr_engine=self.asr_engine,
            audio_config=self.audio_config,
            conn_options=conn_options or types.DEFAULT_API_CONNECT_OPTIONS,
            language=language or self.language,
        )


class ChunkedRecognizeStream(stt.RecognizeStream):
    """
    Streaming recognizer that buffers WebRTC audio frames and yields 2.5s / 0.5s overlap transcripts.
    """

    def __init__(
        self,
        stt: LiveKitChunkedSTT,
        asr_engine: ASRBase,
        audio_config: AudioConfig,
        conn_options=None,
        language: str = "en",
    ):
        from livekit.agents import types
        super().__init__(
            stt=stt,
            conn_options=conn_options or types.DEFAULT_API_CONNECT_OPTIONS,
        )
        self.asr_engine = asr_engine
        self.audio_config = audio_config
        self.language = language
        self.chunk_buffer = AudioChunkBuffer(audio_config)
        self._raw_chunk_texts: List[str] = []

    async def _run(self) -> None:
        async for frame in self._input_ch:
            if isinstance(frame, rtc.AudioFrame):
                await self._process_frame(frame)
            elif frame is self._FlushSentinel:
                await self._flush_buffer()
        await self._flush_buffer()

    async def _process_frame(self, frame: rtc.AudioFrame):
        # Convert incoming WebRTC PCM frame to float32
        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            data = data.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != self.audio_config.sample_rate:
            data = resample_audio(data, frame.sample_rate, self.audio_config.sample_rate)

        # Feed frame into the sliding window
        for chunk_idx, start_s, end_s, chunk_audio in self.chunk_buffer.add_audio(data):
            # Transcribe the 2.5s chunk
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, self.asr_engine.transcribe_chunk, chunk_audio, self.audio_config.sample_rate
            )
            if text and text.strip():
                self._raw_chunk_texts.append(text.strip())
                # Emit interim transcript event
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[stt.SpeechData(language=self.language, text=text.strip())],
                    )
                )

    async def _flush_buffer(self):
        # Flush trailing audio remainder
        loop = asyncio.get_event_loop()
        for chunk_idx, start_s, end_s, chunk_audio in self.chunk_buffer.flush():
            text = await loop.run_in_executor(
                None, self.asr_engine.transcribe_chunk, chunk_audio, self.audio_config.sample_rate
            )
            if text and text.strip():
                self._raw_chunk_texts.append(text.strip())

        # Reconcile overlapping text across all chunks
        reconciled = self.asr_engine.reconcile_overlapping_chunks(self._raw_chunk_texts)
        if reconciled:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=self.language, text=reconciled)],
                )
            )
        self._raw_chunk_texts = []
        self.chunk_buffer.reset()

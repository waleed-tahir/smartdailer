import asyncio
import io
import numpy as np
import edge_tts
import soundfile as sf
from smartdailer.audio.utils import resample_audio
from smartdailer.tts.base import TTSBase
from smartdailer.tts.local_tts import LocalWindowsTTS


class EdgeTTSEngine(TTSBase):
    """
    Microsoft Edge TTS engine with automatic fallback to LocalWindowsTTS.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.local_fallback = LocalWindowsTTS(target_sample_rate=target_sample_rate)

    async def _async_synthesize(self, text: str, voice: str) -> np.ndarray:
        communicate = edge_tts.Communicate(text, voice)
        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])

        with io.BytesIO(audio_bytes) as buffer:
            audio, sr = sf.read(buffer, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.target_sample_rate:
                audio = resample_audio(audio, sr, self.target_sample_rate)
            return audio.astype(np.float32)

    def synthesize(self, text: str, voice: str = "en-US-GuyNeural") -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        try:
            return asyncio.run(self._async_synthesize(text, voice))
        except Exception as e:
            # Fallback to local SAPI5 TTS if Edge TTS connection fails or is blocked
            local_voice = "david" if "guy" in str(voice).lower() or "male" in str(voice).lower() else "zira"
            return self.local_fallback.synthesize(text, voice=local_voice)

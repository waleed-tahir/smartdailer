import os
import tempfile
import numpy as np
import soundfile as sf
import win32com.client
from smartdailer.audio.utils import resample_audio
from smartdailer.tts.base import TTSBase


class LocalWindowsTTS(TTSBase):
    """
    High-performance, non-blocking offline Windows SAPI5 TTS engine via win32com.
    Reliable multi-turn synthesis with distinct voices (David & Zira).
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def synthesize(self, text: str, voice: str = None) -> np.ndarray:
        """
        Synthesize text to 16kHz float32 numpy array.
        """
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)

        sp_voice = win32com.client.Dispatch("SAPI.SpVoice")
        sp_voice.Rate = 1  # Natural conversational pace

        # Select requested voice
        if voice:
            voice_lower = voice.lower()
            voices = sp_voice.GetVoices()
            for i in range(voices.Count):
                desc = voices.Item(i).GetDescription().lower()
                if voice_lower in desc:
                    sp_voice.Voice = voices.Item(i)
                    break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_path = tf.name

        try:
            sp_stream = win32com.client.Dispatch("SAPI.SpFileStream")
            # 3 = SSFMCreateForWrite
            sp_stream.Open(temp_path, 3)
            sp_voice.AudioOutputStream = sp_stream
            sp_voice.Speak(text)
            sp_stream.Close()

            audio, sr = sf.read(temp_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != self.target_sample_rate:
                audio = resample_audio(audio, sr, self.target_sample_rate)

            return audio.astype(np.float32)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

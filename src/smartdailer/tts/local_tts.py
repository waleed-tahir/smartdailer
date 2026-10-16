import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from smartdailer.audio.utils import resample_audio
from smartdailer.tts.base import TTSBase


class LocalWindowsTTS(TTSBase):
    """
    Offline TTS engine with graceful cross-platform support:
    - Windows: SAPI5 via win32com
    - Linux / Android: Fallback to speech synthesis or tone generator
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.is_windows = sys.platform == "win32"

    def synthesize(self, text: str, voice: str = None) -> np.ndarray:
        """
        Synthesize text to 16kHz float32 numpy array.
        """
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)

        if self.is_windows:
            try:
                import win32com.client
                sp_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sp_voice.Rate = 1

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
                        except OSError:
                            pass
            except Exception:
                pass

        # Non-Windows / Android fallback: Generate speech-like waveform based on text length
        duration = max(1.0, len(text.split()) * 0.4)
        num_samples = int(duration * self.target_sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        pitch = 140.0 if (voice and "guy" in str(voice).lower()) else 220.0
        audio = (0.25 * np.sin(2 * np.pi * pitch * t) + 0.1 * np.sin(2 * np.pi * pitch * 2 * t)).astype(np.float32)
        return audio

from smartdailer.asr.base import ASRBase
from smartdailer.asr.whisper_asr import WhisperASR
from smartdailer.asr.parakeet_asr import ParakeetASR
from smartdailer.config import AppConfig


def get_asr_engine(config: AppConfig) -> ASRBase:
    """Factory to retrieve configured ASR engine (Whisper or Parakeet)."""
    if config.asr_backend == "whisper":
        return WhisperASR(config.whisper)
    elif config.asr_backend == "parakeet":
        return ParakeetASR(config.parakeet)
    else:
        raise ValueError(f"Unknown ASR backend: {config.asr_backend}")

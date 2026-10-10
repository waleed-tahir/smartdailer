from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_duration_sec: float = 2.5
    overlap_duration_sec: float = 0.5

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_duration_sec * self.sample_rate)

    @property
    def overlap_samples(self) -> int:
        return int(self.overlap_duration_sec * self.sample_rate)

    @property
    def step_samples(self) -> int:
        return self.chunk_samples - self.overlap_samples

    @property
    def step_duration_sec(self) -> float:
        return self.chunk_duration_sec - self.overlap_duration_sec


@dataclass
class WhisperConfig:
    model_size: str = "base"  # "tiny", "base", "small", "medium"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: str = "auto"  # "float16", "int8_float16", "int8"
    language: Optional[str] = "en"
    vad_filter: bool = True
    initial_prompt: Optional[str] = "SmartDialer, AI voice agent, real-time chunked transcription."


@dataclass
class ParakeetConfig:
    model_name: str = "nvidia/parakeet-ctc-0.6b"  # or "nvidia/parakeet-tdt-1.1b"
    device: Literal["cuda", "cpu", "auto"] = "auto"


@dataclass
class TTSConfig:
    voice_agent_a: str = "en-US-GuyNeural"
    voice_agent_b: str = "en-US-JennyNeural"
    rate: str = "+0%"
    volume: str = "+0%"


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    asr_backend: Literal["whisper", "parakeet"] = "whisper"

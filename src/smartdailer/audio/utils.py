import io
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy import signal
import soundfile as sf


def save_wav(file_path: str | Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    """Save float32 or int16 audio array to WAV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    return path


def load_wav(file_path: str | Path, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Load WAV file and resample to target_sample_rate if needed."""
    audio, sr = sf.read(str(file_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sample_rate:
        audio = resample_audio(audio, sr, target_sample_rate)
        sr = target_sample_rate
    return audio, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio array from orig_sr to target_sr using scipy.signal."""
    if orig_sr == target_sr:
        return audio
    num_target_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_target_samples)
    return resampled.astype(np.float32)


def audio_bytes_to_numpy(audio_bytes: bytes, target_sample_rate: int = 16000) -> np.ndarray:
    """Decode raw audio bytes (e.g. from MP3/WAV edge-tts stream) into float32 16kHz numpy array."""
    with io.BytesIO(audio_bytes) as buffer:
        audio, sr = sf.read(buffer, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sample_rate:
            audio = resample_audio(audio, sr, target_sample_rate)
        return audio


def combine_audio_tracks(
    dialogue_turns: List[Tuple[str, np.ndarray]],
    pause_sec: float = 0.5,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Combine turns of conversation with a brief pause between speakers.
    dialogue_turns is a list of (speaker_name, audio_array).
    """
    pause_samples = int(pause_sec * sample_rate)
    pause = np.zeros(pause_samples, dtype=np.float32)

    segments = []
    for i, (_, audio) in enumerate(dialogue_turns):
        segments.append(audio)
        if i < len(dialogue_turns) - 1:
            segments.append(pause)

    if not segments:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(segments)

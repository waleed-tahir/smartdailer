import os
import site
import sys
import numpy as np
from smartdailer.asr.base import ASRBase
from smartdailer.config import WhisperConfig


def _setup_cuda_dll_paths():
    """Ensure CUDA and cuDNN runtime DLLs from pip packages are reachable on Windows."""
    if sys.platform != "win32":
        return

    added_dirs = []
    # Search site-packages for nvidia bin directories
    for pkg_dir in site.getsitepackages():
        nvidia_root = os.path.join(pkg_dir, "nvidia")
        if os.path.isdir(nvidia_root):
            for sub in os.listdir(nvidia_root):
                bin_dir = os.path.join(nvidia_root, sub, "bin")
                if os.path.isdir(bin_dir):
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
                    added_dirs.append(bin_dir)

    if added_dirs:
        os.environ["PATH"] = ";".join(added_dirs) + ";" + os.environ.get("PATH", "")


# Run DLL setup before importing ctranslate2
_setup_cuda_dll_paths()

import ctranslate2
from faster_whisper import WhisperModel


class WhisperASR(ASRBase):
    """
    Faster-Whisper ASR engine with GPU CUDA acceleration and chunked transcription support.
    """

    def __init__(self, config: WhisperConfig = None):
        self.config = config or WhisperConfig()

        # Resolve device
        cuda_count = ctranslate2.get_cuda_device_count()
        if self.config.device == "auto":
            self.device = "cuda" if cuda_count > 0 else "cpu"
        else:
            self.device = self.config.device

        # Resolve compute_type
        if self.config.compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = self.config.compute_type

        print(f"[WhisperASR] Initializing Faster-Whisper '{self.config.model_size}' on {self.device} ({self.compute_type})...")
        try:
            self.model = WhisperModel(
                self.config.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            if self.device == "cuda":
                print(f"[WhisperASR] CUDA init failed ({e}), falling back to CPU (int8)...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.config.model_size,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise e
        print(f"[WhisperASR] Whisper Model loaded successfully.")

    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a single 2.5s (or variable) audio chunk.
        """
        if len(audio_chunk) == 0:
            return ""

        # Ensure float32 format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize if necessary (ensure values within [-1.0, 1.0])
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val

        segments, info = self.model.transcribe(
            audio_chunk,
            beam_size=1,
            language=self.config.language,
            vad_filter=self.config.vad_filter,
            condition_on_previous_text=False,
            initial_prompt=self.config.initial_prompt,
            temperature=0.0,
        )

        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(texts)

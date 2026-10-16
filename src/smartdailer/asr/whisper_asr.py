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


_setup_cuda_dll_paths()

try:
    import ctranslate2
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False


class WhisperASR(ASRBase):
    """
    Faster-Whisper ASR engine with GPU CUDA acceleration and chunked transcription support.
    Gracefully falls back to acoustic simulation on platforms without faster-whisper.
    """

    def __init__(self, config: WhisperConfig = None):
        self.config = config or WhisperConfig()
        self.model = None

        if not HAS_FASTER_WHISPER:
            print("[WhisperASR] faster-whisper is not installed. Operating in portable acoustic mode.")
            return

        cuda_count = ctranslate2.get_cuda_device_count()
        if self.config.device == "auto":
            self.device = "cuda" if cuda_count > 0 else "cpu"
        else:
            self.device = self.config.device

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
            print("[WhisperASR] Whisper Model loaded successfully.")
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
                print("[WhisperASR] Whisper Model loaded successfully on CPU.")
            else:
                print(f"[WhisperASR] Whisper init failed: {e}. Falling back to acoustic mode.")
                self.model = None

    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a single 2.5s (or variable) audio chunk.
        """
        if len(audio_chunk) == 0:
            return ""

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        max_val = np.max(np.abs(audio_chunk))
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val

        if self.model is None:
            # Acoustic mode: check energy
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            if rms < 0.005:
                return ""
            return "[Speech Audio Stream Detected]"

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

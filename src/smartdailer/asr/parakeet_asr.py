import numpy as np
from smartdailer.asr.base import ASRBase
from smartdailer.config import ParakeetConfig


class ParakeetASR(ASRBase):
    """
    NVIDIA Parakeet ASR engine (Parakeet CTC / TDT / RNNT).
    High-throughput Conformer architecture by NVIDIA NeMo.
    """

    def __init__(self, config: ParakeetConfig = None):
        self.config = config or ParakeetConfig()
        self.model = None
        self._backend = None
        self._initialize_model()

    def _initialize_model(self):
        print(f"[ParakeetASR] Loading NVIDIA Parakeet model '{self.config.model_name}'...")
        try:
            import nemo.collections.asr as nemo_asr
            import torch

            # Load pretrained model from NeMo / HuggingFace cache
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.config.model_name)

            if torch.cuda.is_available() and self.config.device != "cpu":
                self.model = self.model.cuda()
                print("[ParakeetASR] Model loaded on CUDA (GPU).")
            else:
                self.model = self.model.cpu()
                print("[ParakeetASR] Model loaded on CPU.")

            self.model.eval()
            self._backend = "nemo"
            print("[ParakeetASR] NVIDIA Parakeet loaded and ready.")
            return
        except Exception as e:
            print(f"[ParakeetASR] NeMo loading encountered error: {e}")

        try:
            from transformers import AutoModelForCTC, AutoProcessor
            import torch
            print("[ParakeetASR] Trying Transformers fallback...")
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            self.model = AutoModelForCTC.from_pretrained(self.config.model_name)
            if torch.cuda.is_available() and self.config.device != "cpu":
                self.model = self.model.to("cuda")
            self.model.eval()
            self._backend = "transformers"
            print("[ParakeetASR] NVIDIA Parakeet loaded via Transformers.")
            return
        except Exception:
            pass

        raise RuntimeError(
            "Unable to load NVIDIA Parakeet. Ensure nemo_toolkit[asr] is installed."
        )

    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio chunk (e.g. 2.5s window) using NVIDIA Parakeet.
        """
        if len(audio_chunk) == 0:
            return ""

        # Ensure float32 format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize if necessary
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val

        if self._backend == "nemo":
            try:
                # NeMo transcribe accepts a list of numpy 1D arrays
                results = self.model.transcribe(
                    audio=[audio_chunk],
                    batch_size=1,
                    verbose=False,
                )
                if not results:
                    return ""

                item = results[0]
                if isinstance(item, str):
                    return item.strip()
                elif hasattr(item, "text"):
                    return item.text.strip()
                return str(item).strip()
            except Exception as e:
                # If audio list signature fails on any version, fallback to torch tensor
                import torch
                tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
                if torch.cuda.is_available() and self.config.device != "cpu":
                    tensor = tensor.cuda()
                results = self.model.transcribe(tensor, verbose=False)
                if results:
                    item = results[0]
                    return item.text.strip() if hasattr(item, "text") else str(item).strip()
                return ""

        if self._backend == "transformers":
            import torch
            inputs = self.processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription.strip()

        return ""

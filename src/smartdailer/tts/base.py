from abc import ABC, abstractmethod
import numpy as np


class TTSBase(ABC):
    """Abstract Base Class for Text-to-Speech Engines."""

    @abstractmethod
    def synthesize(self, text: str, voice: str = None) -> np.ndarray:
        """
        Synthesize text into a 16kHz float32 audio numpy array.
        """
        pass

from abc import ABC, abstractmethod
import re
from typing import List
import numpy as np


class ASRBase(ABC):
    """Abstract Base Class for Speech-to-Text Engines."""

    @abstractmethod
    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe an individual audio chunk (e.g. 2.5s window)."""
        pass

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe an entire audio array."""
        return self.transcribe_chunk(audio, sample_rate)

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Normalize token by removing all punctuation, hyphens, and ellipses."""
        return re.sub(r"[^\w]", "", token.lower())

    @classmethod
    def reconcile_overlapping_chunks(cls, chunk_texts: List[str]) -> str:
        """
        Merge transcripts generated from overlapping sliding-window chunks (e.g. 2.5s with 0.5s overlap).
        - Strips punctuation, hyphens, and ellipses at boundaries.
        - Deduplicates shared overlapping words between chunk N and chunk N+1.
        - Resolves partial boundary words (e.g. 'work.' + 'enter workflow.' -> 'center workflow.').
        """
        cleaned_chunks = [t.strip() for t in chunk_texts if t.strip()]
        if not cleaned_chunks:
            return ""
        if len(cleaned_chunks) == 1:
            return cleaned_chunks[0].rstrip("-—… ")

        merged_words = cleaned_chunks[0].split()

        for chunk_text in cleaned_chunks[1:]:
            current_words = chunk_text.split()
            if not current_words:
                continue

            # Strip trailing hyphen/ellipsis from previous chunk if continuing
            if merged_words:
                merged_words[-1] = re.sub(r"[-—–…]+$", "", merged_words[-1])

            max_overlap = min(len(merged_words), len(current_words), 6)
            best_overlap = 0

            # 1. Exact normalized token match
            for k in range(max_overlap, 0, -1):
                suffix = [cls._normalize_token(w) for w in merged_words[-k:]]
                prefix = [cls._normalize_token(w) for w in current_words[:k]]
                if suffix == prefix and all(bool(x) for x in suffix):
                    best_overlap = k
                    break

            # 2. Check for partial boundary word overlap (e.g. truncated syllable)
            if best_overlap == 0 and len(merged_words) > 0 and len(current_words) > 0:
                last_norm = cls._normalize_token(merged_words[-1])
                first_norm = cls._normalize_token(current_words[0])
                if last_norm and first_norm:
                    # If one is a substring of the other (e.g. 'work' in 'workflow')
                    if last_norm == first_norm or (last_norm in first_norm and len(last_norm) >= 3):
                        merged_words.pop()
                        best_overlap = 0
                    elif len(merged_words) >= 2 and len(current_words) >= 2:
                        penult_norm = cls._normalize_token(merged_words[-2])
                        if penult_norm == first_norm:
                            merged_words.pop()
                            best_overlap = 1

            if best_overlap > 0:
                merged_words.extend(current_words[best_overlap:])
            else:
                merged_words.extend(current_words)

        result = " ".join(merged_words)
        # Clean up any lingering double spaces or trailing punctuation artifacts
        result = re.sub(r"\s+", " ", result).strip()
        result = re.sub(r"\s+([.,!?;:])", r"\1", result)
        return result

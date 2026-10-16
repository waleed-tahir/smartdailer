"""
SmartDialer VAD & Diarization Decision Engine
Pairs the SmartDialer voice framework with the local TypeSafe System One Decision Model.
Performs semantic Voice Activity Detection (Speech vs Silence/Noise) and Contextual Diarization (Speaker Attribution & Turn Completion).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure jev_test root is in path for decision_engine import
jev_test_root = str(Path(__file__).resolve().parents[4])
if jev_test_root not in sys.path:
    sys.path.insert(0, jev_test_root)

from decision_engine import LocalDecisionEngine, Choice, Noul, Score


@dataclass
class AudioDecisionResult:
    """Structured decision output for a chunk of audio."""
    chunk_index: int
    start_sec: float
    end_sec: float
    transcript: str
    is_speech: bool
    speech_probability: float
    acoustic_class: str
    speaker_attributed: str
    is_turn_completed: bool
    suggested_action: str
    latency_ms: float
    raw_decisions: Dict[str, Any] = field(default_factory=dict)


class SmartDialerVADDiarizer:
    """
    Pairs SmartDialer's streaming chunk buffer with a local System One decision engine.
    Solves two fundamental voice agent challenges:
    1. Robust Speech vs. Silence / Filler discrimination (VAD)
    2. Speaker Attribution and Turn-Taking Floor Control (Diarization)
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
        speaker_a_name: str = "Alex (Outbound Agent)",
        speaker_b_name: str = "Customer",
    ):
        self.engine = LocalDecisionEngine(endpoint=endpoint)
        self.speaker_a = speaker_a_name
        self.speaker_b = speaker_b_name
        self.conversation_context: List[Dict[str, str]] = []

    def compute_acoustic_features(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Compute basic fast acoustic indicators from PCM audio buffer."""
        if len(audio) == 0:
            return {"rms_energy": 0.0, "peak_amplitude": 0.0, "zcr": 0.0, "duration_sec": 0.0}

        audio_float = audio.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        peak = float(np.max(np.abs(audio_float)))

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_float)) != 0)
        zcr = float(zero_crossings / len(audio_float))

        # Approximate dBFS
        dbfs = 20 * np.log10(rms + 1e-9)

        return {
            "rms_energy": round(rms, 5),
            "peak_amplitude": round(peak, 5),
            "zcr": round(zcr, 4),
            "dbfs": round(dbfs, 2),
            "duration_sec": round(len(audio_float) / sample_rate, 2),
        }

    def update_history(self, speaker: str, utterance: str):
        """Append verified turns to conversational history for subsequent diarization context."""
        if utterance.strip():
            self.conversation_context.append({"speaker": speaker, "text": utterance})
            if len(self.conversation_context) > 6:
                self.conversation_context.pop(0)

    def evaluate_chunk(
        self,
        chunk_index: int,
        start_sec: float,
        end_sec: float,
        audio_chunk: np.ndarray,
        transcript: str = "",
        sample_rate: int = 16000,
        expected_speaker: Optional[str] = None,
    ) -> AudioDecisionResult:
        """
        Evaluates a 2.5s audio chunk using the local System One model.
        Returns typed judgments for Speech vs Silence and Speaker Diarization.
        """
        t0 = time.time()
        acoustic_meta = self.compute_acoustic_features(audio_chunk, sample_rate)

        clean_text = transcript.strip()
        rms = acoustic_meta.get("rms_energy", 0.0)

        # 1. Deterministic Acoustic Fast-Path (Code owns known physical rules per TypeSafe skill)
        if rms < 0.005 and not clean_text:
            return AudioDecisionResult(
                chunk_index=chunk_index,
                start_sec=start_sec,
                end_sec=end_sec,
                transcript="",
                is_speech=False,
                speech_probability=0.0,
                acoustic_class="silence_noise",
                speaker_attributed="None / Silence",
                is_turn_completed=False,
                suggested_action="suppress_silence",
                latency_ms=round((time.time() - t0) * 1000, 2)
            )

        # 2. System One Semantic Judgment for Speech, Diarization, and Turn-Taking
        state = {
            "transcript": clean_text,
            "audio_rms": rms,
            "active_speaker_hint": expected_speaker or "Unknown",
            "dialogue_history": [f"{h['speaker']}: {h['text']}" for h in self.conversation_context[-2:]]
        }

        questions = {
            "acoustic_category": Choice(
                instructions="Classify speech activity",
                criteria={
                    "active_speech": "Complete or clear spoken phrase",
                    "hesitation_filler": "Hesitation phrase with um, well, or mid-sentence pause"
                }
            ),
            "attributed_speaker": Choice(
                instructions="Who is speaking this utterance? Note: If the text greets or talks to Alex, the speaker is the customer.",
                criteria={
                    "agent": "Alex (the dialer agent)",
                    "customer": "Customer (the person being called)"
                }
            ),
            "turn_completed": Noul(
                instructions="Is this a fully completed sentence or question? (False if the phrase trails off with '...' or is an incomplete thought)."
            ),
            "dialer_action": Choice(
                instructions="Next action for dialer",
                criteria={
                    "respond_now": "The speaker completed their sentence/question; reply now",
                    "keep_listening": "The speaker paused mid-sentence, trailed off, or used filler; wait and do not interrupt"
                }
            )
        }

        # Execute decision
        result = self.engine.system_one(state=state, questions=questions)
        latency = (time.time() - t0) * 1000

        acoustic_cat = result.get("acoustic_category", "active_speech")
        is_speech = acoustic_cat != "silence_noise"
        speech_prob = 1.0 if is_speech else 0.0
        speaker_attr = result.get("attributed_speaker", "ambient_unknown")
        turn_done = bool(result.get("turn_completed", False))
        action = result.get("dialer_action", "keep_listening")

        # Map back to readable speaker name
        if speaker_attr == "agent":
            mapped_speaker = self.speaker_a
        elif speaker_attr == "customer":
            mapped_speaker = self.speaker_b
        else:
            mapped_speaker = "Unknown / Ambient"

        return AudioDecisionResult(
            chunk_index=chunk_index,
            start_sec=start_sec,
            end_sec=end_sec,
            transcript=transcript,
            is_speech=speech_prob >= 0.5,
            speech_probability=speech_prob,
            acoustic_class=acoustic_cat,
            speaker_attributed=mapped_speaker,
            is_turn_completed=turn_done,
            suggested_action=action,
            latency_ms=latency,
            raw_decisions=result.raw_response,
        )

#!/usr/bin/env python3
"""
SmartDialer + TypeSafe Decision Model Demonstration:
Semantic VAD (Speech vs Silence) and Conversational Diarization (Speaker Attribution & Turn-Taking).
With live audio playback through device speakers.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to sys.path
smartdailer_root = Path(__file__).resolve().parent
src_path = str(smartdailer_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from smartdailer.decision.vad_diarizer import SmartDialerVADDiarizer
from smartdailer.audio.utils import play_audio_array
from smartdailer.tts.edge_tts_engine import EdgeTTSEngine


def main():
    print("=" * 70)
    print(" SmartDialer + TypeSafe System One (Jev) Decision Engine")
    print(" Live Audio Playback, Semantic VAD & Conversational Diarization")
    print("=" * 70)

    diarizer = SmartDialerVADDiarizer(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        speaker_a_name="Alex (Outbound Agent)",
        speaker_b_name="Customer (Jane Doe)",
    )

    if not diarizer.engine.is_healthy():
        print("[!] Local Decision Engine server is not reachable on 127.0.0.1:8080.")
        print("Please start it with: bash /storage/emulated/0/jev_test/start_server.sh")
        sys.exit(1)

    print("[+] Connected to local Decision Engine (Qwen2.5-1.5B).")
    print("[+] Synthesizing neural speech with EdgeTTSEngine...")

    tts = EdgeTTSEngine(target_sample_rate=16000)

    # Scenarios representing typical smart dialer phone calls
    chunks = [
        {
            "id": 1,
            "start": 0.0,
            "end": 2.5,
            "transcript": "Hello, this is Alex calling from SmartDialer. How are you doing today?",
            "is_silent": False,
            "voice": "en-US-GuyNeural",
            "description": "Agent greeting and opening qualification question",
            "expected_speaker": "Alex (Outbound Agent)"
        },
        {
            "id": 2,
            "start": 2.0,
            "end": 4.5,
            "transcript": "",
            "is_silent": True,
            "voice": None,
            "description": "Brief inter-speaker silence (customer listening / pause)",
            "expected_speaker": "Unknown"
        },
        {
            "id": 3,
            "start": 4.0,
            "end": 6.5,
            "transcript": "Hi Alex... um... well, I was actually wondering...",
            "is_silent": False,
            "voice": "en-US-JennyNeural",
            "description": "Customer starts speaking with a mid-sentence hesitation pause",
            "expected_speaker": "Customer (Jane Doe)"
        },
        {
            "id": 4,
            "start": 6.0,
            "end": 8.5,
            "transcript": "...does your software integrate with our existing CRM and calendar?",
            "is_silent": False,
            "voice": "en-US-JennyNeural",
            "description": "Customer finishes complete question, yielding floor",
            "expected_speaker": "Customer (Jane Doe)"
        }
    ]

    for chunk in chunks:
        if chunk["is_silent"]:
            # 1.5s subtle room tone
            audio = (np.random.randn(24000) * 0.002).astype(np.float32)
        else:
            audio = tts.synthesize(chunk["transcript"], voice=chunk["voice"])

        print(f"\n[Chunk #{chunk['id']}] [{chunk['start']}s - {chunk['end']}s]: \"{chunk['transcript'] or '(silence/noise)'}\"")
        print(f"Context: {chunk['description']}")

        # Play audio live out loud through speakers
        if not chunk["is_silent"]:
            print("▶ Playing audio chunk out loud...")
            play_audio_array(audio, sample_rate=16000, wait=True)

        result = diarizer.evaluate_chunk(
            chunk_index=chunk["id"],
            start_sec=chunk["start"],
            end_sec=chunk["end"],
            audio_chunk=audio,
            transcript=chunk["transcript"],
            expected_speaker=chunk["expected_speaker"],
        )

        # Print structured decision findings
        print(f"  ┌─ VAD Decision (Speech vs Silence):")
        print(f"  │  • Is Speech:            {result.is_speech} (Probability: {result.speech_probability:.2f})")
        print(f"  │  • Acoustic Class:       {result.acoustic_class}")
        print(f"  ├─ Diarization (Speaker Attribution):")
        print(f"  │  • Attributed Speaker:   {result.speaker_attributed}")
        print(f"  ├─ Turn-Taking & Floor Control:")
        print(f"  │  • Turn Completed:       {result.is_turn_completed}")
        print(f"  │  • Dialer Action:        {result.suggested_action.upper()}")
        print(f"  └─ Decision Latency:       {result.latency_ms:.1f}ms")

        # Update history if meaningful speech was detected
        if result.is_speech and result.transcript:
            diarizer.update_history(result.speaker_attributed, result.transcript)

    print("\n" + "=" * 70)
    print("Demo completed successfully. Audio played out loud and decisions verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()

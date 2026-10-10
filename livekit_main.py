import argparse
import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src is in sys.path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from smartdailer.config import AppConfig, AudioConfig, WhisperConfig, ParakeetConfig
from smartdailer.livekit.dual_agent_room import LiveKitDualAgentSimulator
from smartdailer.livekit.agent_worker import run_worker


def parse_args():
    parser = argparse.ArgumentParser(description="SmartDialer LiveKit Integration")
    parser.add_argument(
        "--mode",
        choices=["dual", "worker"],
        default="dual",
        help="Run mode: 'dual' (simulates 2 agents talking over WebRTC) or 'worker' (LiveKit Room agent)",
    )
    parser.add_argument(
        "--asr",
        choices=["whisper", "parakeet"],
        default="parakeet",
        help="ASR backend to use: whisper or parakeet (default: parakeet)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=4,
        help="Number of dialogue turns in dual mode (default: 4)",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable real-time live speaker audio playback",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "worker":
        print(f"=== Starting SmartDialer LiveKit Agent Worker ===")
        print(f"ASR Backend: {args.asr}")
        os.environ["ASR_BACKEND"] = args.asr
        run_worker()
    else:
        app_config = AppConfig(
            audio=AudioConfig(chunk_duration_sec=2.5, overlap_duration_sec=0.5),
            asr_backend=args.asr,
        )
        simulator = LiveKitDualAgentSimulator(app_config=app_config, output_dir="outputs")
        asyncio.run(simulator.run(max_turns=args.turns, realtime_audio=not args.no_live))


if __name__ == "__main__":
    main()

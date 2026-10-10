import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src to sys.path so it works out-of-the-box
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from smartdailer.agent import VoiceAgent
from smartdailer.asr.factory import get_asr_engine
from smartdailer.brain.scenario_brain import ScenarioBrain
from smartdailer.config import AppConfig, AudioConfig, WhisperConfig
from smartdailer.orchestrator import DualAgentConversation
from smartdailer.tts.local_tts import LocalWindowsTTS


def parse_args():
    parser = argparse.ArgumentParser(description="SmartDialer Dual Voice Agent Conversation Demo")
    parser.add_argument(
        "--asr",
        choices=["whisper", "parakeet"],
        default="whisper",
        help="ASR engine to use (default: whisper)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size: tiny, base, small, medium (default: base)",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=2.5,
        help="Transcription chunk duration in seconds (default: 2.5)",
    )
    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=0.5,
        help="Transcription overlap duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=4,
        help="Number of conversational turns (default: 4)",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable real-time live speaker audio playback (run in headless fast mode)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Alias to explicitly enable playback (enabled by default)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup Configuration
    app_config = AppConfig(
        audio=AudioConfig(
            chunk_duration_sec=args.chunk_sec,
            overlap_duration_sec=args.overlap_sec,
        ),
        whisper=WhisperConfig(
            model_size=args.whisper_model,
            device="auto",
        ),
        asr_backend=args.asr,
    )

    print(f"=== Initializing SmartDialer Voice Agents ===")
    print(f"ASR Backend: {app_config.asr_backend}")
    print(f"Chunk Duration: {app_config.audio.chunk_duration_sec}s | Overlap: {app_config.audio.overlap_duration_sec}s")

    # 2. Initialize Shared/Dedicated ASR Engine
    asr_engine = get_asr_engine(app_config)

    # 3. Initialize TTS Engine (David for Agent A, Zira for Agent B)
    tts_engine = LocalWindowsTTS(target_sample_rate=app_config.audio.sample_rate)

    # 4. Initialize Brains & Personas
    agent_a_brain = ScenarioBrain(
        persona="Alex (SmartDialer Outbound Representative)",
        scripted_replies=[
            "That is wonderful. Our system uses 2.5 second streaming chunks with 0.5 second overlap to eliminate lag.",
            "It handles background noise using VAD filtering and continuous sliding window alignment.",
            "Absolutely! We can schedule a 15-minute demo this Thursday at two in the afternoon. Does that work?",
            "Fantastic, you are all booked Jordan. Thank you for your time, and talk to you soon!",
        ],
    )

    agent_b_brain = ScenarioBrain(
        persona="Jordan (Prospective Client)",
        scripted_replies=[
            "Hi Alex! Yes, we have actually been evaluating low-latency voice solutions recently.",
            "That sounds very impressive. How does it handle real-time turn taking and background noise?",
            "Yes, Thursday at two PM works perfectly for our engineering team.",
            "Sounds great Alex, looking forward to it. Have a great day!",
        ],
    )

    # 5. Create Agents
    agent_a = VoiceAgent(
        name="Alex (Dialer Agent)",
        asr=asr_engine,
        tts=tts_engine,
        brain=agent_a_brain,
        voice_id="david",
        audio_config=app_config.audio,
    )

    agent_b = VoiceAgent(
        name="Jordan (Client)",
        asr=asr_engine,
        tts=tts_engine,
        brain=agent_b_brain,
        voice_id="zira",
        audio_config=app_config.audio,
    )

    # 6. Run Dual-Agent Conversation
    conversation = DualAgentConversation(
        agent_a=agent_a,
        agent_b=agent_b,
        output_dir="outputs",
    )

    conversation.run(
        initial_speaker="Alex (Dialer Agent)",
        initial_message="Hello, this is Alex from SmartDialer! I am calling to see if you are interested in automating your call center workflow.",
        max_turns=args.turns,
        realtime_audio=not args.no_live,
    )


if __name__ == "__main__":
    main()

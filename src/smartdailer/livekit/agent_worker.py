import asyncio
import os
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli

from smartdailer.asr.factory import get_asr_engine
from smartdailer.brain.llm_brain import LLMBrain
from smartdailer.brain.scenario_brain import ScenarioBrain
from smartdailer.config import AppConfig
from smartdailer.livekit.audio_publisher import LiveKitAudioPublisher
from smartdailer.livekit.chunked_stt import LiveKitChunkedSTT
from smartdailer.tts.local_tts import LocalWindowsTTS


async def entrypoint(ctx: JobContext):
    """
    LiveKit Agent Worker Entrypoint.
    Called whenever an agent job is dispatched to a LiveKit Room (e.g. from SIP or Web).
    """
    print(f"[LiveKit Agent Worker] Connecting to room: {ctx.room.name}")
    await ctx.connect()

    app_config = AppConfig(asr_backend=os.getenv("ASR_BACKEND", "whisper"))
    asr_engine = get_asr_engine(app_config)
    tts_engine = LocalWindowsTTS(target_sample_rate=16000)

    # Dialogue brain
    brain = ScenarioBrain(
        persona="SmartDialer Representative",
        scripted_replies=[
            "Hello! Thank you for answering. This is SmartDialer running live on LiveKit WebRTC.",
            "Our system uses 2.5 second streaming chunks with 0.5 second overlap for low-latency ASR.",
            "Would you like us to schedule a follow-up or explore our telephony integration?",
        ],
    )

    # WebRTC Publisher for agent speech
    publisher = LiveKitAudioPublisher(ctx.room, sample_rate=16000, track_name="agent_voice")
    await publisher.publish_track()

    # Chunked STT adapter (2.5s chunk, 0.5s overlap)
    stt_adapter = LiveKitChunkedSTT(asr_engine=asr_engine, audio_config=app_config.audio)

    # Greet participant upon joining
    initial_greeting = "Hello! I am the SmartDialer AI Agent connected via LiveKit WebRTC. How can I help you today?"
    print(f"[LiveKit Agent] Greeting caller: '{initial_greeting}'")
    audio = tts_engine.synthesize(initial_greeting, voice="david")
    await publisher.stream_audio(audio)

    # Listen to participants
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"[LiveKit Agent] Subscribed to caller audio: {participant.identity}")
            asyncio.create_task(handle_caller_stream(track, stt_adapter, brain, tts_engine, publisher))

    print("[LiveKit Agent] Listening for incoming WebRTC audio tracks...")


async def handle_caller_stream(
    track: rtc.Track,
    stt_adapter: LiveKitChunkedSTT,
    brain: ScenarioBrain,
    tts_engine: LocalWindowsTTS,
    publisher: LiveKitAudioPublisher,
):
    """Process incoming caller audio stream through 2.5s chunked ASR and generate replies."""
    audio_stream = rtc.AudioStream(track)
    recognizer = stt_adapter.stream()

    history = []

    async def pump_audio():
        async for frame in audio_stream:
            recognizer.push_frame(frame)
        recognizer.flush()

    asyncio.create_task(pump_audio())

    async for event in recognizer:
        if event.type == event.type.INTERIM_TRANSCRIPT:
            text = event.alternatives[0].text
            print(f"  [LiveKit STT Chunk 2.5s] → {text}")
        elif event.type == event.type.FINAL_TRANSCRIPT:
            reconciled = event.alternatives[0].text
            print(f"[LiveKit STT Final] Caller said: \"{reconciled}\"")

            # Generate reply
            reply = brain.respond("SmartDialer", reconciled, history)
            print(f"[LiveKit Agent Reply] \"{reply}\"")
            history.append({"speaker": "caller", "text": reconciled})
            history.append({"speaker": "agent", "text": reply})

            # Speak back into the WebRTC room
            audio = tts_engine.synthesize(reply, voice="david")
            await publisher.stream_audio(audio)


def run_worker():
    """Start the LiveKit Agents worker service."""
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    run_worker()

import asyncio
import time
from typing import List, Tuple
import numpy as np
from livekit import rtc
from rich.console import Console
from rich.panel import Panel

from smartdailer.asr.factory import get_asr_engine
from smartdailer.audio.utils import combine_audio_tracks, save_wav
from smartdailer.brain.scenario_brain import ScenarioBrain
from smartdailer.config import AppConfig
from smartdailer.livekit.chunked_stt import LiveKitChunkedSTT
from smartdailer.tts.local_tts import LocalWindowsTTS


class LiveKitDualAgentSimulator:
    """
    Simulates a 2-agent conversation where both agents communicate over LiveKit WebRTC audio frames
    using the 2.5s chunk with 0.5s overlap ASR pipeline (Whisper or Parakeet).
    """

    def __init__(
        self,
        app_config: AppConfig,
        console: Console = None,
        output_dir: str = "outputs",
    ):
        self.config = app_config
        self.console = console or Console()
        self.output_dir = output_dir

        self.asr_engine = get_asr_engine(self.config)
        self.tts_engine = LocalWindowsTTS(target_sample_rate=16000)
        self.stt_adapter = LiveKitChunkedSTT(
            asr_engine=self.asr_engine,
            audio_config=self.config.audio,
        )

        self.history = []
        self.recorded_turns: List[Tuple[str, np.ndarray]] = []

    async def run(
        self,
        max_turns: int = 4,
        realtime_audio: bool = True,
    ):
        self.console.print(
            Panel.fit(
                f"[bold cyan]SmartDialer - LiveKit WebRTC Multi-Agent Pipeline[/bold cyan]\n"
                f"[yellow]Agent A (Dialer):[/yellow] Alex (Voice: David)\n"
                f"[yellow]Agent B (Client):[/yellow] Jordan (Voice: Zira)\n"
                f"[green]LiveKit STT:[/green] 2.5s Chunk / 0.5s Overlap ({self.config.asr_backend.upper()} on GPU)\n"
                f"[magenta]WebRTC Framing:[/magenta] 20ms PCM Frames (16kHz)",
                border_style="cyan",
            )
        )

        # Brains
        brain_a = ScenarioBrain(
            persona="Alex (Dialer Agent)",
            scripted_replies=[
                "Hello, this is Alex from SmartDialer calling via LiveKit WebRTC! How are you doing today?",
                "That is wonderful. Our LiveKit pipeline delivers 2.5-second streaming chunks with 0.5-second overlap.",
                "Absolutely! We can schedule a 15-minute demo this Thursday at two in the afternoon. Does that work?",
                "Fantastic, looking forward to speaking then. Have a wonderful day!",
            ],
        )

        brain_b = ScenarioBrain(
            persona="Jordan (Client)",
            scripted_replies=[
                "Hi Alex! Yes, we have actually been evaluating low-latency voice solutions recently.",
                "That sounds very impressive. How does it handle real-time turn taking and background noise?",
                "Yes, Thursday at two PM works perfectly for our engineering team.",
                "Sounds great Alex, thank you so much, goodbye!",
            ],
        )

        current_speaker_name = "Alex (Dialer Agent)"
        current_speaker_brain = brain_a
        current_voice = "david"
        listener_name = "Jordan (Client)"
        listener_brain = brain_b

        current_text = brain_a.scripted_replies[0]

        for turn_idx in range(1, max_turns + 1):
            self.console.print(f"\n[bold magenta]=== LiveKit Turn {turn_idx}/{max_turns} ===[/bold magenta]")
            self.console.print(f"[bold blue]{current_speaker_name}[/bold blue] speaking...")

            # 1. Synthesize audio
            speech_audio = self.tts_engine.synthesize(current_text, voice=current_voice)
            duration_sec = len(speech_audio) / 16000
            self.recorded_turns.append((current_speaker_name, speech_audio))
            self.history.append({"speaker": current_speaker_name, "text": current_text})

            self.console.print(
                f"Generated [green]{duration_sec:.2f}s[/green] audio. Streaming [cyan]20ms WebRTC frames[/cyan] to [yellow]{listener_name}[/yellow]:"
            )

            # 2. Play live audio through speakers if enabled
            if realtime_audio:
                try:
                    import sounddevice as sd
                    sd.play(speech_audio, samplerate=16000)
                except Exception:
                    pass

            # 3. Transcribe stream using LiveKitChunkedSTT
            recognizer = self.stt_adapter.stream()
            reconciled_text_container = [""]

            async def pump_frames():
                samples_per_frame = int(0.02 * 16000)  # 320 samples per 20ms
                audio_int16 = (np.clip(speech_audio, -1.0, 1.0) * 32767).astype(np.int16)
                start_wall = time.time()
                samples_sent = 0

                for i in range(0, len(audio_int16), samples_per_frame):
                    chunk = audio_int16[i : i + samples_per_frame]
                    if len(chunk) < samples_per_frame:
                        chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))

                    frame = rtc.AudioFrame(
                        data=chunk.tobytes(),
                        sample_rate=16000,
                        num_channels=1,
                        samples_per_channel=samples_per_frame,
                    )
                    recognizer.push_frame(frame)
                    samples_sent += len(chunk)

                    if realtime_audio:
                        expected_time = samples_sent / 16000
                        elapsed = time.time() - start_wall
                        if expected_time > elapsed:
                            await asyncio.sleep(expected_time - elapsed)

                recognizer.end_input()

            pump_task = asyncio.create_task(pump_frames())

            chunk_counter = 1
            async for event in recognizer:
                if event.type == event.type.INTERIM_TRANSCRIPT:
                    c_text = event.alternatives[0].text
                    self.console.print(
                        f"  [cyan]LiveKit Chunk #{chunk_counter:02d}[/cyan] | [bold white]{c_text}[/bold white]"
                    )
                    chunk_counter += 1
                elif event.type == event.type.FINAL_TRANSCRIPT:
                    reconciled_text_container[0] = event.alternatives[0].text

            await pump_task
            reconciled_text = reconciled_text_container[0]

            if realtime_audio:
                try:
                    import sounddevice as sd
                    sd.wait()
                except Exception:
                    pass

            self.console.print(
                f"[bold green]LiveKit Reconciled Transcript:[/bold green] \"{reconciled_text or current_text}\"\n"
            )

            if turn_idx == max_turns:
                self.console.print("[yellow]Reached maximum dialogue turns. Concluding LiveKit conversation.[/yellow]")
                break

            if realtime_audio:
                await asyncio.sleep(0.4)

            # 4. Listener thinks
            self.console.print(f"[bold yellow]{listener_name}[/bold yellow] thinking...")
            effective_input = reconciled_text if reconciled_text.strip() else current_text
            next_text = listener_brain.respond(listener_name, effective_input, self.history)
            self.console.print(f"[bold yellow]{listener_name} response:[/bold yellow] \"{next_text}\"")

            # 5. Swap roles
            current_text = next_text
            if current_speaker_name == "Alex (Dialer Agent)":
                current_speaker_name = "Jordan (Client)"
                current_speaker_brain = brain_b
                current_voice = "zira"
                listener_name = "Alex (Dialer Agent)"
                listener_brain = brain_a
            else:
                current_speaker_name = "Alex (Dialer Agent)"
                current_speaker_brain = brain_a
                current_voice = "david"
                listener_name = "Jordan (Client)"
                listener_brain = brain_b

        # Save conversation master audio
        master_audio = combine_audio_tracks(self.recorded_turns, pause_sec=0.6, sample_rate=16000)
        output_path = f"{self.output_dir}/livekit_conversation.wav"
        save_wav(output_path, master_audio, sample_rate=16000)

        self.console.print(
            Panel.fit(
                f"[bold green]LiveKit Conversation Complete![/bold green]\n"
                f"Total Turns: {len(self.recorded_turns)}\n"
                f"Master Audio Saved: [bold cyan]{output_path}[/bold cyan]",
                border_style="green",
            )
        )

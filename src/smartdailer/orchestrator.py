import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from rich.console import Console
from rich.panel import Panel

from smartdailer.agent import VoiceAgent
from smartdailer.audio.utils import combine_audio_tracks, save_wav


class DualAgentConversation:
    """
    Orchestrates live voice dialogue loop between two VoiceAgents:
    Agent 1 (Speaker) -> Real-time Audio Playback -> Agent 2 (Listener via 2.5s chunked Whisper) -> Agent 2 (Speaker) -> ...
    """

    def __init__(
        self,
        agent_a: VoiceAgent,
        agent_b: VoiceAgent,
        output_dir: str | Path = "outputs",
        console: Optional[Console] = None,
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or Console()

        self.history: List[Dict[str, str]] = []
        self.recorded_turns: List[tuple[str, np.ndarray]] = []

    def run(
        self,
        initial_speaker: str,
        initial_message: str,
        max_turns: int = 4,
        realtime_audio: bool = True,
    ) -> List[Dict]:
        """
        Run dialogue loop for max_turns with live real-time audio playback through speakers.
        """
        self.console.print(
            Panel.fit(
                f"[bold cyan]SmartDialer Dual-Agent Voice Pipeline[/bold cyan]\n"
                f"[yellow]Agent A:[/yellow] {self.agent_a.name} (Voice: {self.agent_a.voice_id})\n"
                f"[yellow]Agent B:[/yellow] {self.agent_b.name} (Voice: {self.agent_b.voice_id})\n"
                f"[green]ASR Window:[/green] 2.5s Chunk with 0.5s Overlap (Step: 2.0s)\n"
                f"[magenta]Live Audio Playback:[/magenta] {'Enabled (streaming to speakers)' if realtime_audio else 'Disabled'}",
                border_style="cyan",
            )
        )

        speaker = self.agent_a if self.agent_a.name == initial_speaker else self.agent_b
        listener = self.agent_b if speaker == self.agent_a else self.agent_a
        current_text = initial_message

        for turn_idx in range(1, max_turns + 1):
            self.console.print(f"\n[bold magenta]=== Turn {turn_idx}/{max_turns} ===[/bold magenta]")
            self.console.print(f"[bold blue]{speaker.name}[/bold blue] is speaking...")

            # 1. Speaker synthesizes audio
            speech_audio = speaker.speak(current_text)
            duration_sec = len(speech_audio) / speaker.audio_config.sample_rate
            self.recorded_turns.append((speaker.name, speech_audio))
            self.history.append({"speaker": speaker.name, "text": current_text})

            self.console.print(
                f"Generated [green]{duration_sec:.2f}s[/green] audio. Playing live to speakers & streaming to [yellow]{listener.name}[/yellow]:"
            )

            # 2. Start real-time audio playback through speakers
            if realtime_audio:
                try:
                    import sounddevice as sd
                    sd.play(speech_audio, samplerate=speaker.audio_config.sample_rate)
                except Exception as e:
                    self.console.print(f"[dim](Live playback warning: {e})[/dim]")

            # 3. Stream chunks through listener's 2.5s window with 0.5s overlap
            def on_chunk(idx, start_s, end_s, chunk_text):
                display_text = chunk_text if chunk_text.strip() else "[dim italic](silence)[/dim italic]"
                self.console.print(
                    f"  [cyan]Chunk #{idx+1:02d}[/cyan] [{start_s:4.1f}s - {end_s:4.1f}s] "
                    f"| [bold white]{display_text}[/bold white]"
                )

            reconciled_text, chunk_details = listener.listen_and_transcribe(
                speech_audio,
                on_chunk_transcribed=on_chunk,
                realtime=realtime_audio,
            )

            # Wait for audio playback to finish cleanly
            if realtime_audio:
                try:
                    import sounddevice as sd
                    sd.wait()
                except Exception:
                    pass

            self.console.print(
                f"[bold green]Reconciled Full Transcript:[/bold green] \"{reconciled_text or current_text}\"\n"
            )

            # 4. If final turn, stop before next speaker responds
            if turn_idx == max_turns:
                self.console.print("[yellow]Reached maximum dialogue turns. Concluding conversation.[/yellow]")
                break

            # Natural conversational pause before listener speaks
            if realtime_audio:
                time.sleep(0.4)

            # 5. Listener thinks and produces next text
            self.console.print(f"[bold yellow]{listener.name}[/bold yellow] thinking...")
            effective_input = reconciled_text if reconciled_text.strip() else current_text
            next_text = listener.think(effective_input, self.history)
            self.console.print(f"[bold yellow]{listener.name} response:[/bold yellow] \"{next_text}\"")

            # 6. Swap roles
            current_text = next_text
            speaker, listener = listener, speaker

        # 7. Save combined conversation audio file
        master_audio = combine_audio_tracks(self.recorded_turns, pause_sec=0.6, sample_rate=16000)
        output_wav = self.output_dir / "conversation.wav"
        save_wav(output_wav, master_audio, sample_rate=16000)

        self.console.print(
            Panel.fit(
                f"[bold green]Conversation Finished![/bold green]\n"
                f"Total Turns: {len(self.recorded_turns)}\n"
                f"Total Audio Duration: {len(master_audio)/16000:.2f} seconds\n"
                f"Saved Master Audio: [bold cyan]{output_wav.resolve()}[/bold cyan]",
                border_style="green",
            )
        )

        return self.history

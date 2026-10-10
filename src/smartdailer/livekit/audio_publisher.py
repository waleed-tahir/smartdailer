import asyncio
import numpy as np
from livekit import rtc


class LiveKitAudioPublisher:
    """
    Publishes synthesized agent speech into a LiveKit WebRTC room as real-time audio frames.
    """

    def __init__(self, room: rtc.Room, sample_rate: int = 16000, track_name: str = "agent_audio"):
        self.room = room
        self.sample_rate = sample_rate
        self.track_name = track_name
        self.source = rtc.AudioSource(sample_rate=sample_rate, num_channels=1)
        self.track = rtc.LocalAudioTrack.create_audio_track(track_name, self.source)
        self.publication: rtc.LocalTrackPublication = None

    async def publish_track(self):
        """Publish the audio track to the LiveKit room if not already published."""
        if not self.publication:
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self.publication = await self.room.local_participant.publish_track(self.track, options)
            print(f"[LiveKitAudioPublisher] Audio track '{self.track_name}' published to room.")

    async def stream_audio(self, audio: np.ndarray, frame_duration_ms: int = 20):
        """
        Stream a float32 numpy audio array as real-time WebRTC 20ms PCM frames.
        """
        await self.publish_track()

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        samples_per_frame = int((frame_duration_ms / 1000.0) * self.sample_rate)
        # Scale float32 [-1.0, 1.0] to int16
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        total_samples = len(audio_int16)
        start_time = asyncio.get_event_loop().time()
        samples_sent = 0

        for i in range(0, total_samples, samples_per_frame):
            chunk = audio_int16[i : i + samples_per_frame]
            # Pad final frame if necessary
            if len(chunk) < samples_per_frame:
                chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))

            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_frame,
            )
            await self.source.capture_frame(frame)
            samples_sent += len(chunk)

            # Pace WebRTC delivery to wall-clock time
            expected_time = samples_sent / self.sample_rate
            now = asyncio.get_event_loop().time() - start_time
            if expected_time > now:
                await asyncio.sleep(expected_time - now)

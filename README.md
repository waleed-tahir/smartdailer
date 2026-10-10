# SmartDialer - Dual Voice Agent Framework

A voice agent framework with chunked streaming transcription (2.5-second chunks with 0.5-second overlap), multi-agent dialogue loops, modular support for **Faster-Whisper** and **NVIDIA Parakeet** (both GPU accelerated), and full **LiveKit WebRTC & SIP Telephony** integration.

---

## Architecture Overview

```
[Agent A (Speaker)] ──> WebRTC Track (20ms PCM Frames)
                                │
                                ▼
                       [AudioChunkBuffer]
               Window: 2.5s | Overlap: 0.5s | Step: 2.0s
                                │
                                ▼
                   [ASR Engine (Whisper / Parakeet)]
                   Transcribes sliding window chunks
                                │
                                ▼
                    [Transcript Reconciliation]
              Deduplicates overlapping boundary words
                                │
                                ▼
                      [Agent B Brain (LLM / Script)]
                                │
                                ▼
[Agent B (Speaker)] ──> WebRTC Track ──> [Agent A Listens] ...
```

---

## LiveKit WebRTC Integration

The `livekit` branch ports the SmartDialer architecture to the LiveKit ecosystem:

- **Custom LiveKit STT Adapter (`LiveKitChunkedSTT`)**: Implements `livekit.agents.stt.STT`, ingesting 20ms WebRTC `rtc.AudioFrame`s into the 2.5s chunk / 0.5s overlap buffer.
- **WebRTC Audio Publisher (`LiveKitAudioPublisher`)**: Streams synthesized speech as 20ms PCM audio frames into any LiveKit Room track.
- **Multi-Agent WebRTC Simulation (`livekit_main.py --mode dual`)**: Runs two agents exchanging WebRTC audio tracks in real time.
- **Production Agent Worker (`livekit_main.py --mode worker`)**: Joins LiveKit rooms to handle real telephone callers via LiveKit SIP or web callers.

---

## Quick Start

### 1. LiveKit Multi-Agent Demo (WebRTC Frames + Parakeet on GPU)

```bash
# Run 2 agents conversing over LiveKit WebRTC audio frames using NVIDIA Parakeet
python livekit_main.py --mode dual --asr parakeet --turns 4

# Or run using Faster-Whisper
python livekit_main.py --mode dual --asr whisper --turns 4
```

### 2. Standalone In-Memory Simulation

```bash
# Run standalone dialer demo
python main.py --asr parakeet --turns 4
python main.py --asr whisper --turns 4
```

### 3. Production LiveKit Agent Worker

Configure your LiveKit credentials in `.env` (copy from `.env.example`), then run:

```bash
python livekit_main.py --mode worker --asr parakeet
```

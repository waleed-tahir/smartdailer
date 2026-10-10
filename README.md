# SmartDialer - Dual Voice Agent Framework

A voice agent framework with chunked streaming transcription (2.5-second chunks with 0.5-second overlap), multi-agent dialogue loops, and modular support for both **Faster-Whisper** (GPU CUDA accelerated) and **NVIDIA Parakeet**.

---

## Architecture Overview

```
[Agent A (Speaker)] ──> Synthesizes Audio (16kHz PCM)
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
[Agent B (Speaker)] ──> Synthesizes Audio ──> [Agent A Listens] ...
```

---

## Features

1. **Sliding-Window Audio Chunking**:
   - Window size: **2.5 seconds** (40,000 samples @ 16kHz)
   - Overlap: **0.5 seconds** (8,000 samples @ 16kHz)
   - Step / Hop: **2.0 seconds** (32,000 samples @ 16kHz)
   - Real-time chunk streaming with final remainder flush so no trailing words are lost.

2. **ASR Engines**:
   - **Faster-Whisper**: GPU-accelerated CTranslate2 runtime with automatic Windows CUDA DLL discovery and float16/int8 precision.
   - **NVIDIA Parakeet**: Native integration class (`ParakeetASR`) ready for NeMo (`nvidia/parakeet-ctc-0.6b` / `nvidia/parakeet-tdt-1.1b`) or HuggingFace.

3. **Dual-Agent Dialogue Orchestrator**:
   - Agents speak, listen, transcribe each other's audio through chunked ASR, and reply back and forth.
   - Saves master dialogue recording to `outputs/conversation.wav`.

4. **Multi-Engine Dialogue Brains**:
   - `ScenarioBrain`: Deterministic script engine for instant offline testing without API keys.
   - `LLMBrain`: OpenAI, Groq, Ollama, or any OpenAI-compatible LLM endpoint.

---

## Quick Start

### 1. Run the Dual-Agent Demo (Live Audio by Default)

With virtual environment active:
```bash
# Plays audio out loud through speakers in real time as agents converse
python main.py --turns 4
```

### 2. CLI Options

```bash
# Run in silent/fast batch mode without live speaker playback
python main.py --turns 4 --no-live

# Custom chunk & overlap durations
python main.py --chunk-sec 2.5 --overlap-sec 0.5 --turns 4

# Change Whisper model size (tiny, base, small, medium)
python main.py --whisper-model small

# Switch to NVIDIA Parakeet (when NeMo toolkit is installed)
python main.py --asr parakeet
```

---

## Switching to NVIDIA Parakeet

To run NVIDIA Parakeet:
1. Install NeMo ASR toolkit:
   ```bash
   pip install nemo_toolkit[asr]
   ```
2. Run with the `--asr parakeet` flag:
   ```bash
   python main.py --asr parakeet
   ```
Or set `asr_backend = "parakeet"` in [src/smartdailer/config.py](file:///c:/Users/PC/Desktop/smartdailer/src/smartdailer/config.py).

---

## Technical Documentation & Knowledge Transfer Report

A comprehensive **26-page Technical Implementation & Knowledge Transfer Report** is provided in the repository:
- **PDF Report**: [`docs/technical_report.pdf`](docs/technical_report.pdf)
- **LaTeX Source**: [`docs/technical_report.tex`](docs/technical_report.tex)

The report covers:
1. Mathematical formulation of the 2.5s window / 0.5s overlap acoustic buffer ($N_w=40\,000, N_o=8\,000, N_s=32\,000$ samples @ 16kHz).
2. Deep architectural comparison of CTranslate2 Faster-Whisper (FP16) vs NVIDIA NeMo 3.0 Conformer-CTC Parakeet (`nvidia/parakeet-ctc-0.6b`).
3. Suffix-prefix token reconciliation and boundary word deduplication algorithm.
4. Non-blocking SAPI5 COM streaming and Edge-TTS synthesis architecture.
5. Multi-agent turn-taking finite state machine and PortAudio live speaker auditing.
6. Engineering post-mortems (Windows CUDA DLL binding, COM apartment concurrency, CP1252 character encoding, cuDNN symbol resolution).
7. Developer operational handbook and deployment instructions.


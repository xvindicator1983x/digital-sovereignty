# Hermes Agent Plugin: Local Streaming TTS with Qwen3-TTS

**Problem:** Hermes Agent's desktop voice conversation mode can't stream sentence-by-sentence with local TTS engines. It only streams with online providers (ElevenLabs, OpenAI, Gemini, xAI). With local voices, you get whole-reply playback — the model finishes generating everything before any speech starts. For a 30-second reply, that's 30 seconds of silence.

**Solution:** The `ava-tts-stream` plugin registers a local Qwen3-TTS voice as a Hermes streaming TTS provider, enabling true sentence-by-sentence streaming where speech overlaps generation.

## What This Plugin Does

- Registers the local Qwen3-TTS voice (`qwentts.cpp` on your LAN) as a Hermes `StreamingTTSProvider`
- Requests raw PCM audio (24 kHz mono int16) — exactly what qwentts.cpp emits
- Pipes audio through ffmpeg's `atempo` filter for pitch-preserving speed control
- Yields audio bytes as they arrive, so speech starts while the model is still generating
- Locks voice timbre across sentences using a pinned seed and subtalker sampling parameters

## Architecture

```
[Hermes Agent generates text]
        ↓
[Sentence splitter cuts into chunks]
        ↓
[ava-tts-stream plugin (this)]
        ↓
[qwentts.cpp server on LAN (:8891)]
        ↓
[Raw PCM 24kHz mono int16]
        ↓
[ffmpeg atempo speed adjustment]
        ↓
[Desktop audio output — sentence-by-sentence]
```

## Prerequisites

1. **qwentts.cpp server** running on your LAN (see [Qwen3-TTS deployment guide](https://github.com/NousResearch/hermes-agent/docs) for setup)
2. **ffmpeg** installed on the machine running Hermes Agent
3. **Hermes Agent** with desktop app

## Installation

### Option 1: Install from source

```bash
# Clone or download the plugin
git clone https://github.com/Diemos/digital-sovereignty.git
cp -r digital-sovereignty/plugins/ava-tts-stream ~/.hermes/plugins/

# Restart Hermes backend
systemctl --user restart hermes-backend
```

### Option 2: Manual installation

Create `~/.hermes/plugins/ava-tts-stream/` with two files:

**plugin.yaml:**
```yaml
name: ava-tts-stream
version: 1.0.0
description: "Registers the local Qwen3-TTS voice as a Hermes streaming TTS provider for sentence-by-sentence desktop voice conversation."
author: Diemos
```

**__init__.py:**
```python
"""ava-tts-stream — register the local Qwen3-TTS voice as a Hermes streaming provider.

The desktop's speak-stream relay only streams when a StreamingTTSProvider is registered
for the configured TTS provider name. This plugin closes that gap for local voices.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Dict, Iterator

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8891"  # Your qwentts.cpp server (change to your LAN IP)
DEFAULT_VOICE = "vivian"
DEFAULT_MODEL = "qwen3-tts"
DEFAULT_SPEED = 1.2
DEFAULT_INSTRUCTIONS = (
    "Speak in a calm, warm, confident tone with steady even pacing and natural "
    "conversational rhythm."
)
DEFAULT_SEED = 7
DEFAULT_SUBTALKER_TEMPERATURE = 0.5
DEFAULT_SUBTALKER_TOP_P = 0.8

_MIN_SPEED, _MAX_SPEED = 0.5, 2.0


def _section(tts_config: Dict) -> Dict:
    providers = tts_config.get("providers") or {}
    return providers.get("ava") or {}


class AvaStreamer:
    """Local Qwen3-TTS via qwentts.cpp's OpenAI-compatible endpoint, PCM-streamed."""

    sample_rate = 24000
    channels = 1
    sample_width = 2  # int16

    def __init__(self, tts_config: Dict, section: Dict):
        merged = {**_section(tts_config), **(section or {})}
        self.url = str(merged.get("url") or DEFAULT_URL).rstrip("/")
        self.voice = str(merged.get("voice") or DEFAULT_VOICE)
        self.model = str(merged.get("model") or DEFAULT_MODEL)
        speed = float(merged.get("speed", DEFAULT_SPEED))
        self.speed = min(max(speed, _MIN_SPEED), _MAX_SPEED)
        self.instructions = str(merged.get("instructions") or DEFAULT_INSTRUCTIONS)
        seed = merged.get("seed", DEFAULT_SEED)
        self.seed = int(seed) if seed is not None else None
        self.subtalker_temperature = float(merged.get("subtalker_temperature",
                                                      DEFAULT_SUBTALKER_TEMPERATURE))
        self.subtalker_top_p = float(merged.get("subtalker_top_p", DEFAULT_SUBTALKER_TOP_P))

    @staticmethod
    def available() -> bool:
        return shutil.which("ffmpeg") is not None

    def stream(self, text: str) -> Iterator[bytes]:
        import select
        import requests

        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",
            "instructions": self.instructions,
            "subtalker_temperature": self.subtalker_temperature,
            "subtalker_top_p": self.subtalker_top_p,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(self.sample_rate), "-ac", str(self.channels),
            "-i", "pipe:0",
            "-af", f"atempo={self.speed}",
            "-f", "s16le", "-ar", str(self.sample_rate), "-ac", str(self.channels),
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out_fd = proc.stdout.fileno()

        def _drain_ready() -> Iterator[bytes]:
            while True:
                ready, _, _ = select.select([out_fd], [], [], 0.0)
                if not ready:
                    return
                out = os.read(out_fd, 65536)
                if not out:
                    return
                yield out

        try:
            resp = requests.post(
                f"{self.url}/v1/audio/speech",
                json=payload,
                timeout=(5.0, 120.0),
                stream=True,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"ava TTS HTTP {resp.status_code}: {resp.text[:200]}")
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                proc.stdin.write(chunk)
                yield from _drain_ready()
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                tail = proc.stdout.read()
                if tail:
                    yield tail
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def register(ctx) -> None:
    from tools.tts_streaming import StreamingTTSProvider, register as _register

    @_register("ava")
    class _AvaRegistered(AvaStreamer, StreamingTTSProvider):
        pass

    logger.info("ava-tts-stream: registered 'ava' streaming TTS provider (url=%s)", DEFAULT_URL)


def _module_level_register() -> None:
    try:
        from tools.tts_streaming import StreamingTTSProvider, register as _register

        if "ava" not in __import__("tools.tts_streaming", fromlist=["_REGISTRY"])._REGISTRY:
            @_register("ava")
            class _AvaModuleLevel(AvaStreamer, StreamingTTSProvider):
                pass
    except Exception as exc:
        logger.debug("ava-tts-stream module-level register skipped: %s", exc)


_module_level_register()
```

## Configuration

In your Hermes `config.yaml`, configure the TTS provider:

```yaml
tts:
  provider: ava
  providers:
    ava:
      url: "http://172.16.0.2:8891"  # Your qwentts.cpp server address
      voice: "vivian"               # Voice name
      model: "qwen3-tts"            # Model name
      speed: 1.2                    # Speech rate (0.5 - 2.0)
      seed: 7                       # Pinned for consistent timbre
      subtalker_temperature: 0.5    # Lower = more consistent voice
      subtalker_top_p: 0.8          # Lower = less voice variation
```

## Testing

1. Enable voice mode in the Hermes desktop app
2. Click the waveform button to start a conversation
3. Ask a question and listen — speech should begin while the model is still generating text
4. Verify sentence-by-sentence streaming (not whole-reply playback)

## Why This Matters for Digital Sovereignty

Most local TTS solutions either:
- Require cloud APIs (ElevenLabs, OpenAI) — sending your data out
- Are slow and produce robotic speech (older local engines)
- Can't stream sentence-by-sentence — defeating the purpose of voice conversation

This plugin proves you can have a fully local, fast, natural-sounding voice assistant that streams in real-time. No cloud dependencies, no telemetry, no API keys. Just your hardware, your models, your voice. 🔥

## Troubleshooting

**No audio at all:** Check that qwentts.cpp server is running and reachable from the Hermes Agent machine. Test with `curl http://<server>:8891/v1/audio/speech`.

**Robotic or inconsistent voice across sentences:** Increase `subtalker_temperature` toward 1.0 for more variation, or decrease toward 0.5 for more consistency. The default (0.5) was chosen through A/B testing for optimal balance.

**Too fast/slow:** Adjust the `speed` parameter (0.5 = half speed, 2.0 = double speed). Default is 1.2x.

**Latency too high:** Ensure qwentts.cpp is running on a GPU with CUDA support. CPU inference adds significant latency.

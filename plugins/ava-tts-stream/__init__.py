"""ava-tts-stream — register the local Qwen3-TTS voice as a Hermes streaming provider.

The desktop's speak-stream relay (and the CLI/TUI speaker pipeline) only stream when a
``StreamingTTSProvider`` is registered for the configured TTS provider name. The built-ins
cover online providers (elevenlabs/openai/gemini/xai); our local voice ``ava`` (qwentts.cpp,
OpenAI-compatible /v1/audio/speech on LOQ-AUTONOMY:8891) had none, so the relay answered
{"type": "fallback"} and desktop conversation mode fell back to whole-reply playback.

This plugin closes that gap through the intended extension point: it registers an ``ava``
streamer that requests raw PCM (24 kHz mono int16 — exactly what qwentts.cpp emits and what
the framework consumes) and pipes it through ffmpeg ``atempo`` for the user's preferred
speaking pace. Audio bytes are yielded as they arrive, so speech overlaps generation.

Config lives in the existing provider section:  tts.providers.ava.{url,voice,model,speed}
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Dict, Iterator

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8891"  # Your qwentts.cpp server address
DEFAULT_VOICE = "vivian"
DEFAULT_MODEL = "qwen3-tts"
DEFAULT_SPEED = 1.2

# Per-sentence streaming means every sentence is an independent synthesis call — with no
# guidance the model samples fresh prosody each time and the voice "wanders" (A/B tested
# 2026-09-17: bare runs of one sentence spanned 3.6–4.6s with different energy profiles;
# instruction + pinned seed = byte-identical). A stable persona instruction keeps Ava's
# tone consistent across the whole reply; the seed pins the sampler for reproducibility.
DEFAULT_INSTRUCTIONS = (
    "Speak in a calm, warm, confident tone with steady even pacing and natural "
    "conversational rhythm."
)
DEFAULT_SEED = 7

# Cross-sentence timbre lock: with per-sentence streaming the subtalker (token->acoustic
# stage) re-samples the voice realization on every call, so Ava sounded like a different
# person each sentence. Tightening both the subtalker temperature AND top_p was the only
# setting that held one identity across a 4-sentence passage.
DEFAULT_SUBTALKER_TEMPERATURE = 0.5
DEFAULT_SUBTALKER_TOP_P = 0.8

# ffmpeg atempo accepts one filter in [0.5, 2.0]; clamp instead of chaining for now —
# speeds outside that band are a config mistake, not an expected case.
_MIN_SPEED, _MAX_SPEED = 0.5, 2.0


def _section(tts_config: Dict) -> Dict:
    """Our provider settings live under tts.providers.ava (command-provider layout)."""
    providers = tts_config.get("providers") or {}
    return providers.get("ava") or {}


class AvaStreamer:
    """Local Qwen3-TTS via qwentts.cpp's OpenAI-compatible endpoint, PCM-streamed."""

    sample_rate = 24000
    channels = 1
    sample_width = 2  # int16

    def __init__(self, tts_config: Dict, section: Dict):
        # The resolver hands us tts.<name> (usually absent for command providers); our real
        # settings live under tts.providers.ava. Merge both, explicit tts.ava wins.
        merged = {**_section(tts_config), **(section or {})}
        self.url = str(merged.get("url") or DEFAULT_URL).rstrip("/")
        self.voice = str(merged.get("voice") or DEFAULT_VOICE)
        self.model = str(merged.get("model") or DEFAULT_MODEL)
        speed = float(merged.get("speed", DEFAULT_SPEED))
        self.speed = min(max(speed, _MIN_SPEED), _MAX_SPEED)
        # Prosody anchors for per-sentence streaming (see DEFAULT_INSTRUCTIONS note).
        self.instructions = str(merged.get("instructions") or DEFAULT_INSTRUCTIONS)
        seed = merged.get("seed", DEFAULT_SEED)
        self.seed = int(seed) if seed is not None else None
        # Subtalker sampling — the cross-sentence timbre lock (see defaults above).
        self.subtalker_temperature = float(merged.get("subtalker_temperature",
                                                      DEFAULT_SUBTALKER_TEMPERATURE))
        self.subtalker_top_p = float(merged.get("subtalker_top_p", DEFAULT_SUBTALKER_TOP_P))

    @staticmethod
    def available() -> bool:
        # Endpoint reachability is checked lazily per-stream (the laptop may be asleep
        # after a lid close); registration only requires the binary we pipe through.
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
            """Yield whatever decoded audio is ready NOW (non-blocking)."""
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
            # EOF on stdin makes ffmpeg flush its tail; drain it blocking.
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
            if proc.returncode not in (0, None):
                logger.warning("ava TTS ffmpeg exited %s", proc.returncode)


def register(ctx) -> None:  # noqa: ANN001 - plugin contract
    from tools.tts_streaming import StreamingTTSProvider, register as _register

    @_register("ava")
    class _AvaRegistered(AvaStreamer, StreamingTTSProvider):
        pass

    logger.info("ava-tts-stream: registered 'ava' streaming TTS provider (url=%s)", DEFAULT_URL)


# Module-level registration so the streamer is live even if a consumer imports this module
# directly (tests, manual probes) without going through the plugin loader.
def _module_level_register() -> None:
    try:
        from tools.tts_streaming import StreamingTTSProvider, register as _register

        if "ava" not in __import__("tools.tts_streaming", fromlist=["_REGISTRY"])._REGISTRY:
            @_register("ava")
            class _AvaModuleLevel(AvaStreamer, StreamingTTSProvider):
                pass
    except Exception as exc:  # core import unavailable (e.g. standalone test) — fine
        logger.debug("ava-tts-stream module-level register skipped: %s", exc)


_module_level_register()

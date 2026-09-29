# ava-tts speak-stream discovery patch

## What it is
4-line local patch to `hermes_cli/web_routers/audio.py` (Hermes Agent repo at
`~/.hermes/hermes-agent`). The `/api/audio/speak-stream` WebSocket endpoint —
used by desktop voice conversation for sentence-by-sentence streaming TTS —
resolves the streaming provider WITHOUT running plugin discovery. Plugin-
registered streamers (our local `ava` voice, plugin `ava-tts-stream`) therefore
never appear in the registry and voice chat silently falls back to whole-reply
playback (or nothing).

The fix: call `discover_plugins(force=True)` inside the `_resolve()` closure
before resolving the provider. See patch file for exact diff.

## Why it's local-only
Upstream does not have this fix (verified absent in commit b8a8be185,
2026-09-29). We are not sending it upstream at this time (Mateo decision).

## Re-apply after ANY Hermes update
```bash
cd ~/.hermes/hermes-agent
grep -q "discover_plugins(force=True)" hermes_cli/web_routers/audio.py \
  && echo "patch present" \
  || { git apply ~/digital-sovereignty/patches/ava-tts-speakstream-discovery.patch
       && systemctl --user restart hermes-backend-default; }
```
If the patch context has drifted on a newer upstream, re-derive it:
1. Read `speak_stream_ws` in `hermes_cli/web_routers/audio.py`
2. Add the 4 lines (comment + import + call) at the top of `_resolve()`,
   before `from tools.tts_streaming import resolve_streaming_provider`
3. Restart backend, then verify: desktop voice conversation streams
   sentence-by-sentence (synthesis hits on LOQ-AUTONOMY qwen3-tts-cpp
   service DURING generation, per journalctl)

## Safety copies of this patch
1. `~/.hermes/ava-tts-speakstream-discovery.patch` (this file's twin)
2. Git branch `local/ava-tts-discovery` in `~/.hermes/hermes-agent`
3. This repo: `patches/ava-tts-speakstream-discovery.patch` (+ git history, remote)

## Related
- Live plugin: `~/.hermes/plugins/ava-tts-stream/__init__.py`
  (backup copy: `~/.hermes/ava-tts-stream-init-backup.py`)
- Plugin source in this repo: `plugins/ava-tts-stream/`

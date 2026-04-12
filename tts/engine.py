"""
TTS Voice Definitions & Synthesis
==================================
Uses edge-tts (Microsoft Edge Read Aloud) — free, no API key needed.
Also supports Fish Audio voice cloning via reference MP3 files.

Voice catalogue:
  English:
    - Female / American  (Jenny – news anchor style)
    - Male   / American  (Guy – news anchor style)
    - Female / British   (Sonia – London accent)
    - Male   / British   (Ryan – London accent)
  Chinese:
    - Female / Beijing Mandarin  (Xiaoxiao – broadcast style)
    - Male   / Beijing Mandarin  (Yunxi – broadcast style)
    - Female / Cantonese         (XiaoMin – Guangzhou)
    - Male   / Cantonese         (YunSong – Guangzhou)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from dataclasses import dataclass

import edge_tts


# ---------------------------------------------------------------------------
# Voice definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Voice:
    """A selectable TTS voice."""
    id: str            # edge-tts voice name
    label_zh: str      # Chinese UI label
    label_en: str      # English UI label
    lang: str          # "en" or "zh"


# English voices
EN_VOICES: list[Voice] = [
    Voice("en-US-JennyNeural",      "🇺🇸 女声 · 美式播音 (Jenny)",       "🇺🇸 Female · American News (Jenny)",      "en"),
    Voice("en-US-GuyNeural",        "🇺🇸 男声 · 美式播音 (Guy)",         "🇺🇸 Male · American News (Guy)",          "en"),
    Voice("en-US-AriaNeural",       "🇺🇸 女声 · 纽约腔 (Aria)",         "🇺🇸 Female · New York (Aria)",            "en"),
    Voice("en-US-DavisNeural",      "🇺🇸 男声 · 纽约腔 (Davis)",        "🇺🇸 Male · New York (Davis)",             "en"),
    Voice("en-GB-SoniaNeural",      "🇬🇧 女声 · 英式伦敦腔 (Sonia)",    "🇬🇧 Female · British London (Sonia)",     "en"),
    Voice("en-GB-RyanNeural",       "🇬🇧 男声 · 英式伦敦腔 (Ryan)",     "🇬🇧 Male · British London (Ryan)",        "en"),
]

# Chinese voices
ZH_VOICES: list[Voice] = [
    Voice("zh-CN-XiaoxiaoNeural",   "🇨🇳 女声 · 北京普通话 · 播音腔 (晓晓)", "🇨🇳 Female · Beijing Mandarin · Broadcast (Xiaoxiao)", "zh"),
    Voice("zh-CN-YunxiNeural",      "🇨🇳 男声 · 北京普通话 · 播音腔 (云希)", "🇨🇳 Male · Beijing Mandarin · Broadcast (Yunxi)",     "zh"),
    Voice("zh-CN-XiaoyiNeural",     "🇨🇳 女声 · 北京普通话 · 主持腔 (晓伊)", "🇨🇳 Female · Beijing Mandarin · Host (Xiaoyi)",       "zh"),
    Voice("zh-CN-YunjianNeural",    "🇨🇳 男声 · 北京普通话 · 主持腔 (云健)", "🇨🇳 Male · Beijing Mandarin · Host (Yunjian)",        "zh"),
    Voice("zh-HK-HiuGaaiNeural",   "🇭🇰 女声 · 广州粤语 (曉佳)",           "🇭🇰 Female · Cantonese Guangzhou (HiuGaai)",          "zh"),
    Voice("zh-HK-WanLungNeural",   "🇭🇰 男声 · 广州粤语 (万龙)",           "🇭🇰 Male · Cantonese Guangzhou (WanLung)",            "zh"),
]

ALL_VOICES: list[Voice] = EN_VOICES + ZH_VOICES


def get_voice_options(lang: str, ui_lang: str) -> dict[str, str]:
    """Return {display_label: voice_id} for a given content language.

    Args:
        lang: content language — "en" for English voices, "zh" for Chinese voices, "all" for both.
        ui_lang: UI language for label display — "zh" or "en".
    """
    if lang == "all":
        voices = ALL_VOICES
    elif lang == "en":
        voices = EN_VOICES
    elif lang == "zh":
        voices = ZH_VOICES
    else:
        voices = ALL_VOICES

    if ui_lang == "zh":
        return {v.label_zh: v.id for v in voices}
    else:
        return {v.label_en: v.id for v in voices}


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

async def _synthesize_async(text: str, voice_id: str, rate: str = "+0%") -> bytes:
    """Synthesize text to MP3 bytes using edge-tts."""
    communicate = edge_tts.Communicate(text=text, voice=voice_id, rate=rate)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    return buffer.getvalue()


def synthesize(text: str, voice_id: str, rate: str = "+0%") -> bytes:
    """Synchronous wrapper — returns MP3 bytes.

    Args:
        text: The text to speak.
        voice_id: edge-tts voice name, e.g. "en-US-JennyNeural".
        rate: Speed adjustment, e.g. "+0%", "+20%", "-10%".

    Returns:
        MP3 audio bytes.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an already-running loop (e.g. Streamlit)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _synthesize_async(text, voice_id, rate))
                return future.result(timeout=120)
        else:
            return loop.run_until_complete(_synthesize_async(text, voice_id, rate))
    except RuntimeError:
        return asyncio.run(_synthesize_async(text, voice_id, rate))


# ---------------------------------------------------------------------------
# Fish Audio voice cloning
# ---------------------------------------------------------------------------

def synthesize_with_sample(text: str, reference_audio_path: str, api_key: str) -> bytes:
    """Synthesize speech using Fish Audio TTS with a reference MP3 for voice cloning.

    Calls the Fish Audio /v1/tts endpoint with the ``otter`` model.
    Reference audio is encoded as base64 and sent alongside the text.

    Args:
        text: The text to synthesize.
        reference_audio_path: Absolute path to the reference MP3 file.
        api_key: Fish Audio API key (https://fish.audio/).

    Returns:
        MP3 audio bytes.

    Raises:
        RuntimeError: On API error or missing dependencies.
    """
    import base64

    try:
        import requests as _requests
    except ImportError as exc:
        raise RuntimeError("requests library is required: pip install requests") from exc

    if not api_key:
        raise RuntimeError("Fish Audio API key is not configured. Please set FISH_AUDIO_API_KEY in config.py or the sidebar.")

    if not os.path.isfile(reference_audio_path):
        raise RuntimeError(f"Reference audio file not found: {reference_audio_path}")

    # Read & base64-encode the reference audio
    with open(reference_audio_path, "rb") as f:
        ref_bytes = f.read()
    ref_b64 = base64.b64encode(ref_bytes).decode("utf-8")

    # Determine MIME type from extension
    ext = os.path.splitext(reference_audio_path)[1].lower().lstrip(".")
    mime_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "m4a": "audio/mp4", "ogg": "audio/ogg"}
    ref_mime = mime_map.get(ext, "audio/mpeg")

    payload = {
        "text": text,
        "format": "mp3",
        "references": [
            {
                "audio": f"data:{ref_mime};base64,{ref_b64}",
                "text": "",   # no transcript required
            }
        ],
        "normalize": True,
        "latency": "normal",
    }

    resp = _requests.post(
        "https://api.fish.audio/v1/tts",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=120,
        stream=True,
    )

    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Fish Audio API error {resp.status_code}: {err}")

    # Stream the response into memory
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            buf.write(chunk)
    return buf.getvalue()


def get_sample_voices(sample_dir: str) -> dict[str, str]:
    """Scan *sample_dir* for audio files and return {display_name: abs_path}.

    Supported formats: mp3, wav, flac, m4a, ogg.
    Display name is the stem of the filename (without extension).
    """
    if not os.path.isdir(sample_dir):
        return {}
    exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    result: dict[str, str] = {}
    for fname in sorted(os.listdir(sample_dir)):
        fpath = os.path.join(sample_dir, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in exts:
            label = os.path.splitext(fname)[0]
            result[label] = fpath
    return result


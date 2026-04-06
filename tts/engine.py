"""
TTS Voice Definitions & Synthesis
==================================
Uses edge-tts (Microsoft Edge Read Aloud) — free, no API key needed.

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

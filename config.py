import os

# --- LLM Provider ---
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL", None)

# --- Default language ---
DEFAULT_LANGUAGE: str = "zh"  # "zh" | "en"

# --- Database ---
# Use THINKER_DB_PATH env var when set (useful for Streamlit Cloud or Docker).
# Otherwise, prefer the local ./db/ directory; if it is not writable (e.g. on
# Streamlit Cloud where /mount/src is read-only), fall back to /tmp/thinker/.
def _resolve_db_path() -> str:
    env_path = os.environ.get("THINKER_DB_PATH", "")
    if env_path:
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        return env_path

    default_dir = os.path.join(os.path.dirname(__file__), "db")
    try:
        os.makedirs(default_dir, exist_ok=True)
        probe = os.path.join(default_dir, ".write_probe")
        with open(probe, "w") as f:
            f.write("")
        os.unlink(probe)
        return os.path.join(default_dir, "thinker.db")
    except OSError:
        fallback_dir = os.path.join(os.sep, "tmp", "thinker")
        os.makedirs(fallback_dir, exist_ok=True)
        return os.path.join(fallback_dir, "thinker.db")


DB_PATH: str = _resolve_db_path()

# --- Book PDF directory ---
BOOK_DIR: str = os.path.join(os.path.dirname(__file__), "book")

# --- Fish Audio TTS (voice cloning with reference audio) ---
# Get your API key at: https://fish.audio/
FISH_AUDIO_API_KEY: str = os.environ.get("FISH_AUDIO_API_KEY", "")

# --- Sample voice directory (MP3 reference files for Fish Audio voice cloning) ---
# Add more MP3 files to this directory to extend the sample voice list
SAMPLE_VOICE_DIR: str = os.path.join(os.path.dirname(__file__), "video_sample")


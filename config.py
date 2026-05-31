import os

# --- LLM Provider ---
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL", None)

# --- Default language ---
DEFAULT_LANGUAGE: str = "zh"  # "zh" | "en"

# --- Database ---
# On Streamlit Cloud the source directory is read-only, so fall back to /tmp
# which is always writable.  Local development uses the in-repo path so that
# the DB file is preserved across restarts.
_default_db_path: str = os.path.join(os.path.dirname(__file__), "db", "thinker.db")
DB_PATH: str = (
    _default_db_path
    if os.access(os.path.dirname(_default_db_path), os.W_OK)
    else os.path.join("/tmp", "thinker.db")
)

# --- Book PDF directory ---
BOOK_DIR: str = os.path.join(os.path.dirname(__file__), "book")

# --- Fish Audio TTS (voice cloning with reference audio) ---
# Get your API key at: https://fish.audio/
FISH_AUDIO_API_KEY: str = os.environ.get("FISH_AUDIO_API_KEY", "")

# --- Sample voice directory (MP3 reference files for Fish Audio voice cloning) ---
# Add more MP3 files to this directory to extend the sample voice list
SAMPLE_VOICE_DIR: str = os.path.join(os.path.dirname(__file__), "video_sample")


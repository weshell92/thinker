import os

# --- LLM Provider ---
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL", None)

# --- Default language ---
DEFAULT_LANGUAGE: str = "zh"  # "zh" | "en"

# --- Database ---
DB_PATH: str = os.path.join(os.path.dirname(__file__), "db", "thinker.db")

# --- Book PDF directory ---
BOOK_DIR: str = os.path.join(os.path.dirname(__file__), "book")

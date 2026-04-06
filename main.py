"""
Thinker – Critical Thinking Analyzer
=====================================
Streamlit application entry‑point.

Run:
    streamlit run main.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so local imports work with Streamlit
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from analyzer.engine import ThinkerEngine  # noqa: E402
from analyzer.models import AnalysisResult  # noqa: E402
from analyzer.providers.openai_provider import OpenAIProvider, ProviderError  # noqa: E402
from book.reader import (  # noqa: E402
    discover_books, load_book, extract_chapter_text,
    extract_chapter_images, is_scanned_pdf, Chapter,
)
from db.database import Database  # noqa: E402
from tts.engine import get_voice_options, synthesize  # noqa: E402

# ---------------------------------------------------------------------------
# i18n helper
# ---------------------------------------------------------------------------

_I18N_CACHE: dict[str, dict[str, str]] = {}


def t(key: str, lang: str = "zh", **kwargs: str) -> str:
    """Return a translated UI string."""
    if lang not in _I18N_CACHE:
        i18n_path = os.path.join(_PROJECT_ROOT, "i18n", f"{lang}.json")
        with open(i18n_path, "r", encoding="utf-8") as fh:
            _I18N_CACHE[lang] = json.load(fh)
    text = _I18N_CACHE[lang].get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db() -> Database:
    return Database(config.DB_PATH)


@st.cache_data
def cached_load_book(pdf_path: str):
    """Load and cache a book's TOC structure."""
    return load_book(pdf_path)


@st.cache_data
def cached_extract_chapter(pdf_path: str, page_start: int, page_end: int) -> str:
    """Extract and cache chapter text by page range."""
    ch = Chapter(level=1, title="", page_start=page_start, page_end=page_end)
    return extract_chapter_text(pdf_path, ch)


@st.cache_data
def cached_extract_images(pdf_path: str, page_start: int, page_end: int) -> list[bytes]:
    """Extract and cache chapter page images."""
    ch = Chapter(level=1, title="", page_start=page_start, page_end=page_end)
    return extract_chapter_images(pdf_path, ch)


@st.cache_data
def cached_is_scanned(pdf_path: str) -> bool:
    """Check and cache whether a PDF is scanned (image-only)."""
    return is_scanned_pdf(pdf_path)



# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _render_bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "—"


def render_result(result: AnalysisResult, lang: str) -> None:
    """Render the four‑step analysis result as Streamlit expanders."""

    # Step 1 – Facts / Emotions / Assumptions
    with st.expander(t("step1_title", lang), expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**{t('facts_title', lang)}**")
            st.markdown(_render_bullet_list(result.facts))
        with col2:
            st.markdown(f"**{t('emotions_title', lang)}**")
            st.markdown(_render_bullet_list(result.emotions))
        with col3:
            st.markdown(f"**{t('assumptions_title', lang)}**")
            st.markdown(_render_bullet_list(result.assumptions))

    # Step 2 – Fallacies
    with st.expander(t("step2_title", lang), expanded=True):
        if result.fallacies:
            for f in result.fallacies:
                st.markdown(f"**{f.name}**")
                st.markdown(f"> {f.explanation}")
        else:
            st.info(t("no_fallacies", lang))

    # Step 3 – Three explanations
    with st.expander(t("step3_title", lang), expanded=True):
        for i, explanation in enumerate(result.explanations, 1):
            st.markdown(f"**{i}.** {explanation}")

    # Step 4 – Rational conclusion
    with st.expander(t("step4_title", lang), expanded=True):
        st.markdown(result.rational_conclusion)

    # Read aloud the full analysis summary
    _full_text = "\n".join([
        *result.facts,
        *result.emotions,
        *result.assumptions,
        *[f"{f.name}: {f.explanation}" for f in result.fallacies],
        *result.explanations,
        result.rational_conclusion,
    ])
    _do_tts(_full_text, lang, btn_key="tts_analysis_result")


# ---------------------------------------------------------------------------
# Tab 1: Analysis
# ---------------------------------------------------------------------------

def page_analyze(lang: str, api_key: str, model: str, base_url: str, provider_name: str = "") -> None:
    """Render the critical-thinking analysis page."""
    db = get_db()

    # ---- View a saved record ----
    if st.session_state.get("view_record_id") is not None:
        rec = db.get_record_by_id(st.session_state.view_record_id)
        if rec and rec.result:
            st.info(f"📄 {rec.input_text}")
            render_result(rec.result, rec.language)
            if st.button("↩️ Back / 返回"):
                st.session_state.view_record_id = None
                st.rerun()
            return
        else:
            st.session_state.view_record_id = None

    # ---- New analysis form ----
    user_text = st.text_area(
        t("input_placeholder", lang),
        height=200,
        key="input_text",
    )

    if st.button(t("analyze_button", lang), type="primary"):
        if not api_key:
            st.warning(t("error_no_key", lang))
            return
        if not user_text or not user_text.strip():
            st.warning(t("error_no_text", lang))
            return

        provider = OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url if base_url else None,
        )
        engine = ThinkerEngine(provider)

        with st.spinner(t("analyzing", lang)):
            try:
                result = engine.analyze(user_text.strip(), language=lang)
            except ProviderError as exc:
                error_code = str(exc)
                if "QUOTA_EXCEEDED" in error_code:
                    st.error(t("error_quota", lang, provider=provider_name))
                    st.caption(f"🔍 Detail: `{error_code}`")
                elif "AUTH_ERROR" in error_code:
                    st.error(t("error_auth", lang))
                elif "CONTEXT_TOO_LONG" in error_code:
                    st.error(t("error_context_too_long", lang))
                elif "RATE_LIMIT" in error_code:
                    st.warning(t("error_rate_limit", lang))
                    st.caption(f"🔍 Detail: `{error_code}`")
                else:
                    st.error(t("error_analysis", lang, error=error_code))
                return
            except Exception as exc:
                st.error(t("error_analysis", lang, error=str(exc)))
                return

        render_result(result, lang)
        db.save_record(user_text.strip(), lang, result)
        st.toast(t("saved", lang))


# ---------------------------------------------------------------------------
# Tab 2: Book Reader
# ---------------------------------------------------------------------------

def page_read_book(lang: str) -> None:
    """Render the PDF book reader page with chapter navigation."""
    st.header(t("book_title", lang))

    pdf_paths = discover_books(config.BOOK_DIR)
    if not pdf_paths:
        st.warning(t("book_no_books", lang))
        return

    # ---- Book selector ----
    book_labels = [os.path.basename(p) for p in pdf_paths]
    selected_label = st.selectbox(t("book_select", lang), book_labels)
    selected_path = pdf_paths[book_labels.index(selected_label)]

    # ---- Load TOC ----
    book_info = cached_load_book(selected_path)

    # ---- Layout: TOC left, content right ----
    col_toc, col_content = st.columns([1, 3])

    with col_toc:
        st.subheader(t("book_chapters", lang))
        for idx, ch in enumerate(book_info.chapters):
            indent = "\u3000" * (ch.level - 1)  # CJK space for indentation
            page_hint = f"p.{ch.page_start + 1}\u2013{ch.page_end + 1}"
            if st.button(
                f"{indent}{ch.title}",
                key=f"ch_{idx}",
                help=page_hint,
                use_container_width=True,
            ):
                st.session_state.selected_chapter_idx = idx
                st.session_state.selected_book_path = selected_path

    with col_content:
        ch_idx = st.session_state.get("selected_chapter_idx")
        ch_book = st.session_state.get("selected_book_path")

        if (
            ch_idx is not None
            and ch_book == selected_path
            and ch_idx < len(book_info.chapters)
        ):
            ch = book_info.chapters[ch_idx]
            st.subheader(ch.title)
            st.caption(
                t("book_page_range", lang,
                  start=str(ch.page_start + 1),
                  end=str(ch.page_end + 1))
            )

            scanned = cached_is_scanned(selected_path)

            if not scanned:
                # Text-based PDF → extract and show text
                with st.spinner("Loading…"):
                    text = cached_extract_chapter(
                        selected_path, ch.page_start, ch.page_end
                    )
                if text:
                    st.text(text)
                    _do_tts(text, lang, btn_key=f"tts_chapter_{ch_idx}")
                else:
                    # This specific page has no text (e.g. cover image)
                    with st.spinner("Loading…"):
                        imgs = cached_extract_images(
                            selected_path, ch.page_start, ch.page_end
                        )
                    for i, img_bytes in enumerate(imgs):
                        st.image(img_bytes, caption=f"Page {ch.page_start + i + 1}")
            else:
                # Scanned/image PDF → render as images
                with st.spinner("Loading…"):
                    imgs = cached_extract_images(
                        selected_path, ch.page_start, ch.page_end
                    )
                for i, img_bytes in enumerate(imgs):
                    st.image(img_bytes, caption=f"Page {ch.page_start + i + 1}")
        else:
            st.info(t("book_select_chapter", lang))


# ---------------------------------------------------------------------------
# Tab 3: Book Q&A
# ---------------------------------------------------------------------------

_QA_SYSTEM_PROMPT_ZH = """你是一位专业的书籍阅读助手，精通各类经典书籍的内容。
用户会告诉你一本书的名字，然后向你提问关于这本书的问题。

规则：
1. 根据你对该书内容的了解来回答问题。
2. 回答时引用书中相关的概念、章节或核心观点。
3. 使用清晰、有条理的中文回答。
4. 如果你不确定某个细节，请诚实说明。
5. 在回答末尾用"📌 相关章节"列出涉及的章节名称。"""

_QA_SYSTEM_PROMPT_EN = """You are a professional book reading assistant with deep knowledge of classic books.
The user will tell you a book name and ask questions about it.

Rules:
1. Answer based on your knowledge of the book's content.
2. Reference relevant concepts, chapters, or key ideas from the book.
3. Provide clear, well-structured answers.
4. If you are unsure about a detail, honestly say so.
5. End your answer with "📌 Related chapters" listing the relevant chapter names."""


def _build_qa_user_prompt(question: str, book_name: str, lang: str) -> str:
    if lang == "zh":
        return f"""书名：《{book_name}》

我的问题：{question}

请根据你对这本书的了解来回答。"""
    else:
        return f"""Book: "{book_name}"

My question: {question}

Please answer based on your knowledge of this book."""


def page_qa(lang: str, api_key: str, model: str, base_url: str, provider_name: str = "") -> None:
    """Render the Book Q&A page."""
    st.header(t("qa_title", lang))
    st.caption(t("qa_subtitle", lang))

    # ---- View a saved Q&A record ----
    if st.session_state.get("view_qa_record_id") is not None:
        db = get_db()
        qa_rec = db.get_qa_record_by_id(st.session_state.view_qa_record_id)
        if qa_rec:
            st.info(f"📚 {qa_rec.book_name}")
            st.markdown(f"**❓ {qa_rec.question}**")
            st.markdown("---")
            st.markdown(f"**{t('qa_answer_title', lang)}**")
            st.markdown(qa_rec.answer)
            st.caption(t("qa_context_note", lang, book=qa_rec.book_name))
            _do_tts(qa_rec.answer, lang, btn_key="tts_qa_saved")
            if st.button("↩️ Back / 返回", key="qa_back"):
                st.session_state.view_qa_record_id = None
                st.rerun()
            return
        else:
            st.session_state.view_qa_record_id = None

    # Book name input (default to Beyond Feelings)
    book_name = st.text_input(
        t("qa_book_name", lang),
        value="Beyond Feelings: A Guide to Critical Thinking",
        key="qa_book_name",
    )

    # Question input
    question = st.text_input(
        t("qa_input_placeholder", lang),
        key="qa_question",
    )

    # Chat history
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    if st.button(t("qa_ask_button", lang), type="primary"):
        if not api_key:
            st.warning(t("error_no_key", lang))
            return
        if not book_name or not book_name.strip():
            st.warning(t("qa_no_book_name", lang))
            return
        if not question or not question.strip():
            st.warning(t("qa_no_question", lang))
            return

        with st.spinner(t("qa_thinking", lang)):
            system_prompt = _QA_SYSTEM_PROMPT_ZH if lang == "zh" else _QA_SYSTEM_PROMPT_EN
            user_prompt = _build_qa_user_prompt(question.strip(), book_name.strip(), lang)

            provider = OpenAIProvider(
                api_key=api_key,
                model=model,
                base_url=base_url if base_url else None,
            )

            try:
                answer = provider.complete_text(system_prompt, user_prompt)
            except ProviderError as exc:
                error_code = str(exc)
                if "QUOTA_EXCEEDED" in error_code:
                    st.error(t("error_quota", lang, provider=provider_name))
                    st.caption(f"🔍 Detail: `{error_code}`")
                elif "AUTH_ERROR" in error_code:
                    st.error(t("error_auth", lang))
                elif "CONTEXT_TOO_LONG" in error_code:
                    st.error(t("error_context_too_long", lang))
                elif "RATE_LIMIT" in error_code:
                    st.warning(t("error_rate_limit", lang))
                    st.caption(f"🔍 Detail: `{error_code}`")
                else:
                    st.error(t("error_analysis", lang, error=error_code))
                return
            except Exception as exc:
                st.error(t("error_analysis", lang, error=str(exc)))
                return

        # Append to session history
        st.session_state.qa_history.append({
            "question": question.strip(),
            "answer": answer,
            "book": book_name.strip(),
        })

        # Persist to database
        db = get_db()
        db.save_qa_record(book_name.strip(), question.strip(), answer, lang)
        st.toast(t("saved", lang))

    # Render history grouped by book (newest first)
    if st.session_state.qa_history:
        _qa_by_book: OrderedDict[str, list] = OrderedDict()
        for item in reversed(st.session_state.qa_history):
            _qa_by_book.setdefault(item["book"], []).append(item)
        for _book, _items in _qa_by_book.items():
            with st.expander(f"📚 {_book} ({len(_items)})", expanded=True):
                for _idx, item in enumerate(_items):
                    st.markdown(f"**❓ {item['question']}**")
                    st.markdown(f"**{t('qa_answer_title', lang)}**")
                    st.markdown(item["answer"])
                    st.caption(t("qa_context_note", lang, book=item["book"]))
                    _do_tts(item["answer"], lang, btn_key=f"tts_qa_{_book}_{_idx}")
                    if _idx < len(_items) - 1:
                        st.divider()


# ---------------------------------------------------------------------------
# TTS shared helper
# ---------------------------------------------------------------------------

_RATE_OPTIONS = {
    "tts_rate_slow": "-30%",
    "tts_rate_normal": "+0%",
    "tts_rate_fast": "+30%",
    "tts_rate_faster": "+50%",
}


def _get_tts_settings(lang: str) -> tuple[str, str]:
    """Read current voice_id and rate from session state (set by sidebar)."""
    voice_id = st.session_state.get("_tts_voice_id", "zh-CN-XiaoxiaoNeural" if lang == "zh" else "en-US-JennyNeural")
    rate_labels = {t(k, lang): v for k, v in _RATE_OPTIONS.items()}
    rate_label = st.session_state.get("tts_rate_slider", t("tts_rate_normal", lang))
    rate = rate_labels.get(rate_label, "+0%")
    return voice_id, rate


def _do_tts(text: str, lang: str, btn_key: str) -> None:
    """Render a 🔊 read-aloud button for the given text. Plays inline."""
    if st.button(t("tts_read_this", lang), key=btn_key):
        voice_id, rate = _get_tts_settings(lang)
        with st.spinner(t("tts_generating", lang)):
            try:
                audio_bytes = synthesize(text, voice_id, rate)
            except Exception as exc:
                st.error(t("tts_error", lang, error=str(exc)))
                return
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label=t("tts_download", lang),
                data=audio_bytes,
                file_name="thinker_tts.mp3",
                mime="audio/mp3",
                key=f"{btn_key}_dl",
            )


# ---------------------------------------------------------------------------
# Tab 4: Text-to-Speech (standalone input)
# ---------------------------------------------------------------------------

def page_tts(lang: str) -> None:
    """Render the Text-to-Speech page with a free-form text input."""
    st.header(t("tts_title", lang))
    st.caption(t("tts_subtitle", lang))

    # ---- Text input ----
    tts_text = st.text_area(
        t("tts_input_placeholder", lang),
        height=200,
        key="tts_input_text",
    )

    # ---- Read aloud button ----
    col_play, col_empty = st.columns([1, 3])
    with col_play:
        play_clicked = st.button(t("tts_play_button", lang), type="primary", key="tts_main_play")

    if play_clicked:
        if not tts_text or not tts_text.strip():
            st.warning(t("tts_no_text", lang))
            return

        voice_id, rate = _get_tts_settings(lang)
        with st.spinner(t("tts_generating", lang)):
            try:
                audio_bytes = synthesize(tts_text.strip(), voice_id, rate)
            except Exception as exc:
                st.error(t("tts_error", lang, error=str(exc)))
                return

        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label=t("tts_download", lang),
                data=audio_bytes,
                file_name="thinker_tts.mp3",
                mime="audio/mp3",
                key="tts_main_dl",
            )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Thinker", page_icon="🧠", layout="wide")

    # --- Session state defaults ---
    if "lang" not in st.session_state:
        st.session_state.lang = config.DEFAULT_LANGUAGE
    if "view_record_id" not in st.session_state:
        st.session_state.view_record_id = None
    if "view_qa_record_id" not in st.session_state:
        st.session_state.view_qa_record_id = None

    # =================================================================
    # Sidebar
    # =================================================================
    with st.sidebar:
        lang_options = {"中文": "zh", "English": "en"}
        selected_lang_label = st.selectbox(
            "🌐 Language / 语言",
            options=list(lang_options.keys()),
            index=0 if st.session_state.lang == "zh" else 1,
        )
        lang = lang_options[selected_lang_label]
        st.session_state.lang = lang

        st.header(t("sidebar_title", lang))

        # ---- Provider presets ----
        _PRESETS = {
            "OpenAI": {"base_url": "", "model": "gpt-4o"},
            "DeepSeek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
            "Zhipu (智谱)": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
            "Google Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
            "Ollama (local)": {"base_url": "http://localhost:11434/v1", "model": "llama3"},
            t("custom_provider", lang): {"base_url": "", "model": ""},
        }
        _PRESET_NAMES = list(_PRESETS.keys())

        # Remember last provider to detect switch
        if "_prev_preset" not in st.session_state:
            st.session_state["_prev_preset"] = _PRESET_NAMES[0]

        preset = st.selectbox(
            t("provider_label", lang),
            options=_PRESET_NAMES,
            index=_PRESET_NAMES.index(st.session_state["_prev_preset"])
                  if st.session_state["_prev_preset"] in _PRESET_NAMES else 0,
            key="provider_select",
        )

        # On provider switch: inject new defaults into session state and rerun
        if preset != st.session_state["_prev_preset"]:
            st.session_state["_prev_preset"] = preset
            preset_cfg = _PRESETS[preset]
            db = get_db()
            saved_key = db.get_setting(f"apikey:{preset}", default="")
            # Directly set widget session-state keys so they render with new values
            st.session_state["sidebar_api_key"] = saved_key or config.OPENAI_API_KEY
            st.session_state["sidebar_model"] = preset_cfg["model"] or config.OPENAI_MODEL
            st.session_state["sidebar_base_url"] = preset_cfg["base_url"] or config.OPENAI_BASE_URL or ""
            st.rerun()

        preset_cfg = _PRESETS[preset]

        # Load saved API key for this provider
        db = get_db()
        _saved_key = db.get_setting(f"apikey:{preset}", default="")
        _default_key = _saved_key or config.OPENAI_API_KEY

        api_key = st.text_input(
            t("api_key_label", lang),
            value=_default_key,
            type="password",
            help=t("api_key_help", lang),
            key="sidebar_api_key",
        )

        # Auto-save API key when changed
        if api_key and api_key != _saved_key:
            db.set_setting(f"apikey:{preset}", api_key)

        model = st.text_input(
            t("model_label", lang),
            value=preset_cfg["model"] or config.OPENAI_MODEL,
            key="sidebar_model",
        )
        base_url = st.text_input(
            t("base_url_label", lang),
            value=preset_cfg["base_url"] or config.OPENAI_BASE_URL or "",
            key="sidebar_base_url",
        )

        # Show actual provider config being used
        _key_preview = f"{api_key[:8]}…" if len(api_key) > 8 else (api_key or "—")
        st.caption(f"🔧 {preset} | `{model}` | key: `{_key_preview}`")

        st.divider()

        # ---- TTS Voice Settings ----
        with st.expander(t("tts_sidebar_title", lang), expanded=False):
            # Voice language filter
            _voice_lang_opts = {
                t("tts_voice_lang_zh", lang): "zh",
                t("tts_voice_lang_en", lang): "en",
                t("tts_voice_lang_all", lang): "all",
            }
            _vl_default = 0 if lang == "zh" else 1
            _sel_vl_label = st.radio(
                t("tts_voice_lang", lang),
                options=list(_voice_lang_opts.keys()),
                index=_vl_default,
                horizontal=True,
                key="tts_voice_lang_radio",
            )
            _voice_content_lang = _voice_lang_opts[_sel_vl_label]

            # Voice selector
            _voice_opts = get_voice_options(_voice_content_lang, lang)
            _sel_voice_label = st.selectbox(
                t("tts_voice_label", lang),
                options=list(_voice_opts.keys()),
                key="tts_voice_select",
            )
            # Store selected voice_id in session state for other pages to read
            st.session_state["_tts_voice_id"] = _voice_opts[_sel_voice_label]

            # Speed slider
            _rate_labels = {t(k, lang): v for k, v in _RATE_OPTIONS.items()}
            st.select_slider(
                t("tts_rate_label", lang),
                options=list(_rate_labels.keys()),
                value=t("tts_rate_normal", lang),
                key="tts_rate_slider",
            )

        st.divider()

        # ---- Analysis History ----
        st.subheader(t("history_title", lang))
        db = get_db()
        records = db.get_all_records(limit=30)
        if not records:
            st.caption(t("history_empty", lang))
        else:
            with st.expander(f"📜 {t('history_title', lang)} ({len(records)})", expanded=False):
                for rec in records:
                    label = rec.input_text[:40] + ("…" if len(rec.input_text) > 40 else "")
                    col_btn, col_del = st.columns([4, 1])
                    with col_btn:
                        if st.button(f"📄 {label}", key=f"view_{rec.id}"):
                            st.session_state.view_record_id = rec.id
                    with col_del:
                        if st.button("🗑️", key=f"del_{rec.id}"):
                            db.delete_record(rec.id)  # type: ignore[arg-type]
                            st.rerun()

        # ---- Q&A History ----
        st.subheader(t("qa_history_title", lang))
        qa_records = db.get_all_qa_records(limit=30)
        if not qa_records:
            st.caption(t("history_empty", lang))
        else:
            # Group by book name
            qa_by_book: OrderedDict[str, list] = OrderedDict()
            for qa_rec in qa_records:
                qa_by_book.setdefault(qa_rec.book_name, []).append(qa_rec)
            for book_name, book_recs in qa_by_book.items():
                with st.expander(f"📚 {book_name} ({len(book_recs)})", expanded=False):
                    for qa_rec in book_recs:
                        qa_label = qa_rec.question[:35] + ("…" if len(qa_rec.question) > 35 else "")
                        col_qa_btn, col_qa_del = st.columns([4, 1])
                        with col_qa_btn:
                            if st.button(f"💬 {qa_label}", key=f"view_qa_{qa_rec.id}"):
                                st.session_state.view_qa_record_id = qa_rec.id
                        with col_qa_del:
                            if st.button("🗑️", key=f"del_qa_{qa_rec.id}"):
                                db.delete_qa_record(qa_rec.id)  # type: ignore[arg-type]
                                st.rerun()


    # =================================================================
    # Main area – Tabs
    # =================================================================
    st.title(t("app_title", lang))
    st.caption(t("app_subtitle", lang))

    tab_analyze, tab_read, tab_qa, tab_tts = st.tabs([
        t("tab_analyze", lang),
        t("tab_read", lang),
        t("tab_qa", lang),
        t("tab_tts", lang),
    ])

    with tab_analyze:
        page_analyze(lang, api_key, model, base_url, provider_name=preset)

    with tab_read:
        page_read_book(lang)

    with tab_qa:
        page_qa(lang, api_key, model, base_url, provider_name=preset)

    with tab_tts:
        page_tts(lang)


if __name__ == "__main__":
    main()

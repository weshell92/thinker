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
from analyzer.providers.gemini_native_provider import GeminiNativeProvider  # noqa: E402
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


def _make_provider(api_key: str, model: str, base_url: str, is_native_gemini: bool = False):
    """Create the appropriate LLM provider instance.

    When ``is_native_gemini`` is True, returns a :class:`GeminiNativeProvider`
    that calls the Gemini REST API directly.  Otherwise returns an
    :class:`OpenAIProvider` (works with all OpenAI-compatible endpoints).
    """
    if is_native_gemini:
        return GeminiNativeProvider(
            api_keys=api_key,
            model=model,
            base_url=base_url if base_url else "https://generativelanguage.googleapis.com",
        )
    return OpenAIProvider(
        api_key=api_key,
        model=model,
        base_url=base_url if base_url else None,
    )



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

def page_analyze(lang: str, api_key: str, model: str, base_url: str, provider_name: str = "", is_native_gemini: bool = False) -> None:
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

        provider = _make_provider(api_key, model, base_url, is_native_gemini)
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
                elif "TIMEOUT" in error_code:
                    st.warning(t("error_timeout", lang))
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
        db.save_record(user_text.strip(), lang, result,
                       provider_name=provider_name, model_name=model)
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


def page_qa(lang: str, api_key: str, model: str, base_url: str, provider_name: str = "", is_native_gemini: bool = False) -> None:
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

            provider = _make_provider(api_key, model, base_url, is_native_gemini)

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
                elif "TIMEOUT" in error_code:
                    st.warning(t("error_timeout", lang))
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
        db.save_qa_record(book_name.strip(), question.strip(), answer, lang,
                          provider_name=provider_name, model_name=model)
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
# Tab 4: Free Chat
# ---------------------------------------------------------------------------

_CHAT_SYSTEM_PROMPT_ZH = """你是一位知识渊博且乐于助人的 AI 助手。
请用清晰、准确、有条理的中文回答用户的任何问题。
如果用户上传了文件，请结合文件内容进行回答。
如果你不确定某个答案，请诚实说明。"""

_CHAT_SYSTEM_PROMPT_EN = """You are a knowledgeable and helpful AI assistant.
Answer the user's questions clearly, accurately, and in a well-structured manner.
If the user uploads a file, incorporate the file content in your answer.
If you are unsure about something, honestly say so."""

# Supported file extensions for upload
_CHAT_UPLOAD_TYPES = ["txt", "md", "csv", "json", "py", "pdf", "docx", "log", "xml", "html", "htm", "yaml", "yml", "toml", "ini", "cfg", "rst", "tex",
                      "png", "jpg", "jpeg", "gif", "webp", "bmp"]

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}

_CHAT_MAX_FILE_MB = 10  # maximum upload size in megabytes


def _is_image_file(filename: str) -> bool:
    """Return True if the filename has an image extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in _IMAGE_EXTENSIONS


def _encode_image_base64(uploaded_file) -> str:
    """Return a base64-encoded data URI for an uploaded image."""
    import base64
    raw_bytes: bytes = uploaded_file.getvalue()
    name: str = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else "png"
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp", "bmp": "bmp"}
    mime_type = f"image/{mime_map.get(ext, 'png')}"
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _encode_pil_image_base64(pil_image) -> str:
    """Return a base64-encoded data URI for a PIL Image (e.g. from clipboard paste)."""
    import base64
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_file_text(uploaded_file) -> str:
    """Extract text content from an uploaded file.

    Supports plain-text formats, PDF (via PyMuPDF) and DOCX (via python-docx).
    Returns the extracted text string or raises ValueError on failure.
    """
    name: str = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    raw_bytes: bytes = uploaded_file.getvalue()

    # --- PDF ---
    if ext == "pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages_text: list[str] = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                pages_text.append(page_text)
        doc.close()
        return "\n\n".join(pages_text)

    # --- DOCX ---
    if ext == "docx":
        try:
            import io
            from docx import Document
        except ImportError:
            raise ValueError(
                "读取 DOCX 文件需要安装 python-docx，请运行：pip install python-docx"
            )
        doc = Document(io.BytesIO(raw_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # --- Everything else: treat as plain text ---
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("Unable to decode file as text")


def _human_file_size(n_bytes: int) -> str:
    """Return a human-readable file size string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    else:
        return f"{n_bytes / (1024 * 1024):.1f} MB"


def page_chat(lang: str, api_key: str, model: str, base_url: str, provider_name: str = "", is_native_gemini: bool = False) -> None:
    """Render the Free Chat page with multi-turn conversation."""
    st.header(t("chat_title", lang))
    st.caption(t("chat_subtitle", lang))

    # ---- View a saved chat record ----
    if st.session_state.get("view_chat_record_id") is not None:
        db = get_db()
        chat_rec = db.get_chat_record_by_id(st.session_state.view_chat_record_id)
        if chat_rec:
            st.markdown(f"**❓ {chat_rec.question}**")
            st.markdown("---")
            st.markdown(f"**{t('chat_answer_title', lang)}**")
            st.markdown(chat_rec.answer)
            _do_tts(chat_rec.answer, lang, btn_key="tts_chat_saved")
            if st.button("↩️ Back / 返回", key="chat_back"):
                st.session_state.view_chat_record_id = None
                st.rerun()
            return
        else:
            st.session_state.view_chat_record_id = None

    # ---- File uploader (multi-file) ----
    uploaded_files = st.file_uploader(
        t("chat_upload_label", lang),
        type=_CHAT_UPLOAD_TYPES,
        help=t("chat_upload_help", lang),
        key="chat_file_uploader",
        accept_multiple_files=True,
    )

    # Process uploaded files
    all_file_texts: list[str] = []
    all_image_data_uris: list[tuple[str, str]] = []  # (filename, data_uri)
    for uploaded_file in (uploaded_files or []):
        file_size = len(uploaded_file.getvalue())
        if file_size > _CHAT_MAX_FILE_MB * 1024 * 1024:
            st.warning(t("chat_file_too_large", lang, max_size=str(_CHAT_MAX_FILE_MB)))
            continue
        if _is_image_file(uploaded_file.name):
            data_uri = _encode_image_base64(uploaded_file)
            all_image_data_uris.append((uploaded_file.name, data_uri))
            st.success(t("chat_file_loaded", lang, name=uploaded_file.name, size=_human_file_size(file_size)))
            st.image(uploaded_file, caption=uploaded_file.name, width=300)
        else:
            try:
                text = _extract_file_text(uploaded_file)
                if text and text.strip():
                    file_prefix = t("chat_file_context_prefix", lang, name=uploaded_file.name)
                    all_file_texts.append(file_prefix + text)
                    st.success(t("chat_file_loaded", lang, name=uploaded_file.name, size=_human_file_size(file_size)))
                else:
                    st.warning(t("chat_file_empty", lang))
            except Exception as exc:
                st.error(t("chat_file_extract_error", lang, error=str(exc)))

    # ---- Initialize conversation history ----
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # ---- Render existing conversation ----
    for msg in st.session_state.chat_messages:
        role_label = t("chat_you", lang) if msg["role"] == "user" else t("chat_ai", lang)
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["display"] if "display" in msg else msg["content"])

    # ---- Chat input (multi-line + clipboard paste) ----
    from streamlit_paste_button import paste_image_button as _paste_btn

    user_input = st.text_area(
        t("chat_input_placeholder", lang),
        height=120,
        key="chat_text_area",
        placeholder=t("chat_input_placeholder", lang),
    )
    st.caption(t("chat_input_hint", lang))

    # ---- Multi-image clipboard paste (accumulated in session state) ----
    if "chat_pasted_images" not in st.session_state:
        st.session_state.chat_pasted_images = []  # list of (name, data_uri)
    if "chat_last_paste_hash" not in st.session_state:
        st.session_state.chat_last_paste_hash = None

    _paste_col, _send_col = st.columns([1, 1])
    with _paste_col:
        _paste_result = _paste_btn(
            label=t("paste_image_button", lang),
            key="chat_paste_image",
        )
    # When a new image is pasted, append it — but only if it's genuinely new
    if _paste_result and _paste_result.image_data:
        import hashlib as _hl
        _pasted_data_uri = _encode_pil_image_base64(_paste_result.image_data)
        _paste_hash = _hl.md5(_pasted_data_uri.encode()).hexdigest()
        if _paste_hash != st.session_state.chat_last_paste_hash:
            st.session_state.chat_last_paste_hash = _paste_hash
            _n = len(st.session_state.chat_pasted_images) + 1
            _paste_name = f"clipboard_paste_{_n}.png"
            st.session_state.chat_pasted_images.append((_paste_name, _pasted_data_uri))
            st.toast(t("paste_image_loaded", lang, n=str(_n)))

    # Display all pasted images with delete buttons
    if st.session_state.chat_pasted_images:
        st.info(t("paste_image_count", lang, n=str(len(st.session_state.chat_pasted_images))))
        _del_indices: list[int] = []
        for _i, (_pname, _puri) in enumerate(st.session_state.chat_pasted_images):
            _img_col, _del_col = st.columns([4, 1])
            with _img_col:
                # Decode base64 for display
                import base64 as _b64
                _raw = _b64.b64decode(_puri.split(",", 1)[1])
                st.image(_raw, caption=_pname, width=200)
            with _del_col:
                if st.button(t("paste_image_delete", lang), key=f"chat_paste_del_{_i}"):
                    _del_indices.append(_i)
        if _del_indices:
            st.session_state.chat_pasted_images = [
                v for idx, v in enumerate(st.session_state.chat_pasted_images) if idx not in _del_indices
            ]
            st.rerun()
        # Clear all button
        if len(st.session_state.chat_pasted_images) > 1:
            if st.button(t("paste_image_clear_all", lang), key="chat_paste_clear_all"):
                st.session_state.chat_pasted_images = []
                st.rerun()

    # Merge pasted images into the image list for LLM
    for _pname, _puri in st.session_state.chat_pasted_images:
        all_image_data_uris.append((_pname, _puri))

    with _send_col:
        _chat_send_clicked = st.button(t("chat_send_button", lang), key="chat_send_btn", type="primary", use_container_width=True)

    if _chat_send_clicked and user_input:
        if not api_key:
            st.warning(t("error_no_key", lang))
            return
        if not user_input.strip():
            st.warning(t("chat_no_question", lang))
            return

        # Build the actual content sent to LLM (may include file text or image)
        display_text = user_input.strip()
        content_for_llm = user_input.strip()
        has_vision = False

        if all_image_data_uris:
            # Vision mode: build multimodal content block with all images
            has_vision = True
            parts: list[dict] = []
            display_names: list[str] = []
            for fname, data_uri in all_image_data_uris:
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                display_names.append(fname)
            # Also prepend text files if any
            text_block = user_input.strip()
            if all_file_texts:
                text_block = "\n\n---\n\n".join(all_file_texts) + "\n\n---\n\n" + text_block
            parts.append({"type": "text", "text": text_block})
            content_for_llm = parts
            display_text = "🖼️ *" + ", ".join(display_names) + "*\n\n" + user_input.strip()
        elif all_file_texts:
            combined = "\n\n---\n\n".join(all_file_texts) + "\n\n---\n\n" + user_input.strip()
            content_for_llm = combined
            fnames = ", ".join(f.name for f in uploaded_files if not _is_image_file(f.name))
            display_text = f"📎 *{fnames}*\n\n{user_input.strip()}"

        # Append user message (store both display and full content)
        st.session_state.chat_messages.append({
            "role": "user",
            "content": content_for_llm,
            "display": display_text,
        })
        with st.chat_message("user", avatar="🧑"):
            st.markdown(display_text)

        # Build messages for API call
        system_prompt = _CHAT_SYSTEM_PROMPT_ZH if lang == "zh" else _CHAT_SYSTEM_PROMPT_EN
        api_messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.chat_messages:
            api_messages.append({"role": m["role"], "content": m["content"]})

        provider = _make_provider(api_key, model, base_url, is_native_gemini)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(t("chat_thinking", lang)):
                try:
                    if has_vision:
                        answer = provider.complete_chat_with_vision(api_messages)
                    else:
                        answer = provider.complete_chat(api_messages)
                except ProviderError as exc:
                    error_code = str(exc)
                    if "QUOTA_EXCEEDED" in error_code:
                        st.error(t("error_quota", lang, provider=provider_name))
                    elif "AUTH_ERROR" in error_code:
                        st.error(t("error_auth", lang))
                    elif "CONTEXT_TOO_LONG" in error_code:
                        st.error(t("error_context_too_long", lang))
                    elif "TIMEOUT" in error_code:
                        st.warning(t("error_timeout", lang))
                    elif "RATE_LIMIT" in error_code:
                        st.warning(t("error_rate_limit", lang))
                    else:
                        st.error(t("error_analysis", lang, error=error_code))
                    return
                except Exception as exc:
                    st.error(t("error_analysis", lang, error=str(exc)))
                    return

            st.markdown(answer)

        # Append assistant message
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        # Persist to database
        db = get_db()
        db.save_chat_record(display_text, answer, lang,
                            provider_name=provider_name, model_name=model)
        st.toast(t("saved", lang))
        # Clear pasted images after successful send
        st.session_state.chat_pasted_images = []
        st.session_state.chat_last_paste_hash = None

    # ---- Clear conversation button ----
    if st.session_state.chat_messages:
        if st.button(t("chat_clear_button", lang), key="chat_clear"):
            st.session_state.chat_messages = []
            st.toast(t("chat_cleared", lang))
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 5: Gateway Q&A
# ---------------------------------------------------------------------------

# Load expr.md documentation at module level (once)
_EXPR_MD_PATH = os.path.join(_PROJECT_ROOT, "expr.md")
_EXPR_MD_CONTENT: str = ""
_EXPR_MD_CAUTION: str = ""
if os.path.isfile(_EXPR_MD_PATH):
    with open(_EXPR_MD_PATH, "r", encoding="utf-8") as _f:
        _EXPR_MD_CONTENT = _f.read()
    # Extract "函数使用注意说明" section
    _caution_marker = "**函数使用注意说明**"
    _caution_idx = _EXPR_MD_CONTENT.find(_caution_marker)
    if _caution_idx != -1:
        # Grab from the marker to the next major section heading (** or ###)
        _caution_body = _EXPR_MD_CONTENT[_caution_idx + len(_caution_marker):]
        # Find the end: next "**" bold heading or "###" section heading
        for _end_marker in ("\n**", "\n###", "\n---"):
            _end_idx = _caution_body.find(_end_marker)
            if _end_idx != -1:
                _caution_body = _caution_body[:_end_idx]
                break
        _EXPR_MD_CAUTION = _caution_body.strip()

_GATEWAY_SYSTEM_PROMPT_ZH = """你是一位专业的网关表达式配置助手，精通 Aviator 表达式引擎及其在网关系统中的应用。

以下是完整的 Aviator 表达式引擎文档，你必须严格依据这份文档来回答用户的问题：

---
{doc}
---

⚠️ 注意事项（函数使用注意说明）：
{caution}

规则：
1. 只根据上述文档内容回答问题，不要编造文档中不存在的函数或功能。
2. 回答时给出准确的函数名、参数说明和调用示例。
3. 如果用户问的函数在文档中有多个版本（如 V2、V3），请说明区别。
4. 使用清晰、有条理的中文回答。
5. 如果文档中没有相关内容，请诚实说明。
6. 特别注意"注意事项"中的说明，涉及相关函数时必须提醒用户注意。
7. 在回答末尾用"📌 相关章节"列出涉及的文档章节编号。"""

_GATEWAY_SYSTEM_PROMPT_EN = """You are a professional gateway expression configuration assistant, expert in the Aviator expression engine and its use in gateway systems.

Below is the complete Aviator expression engine documentation. You must answer strictly based on this document:

---
{doc}
---

⚠️ Cautions (Function Usage Notes):
{caution}

Rules:
1. Only answer based on the above documentation. Do not fabricate functions or features not in the document.
2. Provide accurate function names, parameter descriptions, and call examples.
3. If there are multiple versions of a function (e.g. V2, V3), explain the differences.
4. Provide clear, well-structured answers.
5. If the documentation does not cover the topic, honestly say so.
6. Pay special attention to the "Cautions" section above; always warn users about relevant notes when applicable.
7. End your answer with "📌 Related sections" listing the relevant documentation section numbers."""


def page_gateway(lang: str, api_key: str, model: str, base_url: str, provider_name: str = "", is_native_gemini: bool = False) -> None:
    """Render the Gateway Expression Q&A page with multi-turn conversation."""
    st.header(t("gateway_title", lang))
    st.caption(t("gateway_subtitle", lang))

    if not _EXPR_MD_CONTENT:
        st.warning("⚠️ expr.md not found in project root.")
        return

    # ---- View a saved gateway record ----
    if st.session_state.get("view_gateway_qa_record_id") is not None:
        db = get_db()
        gw_rec = db.get_gateway_qa_record_by_id(st.session_state.view_gateway_qa_record_id)
        if gw_rec:
            st.markdown(f"**❓ {gw_rec.question}**")
            st.markdown("---")
            st.markdown(f"**{t('gateway_answer_title', lang)}**")
            st.markdown(gw_rec.answer)
            st.caption(t("gateway_context_note", lang))
            if gw_rec.provider_name or gw_rec.model_name:
                st.caption(t("history_provider_model", lang,
                             provider=gw_rec.provider_name, model=gw_rec.model_name))
            _do_tts(gw_rec.answer, lang, btn_key="tts_gw_saved")
            if st.button("↩️ Back / 返回", key="gw_back"):
                st.session_state.view_gateway_qa_record_id = None
                st.rerun()
            return
        else:
            st.session_state.view_gateway_qa_record_id = None

    # ---- File / Image uploader (multi-file) ----
    gw_uploaded_files = st.file_uploader(
        t("gateway_upload_label", lang),
        type=_CHAT_UPLOAD_TYPES,
        help=t("gateway_upload_help", lang),
        key="gateway_file_uploader",
        accept_multiple_files=True,
    )

    # Process uploaded files
    gw_file_texts: list[str] = []
    gw_image_data_uris: list[tuple[str, str]] = []  # (filename, data_uri)
    for gw_uploaded_file in (gw_uploaded_files or []):
        gw_file_size = len(gw_uploaded_file.getvalue())
        if gw_file_size > _CHAT_MAX_FILE_MB * 1024 * 1024:
            st.warning(t("chat_file_too_large", lang, max_size=str(_CHAT_MAX_FILE_MB)))
            continue
        if _is_image_file(gw_uploaded_file.name):
            data_uri = _encode_image_base64(gw_uploaded_file)
            gw_image_data_uris.append((gw_uploaded_file.name, data_uri))
            st.success(t("chat_file_loaded", lang, name=gw_uploaded_file.name, size=_human_file_size(gw_file_size)))
            st.image(gw_uploaded_file, caption=gw_uploaded_file.name, width=300)
        else:
            try:
                text = _extract_file_text(gw_uploaded_file)
                if text and text.strip():
                    file_prefix = t("chat_file_context_prefix", lang, name=gw_uploaded_file.name)
                    gw_file_texts.append(file_prefix + text)
                    st.success(t("chat_file_loaded", lang, name=gw_uploaded_file.name, size=_human_file_size(gw_file_size)))
                else:
                    st.warning(t("chat_file_empty", lang))
            except Exception as exc:
                st.error(t("chat_file_extract_error", lang, error=str(exc)))

    # ---- Initialize conversation history ----
    if "gateway_messages" not in st.session_state:
        st.session_state.gateway_messages = []

    # ---- Render existing conversation ----
    for msg in st.session_state.gateway_messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["display"] if "display" in msg else msg["content"])

    # ---- Chat input (multi-line + clipboard paste) ----
    from streamlit_paste_button import paste_image_button as _paste_btn

    user_input = st.text_area(
        t("gateway_input_placeholder", lang),
        height=120,
        key="gateway_text_area",
        placeholder=t("gateway_input_placeholder", lang),
    )
    st.caption(t("chat_input_hint", lang))

    # ---- Multi-image clipboard paste (accumulated in session state) ----
    if "gw_pasted_images" not in st.session_state:
        st.session_state.gw_pasted_images = []  # list of (name, data_uri)
    if "gw_last_paste_hash" not in st.session_state:
        st.session_state.gw_last_paste_hash = None

    _gw_paste_col, _gw_send_col = st.columns([1, 1])
    with _gw_paste_col:
        _gw_paste_result = _paste_btn(
            label=t("paste_image_button", lang),
            key="gateway_paste_image",
        )
    # When a new image is pasted, append it — but only if it's genuinely new
    if _gw_paste_result and _gw_paste_result.image_data:
        import hashlib as _hl
        _gw_pasted_data_uri = _encode_pil_image_base64(_gw_paste_result.image_data)
        _gw_paste_hash = _hl.md5(_gw_pasted_data_uri.encode()).hexdigest()
        if _gw_paste_hash != st.session_state.gw_last_paste_hash:
            st.session_state.gw_last_paste_hash = _gw_paste_hash
            _gw_n = len(st.session_state.gw_pasted_images) + 1
            _gw_paste_name = f"clipboard_paste_{_gw_n}.png"
            st.session_state.gw_pasted_images.append((_gw_paste_name, _gw_pasted_data_uri))
            st.toast(t("paste_image_loaded", lang, n=str(_gw_n)))

    # Display all pasted images with delete buttons
    if st.session_state.gw_pasted_images:
        st.info(t("paste_image_count", lang, n=str(len(st.session_state.gw_pasted_images))))
        _gw_del_indices: list[int] = []
        for _gi, (_gpname, _gpuri) in enumerate(st.session_state.gw_pasted_images):
            _gimg_col, _gdel_col = st.columns([4, 1])
            with _gimg_col:
                import base64 as _b64
                _graw = _b64.b64decode(_gpuri.split(",", 1)[1])
                st.image(_graw, caption=_gpname, width=200)
            with _gdel_col:
                if st.button(t("paste_image_delete", lang), key=f"gw_paste_del_{_gi}"):
                    _gw_del_indices.append(_gi)
        if _gw_del_indices:
            st.session_state.gw_pasted_images = [
                v for idx, v in enumerate(st.session_state.gw_pasted_images) if idx not in _gw_del_indices
            ]
            st.rerun()
        # Clear all button
        if len(st.session_state.gw_pasted_images) > 1:
            if st.button(t("paste_image_clear_all", lang), key="gw_paste_clear_all"):
                st.session_state.gw_pasted_images = []
                st.rerun()

    # Merge pasted images into the image list for LLM
    for _gpname, _gpuri in st.session_state.gw_pasted_images:
        gw_image_data_uris.append((_gpname, _gpuri))

    with _gw_send_col:
        _gw_send_clicked = st.button(t("gateway_ask_button", lang), key="gateway_send_btn", type="primary", use_container_width=True)

    if _gw_send_clicked and user_input:
        if not api_key:
            st.warning(t("error_no_key", lang))
            return
        if not user_input.strip():
            st.warning(t("gateway_no_question", lang))
            return

        # Build the actual content sent to LLM (may include file text or image)
        display_text = user_input.strip()
        content_for_llm = user_input.strip()
        gw_has_vision = False

        if gw_image_data_uris:
            gw_has_vision = True
            parts: list[dict] = []
            display_names: list[str] = []
            for fname, data_uri in gw_image_data_uris:
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                display_names.append(fname)
            text_block = user_input.strip()
            if gw_file_texts:
                text_block = "\n\n---\n\n".join(gw_file_texts) + "\n\n---\n\n" + text_block
            parts.append({"type": "text", "text": text_block})
            content_for_llm = parts
            display_text = "🖼️ *" + ", ".join(display_names) + "*\n\n" + user_input.strip()
        elif gw_file_texts:
            combined = "\n\n---\n\n".join(gw_file_texts) + "\n\n---\n\n" + user_input.strip()
            content_for_llm = combined
            fnames = ", ".join(f.name for f in gw_uploaded_files if not _is_image_file(f.name))
            display_text = f"📎 *{fnames}*\n\n{user_input.strip()}"

        # Append user message
        st.session_state.gateway_messages.append({
            "role": "user",
            "content": content_for_llm,
            "display": display_text,
        })
        with st.chat_message("user", avatar="🧑"):
            st.markdown(display_text)

        # Build messages for API call
        system_tmpl = _GATEWAY_SYSTEM_PROMPT_ZH if lang == "zh" else _GATEWAY_SYSTEM_PROMPT_EN
        system_prompt = system_tmpl.format(doc=_EXPR_MD_CONTENT, caution=_EXPR_MD_CAUTION or "无")
        api_messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.gateway_messages:
            api_messages.append({"role": m["role"], "content": m["content"]})

        provider = _make_provider(api_key, model, base_url, is_native_gemini)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(t("gateway_thinking", lang)):
                try:
                    if gw_has_vision:
                        answer = provider.complete_chat_with_vision(api_messages)
                    else:
                        answer = provider.complete_chat(api_messages)
                except ProviderError as exc:
                    error_code = str(exc)
                    if "QUOTA_EXCEEDED" in error_code:
                        st.error(t("error_quota", lang, provider=provider_name))
                    elif "AUTH_ERROR" in error_code:
                        st.error(t("error_auth", lang))
                    elif "CONTEXT_TOO_LONG" in error_code:
                        st.error(t("error_context_too_long", lang))
                    elif "TIMEOUT" in error_code:
                        st.warning(t("error_timeout", lang))
                    elif "RATE_LIMIT" in error_code:
                        st.warning(t("error_rate_limit", lang))
                    else:
                        st.error(t("error_analysis", lang, error=error_code))
                    return
                except Exception as exc:
                    st.error(t("error_analysis", lang, error=str(exc)))
                    return

            st.markdown(answer)

        # Append assistant message
        st.session_state.gateway_messages.append({"role": "assistant", "content": answer})

        # Persist to database
        db = get_db()
        db.save_gateway_qa_record(
            display_text, answer, lang,
            provider_name=provider_name, model_name=model,
        )
        st.toast(t("saved", lang))
        # Clear pasted images after successful send
        st.session_state.gw_pasted_images = []
        st.session_state.gw_last_paste_hash = None

    # ---- Clear conversation button ----
    if st.session_state.gateway_messages:
        if st.button(t("gateway_clear_button", lang), key="gw_clear"):
            st.session_state.gateway_messages = []
            st.toast(t("gateway_cleared", lang))
            st.rerun()


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
# Tab 6: Text-to-Speech (standalone input)
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
    if "view_chat_record_id" not in st.session_state:
        st.session_state.view_chat_record_id = None
    if "view_gateway_qa_record_id" not in st.session_state:
        st.session_state.view_gateway_qa_record_id = None

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
            "OpenAI": {
                "base_url": "",
                "models": ["gpt-5.4","gpt-5.3","gpt-5.2","gpt-5.1","gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            },
            "DeepSeek": {
                "base_url": "https://api.deepseek.com/v1",
                "models": ["deepseek-chat", "deepseek-reasoner"],
            },
            "Zhipu (智谱)": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "models": ["glm-4-flash", "GLM-4-Plus", "GLM-4", "GLM-4-Air", "GLM-4-Long", "GLM-4-FlashX", "GLM-4-AirX"],
            },
            "Google Gemini": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "models": ["gemini-pro-latest","gemini-flash-latest","gemini-3.1-pro-preview","gemini-3.1-flash-lite","gemini-3-flash-live","gemini-3-flash-preview", "gemini-2.5-pro","gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite",  "gemma-4-31b-it", "gemini-embedding-2-preview"],
            },
            "Gemini中转 (Native)": {
                "base_url": "https://gemini-balance-lite-dxsq9gzm0cf3.weshell92.deno.net",
                "models": ["gemini-pro-latest","gemini-flash-latest","gemini-3.1-pro-preview","gemini-3.1-flash-lite","gemini-3-flash-live","gemini-3-flash-preview","gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro", "gemma-4-31b-it", "gemini-embedding-2-preview"],
                "native_gemini": True,
            },
            "Kimi (月之暗面)": {
                "base_url": "https://api.moonshot.cn/v1",
                "models": ["kimi-k2.5","kimi-k2-thinking","kimi-k2-0905","kimi-k2", "moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"],
            },
            "Qwen (通义千问)": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "models": ["qwen3.6-plus","qwen-max", "qwen-plus", "qwen-turbo", "qwen-long", "qwen-vl-max", "qwen-vl-plus", "qwen3-235b-a22b", "qwen3-32b", "qwen3-14b", "qwen3-8b", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct"],
            },
            "Groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "models": ["mixtral-8x7b-32768","llama3-70b-8192", "openai/gpt-oss-120b","llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview", "gemma2-9b-it", "qwen-qwq-32b"],
            },
            "Ollama (local)": {
                "base_url": "http://localhost:11434/v1",
                "models": ["llama3", "llama3.1", "llama3.2", "qwen2.5", "deepseek-r1", "mistral", "phi3"],
            },
            t("custom_provider", lang): {
                "base_url": "",
                "models": [],
            },
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
            st.session_state["sidebar_base_url"] = preset_cfg["base_url"] or config.OPENAI_BASE_URL or ""
            # Force model reset: set to first model of new provider (or clear)
            new_models = preset_cfg["models"]
            if new_models:
                st.session_state["sidebar_model"] = new_models[0]
            else:
                # Custom provider – no preset models, remove selectbox state
                st.session_state.pop("sidebar_model", None)
            st.session_state["sidebar_model_override"] = ""
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

        # Model selector: selectbox for presets + text input for override/custom
        _preset_models = preset_cfg["models"]
        if _preset_models:
            _selected_model = st.selectbox(
                t("model_select_label", lang),
                options=_preset_models,
                index=0,
                key="sidebar_model",
            )
        else:
            _selected_model = ""

        _model_override = st.text_input(
            t("model_custom_input_label", lang),
            value="",
            help=t("model_custom_input_help", lang),
            key="sidebar_model_override",
        )

        # Final model: override wins if non-empty, else selectbox, else config default
        model = _model_override.strip() if _model_override.strip() else (_selected_model or config.OPENAI_MODEL)

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

        # ---- Tab Visibility Settings ----
        _ALL_TABS = {
            "tab_analyze": t("tab_analyze", lang),
            "tab_read": t("tab_read", lang),
            "tab_qa": t("tab_qa", lang),
            "tab_gateway": t("tab_gateway", lang),
            "tab_chat": t("tab_chat", lang),
            "tab_tts": t("tab_tts", lang),
        }
        with st.expander(t("tab_visibility_label", lang), expanded=False):
            st.caption(t("tab_visibility_help", lang))
            _visible_tabs: list[str] = []
            for _tab_key, _tab_label in _ALL_TABS.items():
                _default_on = True
                if st.checkbox(_tab_label, value=_default_on, key=f"_vis_{_tab_key}"):
                    _visible_tabs.append(_tab_key)
        # Store for later use in tab rendering
        st.session_state["_visible_tabs"] = _visible_tabs if _visible_tabs else list(_ALL_TABS.keys())

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
                        _pm = f" [{rec.provider_name}/{rec.model_name}]" if rec.model_name else ""
                        if st.button(f"📄 {label}{_pm}", key=f"view_{rec.id}"):
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
                            _pm = f" [{qa_rec.provider_name}/{qa_rec.model_name}]" if qa_rec.model_name else ""
                            if st.button(f"💬 {qa_label}{_pm}", key=f"view_qa_{qa_rec.id}"):
                                st.session_state.view_qa_record_id = qa_rec.id
                        with col_qa_del:
                            if st.button("🗑️", key=f"del_qa_{qa_rec.id}"):
                                db.delete_qa_record(qa_rec.id)  # type: ignore[arg-type]
                                st.rerun()

        # ---- Free Chat History ----
        st.subheader(t("chat_history_title", lang))
        chat_records = db.get_all_chat_records(limit=30)
        if not chat_records:
            st.caption(t("history_empty", lang))
        else:
            with st.expander(f"💬 {t('chat_history_title', lang)} ({len(chat_records)})", expanded=False):
                for chat_rec in chat_records:
                    chat_label = chat_rec.question[:35] + ("…" if len(chat_rec.question) > 35 else "")
                    col_chat_btn, col_chat_del = st.columns([4, 1])
                    with col_chat_btn:
                        _pm = f" [{chat_rec.provider_name}/{chat_rec.model_name}]" if chat_rec.model_name else ""
                        if st.button(f"💬 {chat_label}{_pm}", key=f"view_chat_{chat_rec.id}"):
                            st.session_state.view_chat_record_id = chat_rec.id
                    with col_chat_del:
                        if st.button("🗑️", key=f"del_chat_{chat_rec.id}"):
                            db.delete_chat_record(chat_rec.id)  # type: ignore[arg-type]
                            st.rerun()

        # ---- Gateway Q&A History ----
        st.subheader(t("gateway_history_title", lang))
        gw_records = db.get_all_gateway_qa_records(limit=30)
        if not gw_records:
            st.caption(t("history_empty", lang))
        else:
            with st.expander(f"🔌 {t('gateway_history_title', lang)} ({len(gw_records)})", expanded=False):
                for gw_rec in gw_records:
                    gw_label = gw_rec.question[:35] + ("…" if len(gw_rec.question) > 35 else "")
                    col_gw_btn, col_gw_del = st.columns([4, 1])
                    with col_gw_btn:
                        _pm = f" [{gw_rec.provider_name}/{gw_rec.model_name}]" if gw_rec.model_name else ""
                        if st.button(f"🔌 {gw_label}{_pm}", key=f"view_gw_{gw_rec.id}"):
                            st.session_state.view_gateway_qa_record_id = gw_rec.id
                    with col_gw_del:
                        if st.button("🗑️", key=f"del_gw_{gw_rec.id}"):
                            db.delete_gateway_qa_record(gw_rec.id)  # type: ignore[arg-type]
                            st.rerun()


    # =================================================================
    # Main area – Tabs (filtered by visibility settings)
    # =================================================================
    st.title(t("app_title", lang))
    st.caption(t("app_subtitle", lang))

    _is_native_gemini = bool(preset_cfg.get("native_gemini"))
    _visible = st.session_state.get("_visible_tabs", ["tab_analyze", "tab_read", "tab_qa", "tab_gateway", "tab_chat", "tab_tts"])

    # Build tab labels and keys for visible tabs only
    _TAB_REGISTRY = [
        ("tab_analyze", t("tab_analyze", lang)),
        ("tab_read", t("tab_read", lang)),
        ("tab_qa", t("tab_qa", lang)),
        ("tab_gateway", t("tab_gateway", lang)),
        ("tab_chat", t("tab_chat", lang)),
        ("tab_tts", t("tab_tts", lang)),
    ]
    _shown = [(k, lbl) for k, lbl in _TAB_REGISTRY if k in _visible]
    if not _shown:
        _shown = _TAB_REGISTRY  # fallback: show all

    _tab_objects = st.tabs([lbl for _, lbl in _shown])
    _tab_map = {k: tab_obj for (k, _), tab_obj in zip(_shown, _tab_objects)}

    if "tab_analyze" in _tab_map:
        with _tab_map["tab_analyze"]:
            page_analyze(lang, api_key, model, base_url, provider_name=preset, is_native_gemini=_is_native_gemini)

    if "tab_read" in _tab_map:
        with _tab_map["tab_read"]:
            page_read_book(lang)

    if "tab_qa" in _tab_map:
        with _tab_map["tab_qa"]:
            page_qa(lang, api_key, model, base_url, provider_name=preset, is_native_gemini=_is_native_gemini)

    if "tab_gateway" in _tab_map:
        with _tab_map["tab_gateway"]:
            page_gateway(lang, api_key, model, base_url, provider_name=preset, is_native_gemini=_is_native_gemini)

    if "tab_chat" in _tab_map:
        with _tab_map["tab_chat"]:
            page_chat(lang, api_key, model, base_url, provider_name=preset, is_native_gemini=_is_native_gemini)

    if "tab_tts" in _tab_map:
        with _tab_map["tab_tts"]:
            page_tts(lang)


if __name__ == "__main__":
    main()

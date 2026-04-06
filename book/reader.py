"""PDF chapter reader – extracts TOC and chapter text from Beyond Feelings PDFs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF


@dataclass
class Chapter:
    """A single chapter/section extracted from the PDF."""
    level: int          # TOC depth (1 = Part, 2 = Chapter, 3 = Section, etc.)
    title: str
    page_start: int     # 0-based page index
    page_end: int       # 0-based page index (inclusive)


@dataclass
class BookInfo:
    """Metadata + chapters for one PDF book."""
    filename: str
    title: str
    chapters: List[Chapter] = field(default_factory=list)


def _build_chapters_from_toc(toc: list, total_pages: int) -> List[Chapter]:
    """Convert PyMuPDF TOC entries into Chapter objects with page ranges."""
    chapters: List[Chapter] = []
    for i, entry in enumerate(toc):
        level, title, page_num = entry[0], entry[1], entry[2]
        page_start = max(page_num - 1, 0)  # fitz TOC is 1-based

        # page_end = next entry's start - 1, or last page
        if i + 1 < len(toc):
            next_page = max(toc[i + 1][2] - 1, 0)
            page_end = max(next_page - 1, page_start)
        else:
            page_end = total_pages - 1

        chapters.append(Chapter(
            level=level,
            title=title.strip(),
            page_start=page_start,
            page_end=page_end,
        ))
    return chapters


def _make_fallback_chapters(total_pages: int, pages_per_chunk: int = 10) -> List[Chapter]:
    """If a PDF has no TOC, split into fixed-size chunks."""
    chapters: List[Chapter] = []
    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk - 1, total_pages - 1)
        chapters.append(Chapter(
            level=1,
            title=f"Pages {start + 1}\u2013{end + 1}",
            page_start=start,
            page_end=end,
        ))
    return chapters


def load_book(pdf_path: str) -> BookInfo:
    """Load a PDF and extract its chapter structure.

    Uses the PDF's built-in TOC (bookmarks). If no TOC exists,
    falls back to splitting by fixed page ranges.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
    total_pages = len(doc)

    if toc:
        chapters = _build_chapters_from_toc(toc, total_pages)
    else:
        chapters = _make_fallback_chapters(total_pages)

    filename = os.path.basename(pdf_path)
    title = doc.metadata.get("title", "") or filename
    doc.close()
    return BookInfo(filename=filename, title=title, chapters=chapters)


def extract_chapter_text(pdf_path: str, chapter: Chapter) -> str:
    """Extract the text content of a specific chapter (page range)."""
    doc = fitz.open(pdf_path)
    lines: list[str] = []
    for page_idx in range(chapter.page_start, min(chapter.page_end + 1, len(doc))):
        page = doc[page_idx]
        text = page.get_text("text")
        if text.strip():
            lines.append(f"\n---  Page {page_idx + 1}  ---\n")
            lines.append(text)
    doc.close()
    return "\n".join(lines) if lines else ""


def extract_chapter_images(pdf_path: str, chapter: Chapter, dpi: int = 150) -> List[bytes]:
    """Render chapter pages as PNG images (for scanned / image-only PDFs)."""
    doc = fitz.open(pdf_path)
    images: List[bytes] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_idx in range(chapter.page_start, min(chapter.page_end + 1, len(doc))):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def is_scanned_pdf(pdf_path: str, sample_pages: int = 10) -> bool:
    """Check if a PDF is image-only (scanned) by sampling a few pages for text."""
    doc = fitz.open(pdf_path)
    total = min(sample_pages, len(doc))
    has_text = False
    for i in range(total):
        if doc[i].get_text("text").strip():
            has_text = True
            break
    doc.close()
    return not has_text


def discover_books(book_dir: str) -> List[str]:
    """Return a list of PDF file paths in the given directory."""
    if not os.path.isdir(book_dir):
        return []
    return sorted(
        os.path.join(book_dir, f)
        for f in os.listdir(book_dir)
        if f.lower().endswith(".pdf")
    )


def extract_full_text(pdf_path: str, max_chars: int = 80_000) -> str:
    """Extract all text from a PDF, truncated to *max_chars*.

    Used as context for Q&A. Skips image-only pages.
    Returns an empty string for fully scanned PDFs.
    """
    doc = fitz.open(pdf_path)
    parts: list[str] = []
    total = 0
    for page in doc:
        text = page.get_text("text").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    doc.close()
    full = "\n\n".join(parts)
    return full[:max_chars] if len(full) > max_chars else full


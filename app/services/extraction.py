from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


class ExtractionError(Exception):
    """Base class for text extraction failures."""


class UnsupportedFileTypeError(ExtractionError):
    """Raised when the file extension is not .pdf or .txt."""


class EmptyDocumentError(ExtractionError):
    """Raised when no text could be extracted (e.g. a scanned PDF)."""


def extract_text(filename: str, content: bytes) -> str:
    """Extract and normalize plain text from a PDF or TXT file's bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        raw = _extract_pdf(content)
    elif suffix == ".txt":
        raw = _extract_txt(content)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    text = _normalize(raw)
    if not text:
        raise EmptyDocumentError("No extractable text found in the document.")
    return text


def _extract_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise ExtractionError("The PDF file is corrupted or unreadable.") from exc
    return "\n\n".join(pages)


def _extract_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r" ?\n ?", "\n", text)         # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)       # max one blank line
    return text.strip()
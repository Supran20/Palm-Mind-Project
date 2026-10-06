from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.services.extraction import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    extract_text,
)


def test_extracts_txt() -> None:
    assert extract_text("notes.txt", b"Hello   world\r\n\r\n\r\n\r\nBye") == "Hello world\n\nBye"


def test_rejects_unsupported_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("image.png", b"data")


def test_empty_txt_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"   \n  ")


def test_blank_pdf_raises() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(EmptyDocumentError):
        extract_text("blank.pdf", buffer.getvalue())
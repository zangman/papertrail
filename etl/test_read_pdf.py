"""Unit tests for read_pdf.strip_md."""

import pytest

from read_pdf import strip_md


class TestStripMd:

  def test_removes_heading_markers(self):
    assert strip_md("# Title") == "Title"
    assert strip_md("## Subtitle") == "Subtitle"
    assert strip_md("### Heading") == "Heading"

  def test_removes_bold_and_italic_markers(self):
    assert strip_md("**bold**") == "bold"
    assert strip_md("*italic*") == "italic"

  def test_removes_html_tags(self):
    # Intended behavior: HTML tags are stripped.
    assert strip_md("<b>bold</b>") == "bold"
    assert strip_md('<a href="x">link</a>') == "link"

  def test_no_markdown_is_unchanged(self):
    assert strip_md("plain text") == "plain text"

  def test_empty_string(self):
    assert strip_md("") == ""

  def test_combined(self):
    # Intended behavior: markers and HTML tags are all stripped.
    assert strip_md("## **Bold** <i>title</i>") == "Bold title"

  def test_strips_hash_and_asterisk_everywhere(self):
    # Current behavior removes '#' and '*' anywhere, even mid-word.
    assert strip_md("a#b*c") == "abc"

  def test_strips_surrounding_whitespace(self):
    assert strip_md("  padded  ") == "padded"
    assert strip_md("\n# Title\n") == "Title"

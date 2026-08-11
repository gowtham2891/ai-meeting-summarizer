"""Guards against mojibake in shipped source.

Text decoded with the wrong codec and re-encoded as UTF-8 stays *valid* UTF-8,
so nothing else catches it -- it simply renders as garbage in the deployed app.
This suite exists because exactly that shipped: the sidebar read
"Mock mode <junk> no credentials required" on all four live apps.

Markers are built from codepoints rather than written as literal glyphs, so
this file cannot match its own patterns.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Sequences that only occur when UTF-8 bytes were read as cp1252 or latin-1.
#: U+00E2 U+20AC  - an em dash, en dash or curly quote round-tripped wrongly
#: U+00C3 U+00A2  - the same text round-tripped twice
#: U+00EF U+00BB U+00BF - a UTF-8 byte-order mark read as latin-1
#: U+FFFD         - the decoder gave up entirely
MOJIBAKE_MARKERS = (
    chr(0x00E2) + chr(0x20AC),
    chr(0x00C3) + chr(0x00A2),
    chr(0x00EF) + chr(0x00BB) + chr(0x00BF),
    chr(0xFFFD),
)

SKIP_PARTS = {"__pycache__", ".git", "build", "dist", ".venv", "venv", ".testenv"}


def source_files():
    found = set()
    for pattern in ("*.py", "**/*.py", "*.md", "*.toml", "*.txt"):
        for path in ROOT.glob(pattern):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            found.add(path)
    return sorted(found)


FILES = source_files()


def test_there_is_something_to_check():
    """A glob that silently matched nothing would make this suite vacuous."""
    assert len(FILES) > 5


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_mojibake(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    for marker in MOJIBAKE_MARKERS:
        codepoints = " ".join(f"U+{ord(c):04X}" for c in marker)
        assert marker not in text, (
            f"{path.name} contains {codepoints} - this text was round-tripped "
            f"through the wrong codec and renders as garbage"
        )


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_decodes_as_utf8(path):
    """Every shipped file must be valid UTF-8."""
    path.read_bytes().decode("utf-8")


def test_this_file_uses_escapes_not_literals():
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    for marker in MOJIBAKE_MARKERS:
        assert marker not in source, (
            "build markers with chr() so this file does not fail itself"
        )

"""Hangul-presence language detection used to pick chat prompt variants."""
import re

_HANGUL_RE = re.compile(r"[가-힣]")


def detect_lang(text: str) -> str:
    """Returns "ko" if the text contains any Hangul characters, else "en"."""
    return "ko" if _HANGUL_RE.search(text or "") else "en"

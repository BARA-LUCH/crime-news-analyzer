"""
utils/language.py
Detects the language of input text.
Supports English, Hebrew, Arabic, and others.
"""

import re


def detect_language(text: str) -> str:
    """
    Detect language from text using Unicode character ranges.
    Returns ISO 639-1 language code: 'en', 'he', 'ar', or 'unknown'.
    """
    if not text or len(text.strip()) < 5:
        return "unknown"

    # Count characters by script
    hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total = max(len(text), 1)

    hebrew_ratio = hebrew_chars / total
    arabic_ratio = arabic_chars / total
    latin_ratio = latin_chars / total

    # Determine dominant script
    if hebrew_ratio > 0.15:
        return "he"
    elif arabic_ratio > 0.15:
        return "ar"
    elif latin_ratio > 0.2:
        return "en"

    # Fallback: use langdetect if available
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        pass

    return "en"  # Default to English

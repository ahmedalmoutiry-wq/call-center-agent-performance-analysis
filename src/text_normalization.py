"""Text normalization for Word Error Rate (WER).

WER compares words exactly, so "1:30 PM" and "one thirty p m" would count as
errors even though they mean the same thing. We normalize BOTH the reference
(human transcript) and the hypothesis (Whisper output) with the same rules,
so WER measures recognition mistakes, not formatting differences.

The rules follow the conventions of the HarperValleyBank reference:
lower case, no punctuation, numbers written as words, "a m" / "p m".
"""

import re

from num2words import num2words

# Hesitation sounds. Whisper usually drops them, while human transcribers write
# them, so they are removed from both sides (Whisper's own normalizer does the same).
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "mm", "mhm", "ah", "er", "eh"}

# Same word, different spelling
SPELLING = {"ok": "okay", "alright": "alright"}


def _number_to_words(match: re.Match) -> str:
    """Convert '146' to 'one hundred and forty six'."""
    return " " + num2words(int(match.group(0))).replace("-", " ").replace(",", "") + " "


def _ordinal_to_words(match: re.Match) -> str:
    """Convert '3rd' to 'third'."""
    return " " + num2words(int(match.group(1)), to="ordinal").replace("-", " ") + " "


def _time_to_words(match: re.Match) -> str:
    """Convert '1:30' to 'one thirty' and '5:00' to 'five'."""
    hours, minutes = int(match.group(1)), int(match.group(2))
    words = num2words(hours)
    if minutes:
        words += " " + ("oh " if minutes < 10 else "") + num2words(minutes)
    return " " + words.replace("-", " ") + " "


def normalize_text(text: str) -> str:
    """Return lower-case words only, with numbers spelled out and noise removed."""
    text = text.lower()

    # 1. annotation tags used in the reference: [noise], [laughter], <unk>
    text = re.sub(r"\[[^\]]*\]|<unk>", " ", text)

    # 2. word fragments marked with '~' in the reference (e.g. "wou~")
    text = re.sub(r"\S*~", " ", text)

    # 3. numbers written with digits (Whisper style) -> words (reference style)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)                       # 1,000 -> 1000
    text = re.sub(r"\$\s?(\d+)(?:\.\d+)?", r"\1 dollars", text)         # $146 -> 146 dollars
    text = re.sub(r"\b(\d{1,2})\s*([ap])\.?\s?m\b\.?", r"\1 \2 m", text)  # 11am, 11 a.m. -> 11 a m
    text = re.sub(r"\b([ap])\.m\.?", r"\1 m", text)                     # a.m. -> a m
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time_to_words, text)       # 1:30 -> one thirty
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", _ordinal_to_words, text)   # 3rd -> third
    text = re.sub(r"\d+", _number_to_words, text)                       # 146 -> one hundred ...

    # 4. punctuation: keep letters, apostrophes (it's, i'd) and spaces
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\ball right\b", "alright", text)

    words = [SPELLING.get(w, w) for w in text.split()]
    words = [w.strip("'") for w in words if w not in FILLERS]
    return " ".join(w for w in words if w)


# Examples that double as tests (run in notebook 02 before any evaluation)
NORMALIZATION_EXAMPLES = [
    ("Hello, this is Harper Valley National Bank.", "hello this is harper valley national bank"),
    ("uh i wou~ i need [noise] <unk> help", "i i need help"),
    ("$146 has been transferred.", "one hundred and forty six dollars has been transferred"),
    ("Thursday at 1:30 p.m.", "thursday at one thirty p m"),
    ("11 AM", "eleven a m"),
    ("11am", "eleven a m"),
    ("Okay, all right.", "okay alright"),
    ("It's the 3rd.", "it's the third"),
]


def check_normalization() -> None:
    """Raise an error if any example is not normalized as expected."""
    for raw, expected in NORMALIZATION_EXAMPLES:
        result = normalize_text(raw)
        assert result == expected, f"{raw!r} -> {result!r}, expected {expected!r}"

"""Evaluation helpers: Word Error Rate (WER) and overlap checks."""

import jiwer
import pandas as pd

from src.text_normalization import normalize_text


def word_error_rate(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus-level WER with its parts, after normalizing both sides.

    WER = (substitutions + deletions + insertions) / words in the reference
      substitution - a wrong word         ("card" heard as "part")
      deletion     - a missed word        (reference word with no match)
      insertion    - an extra word        (word that was never said)

    Corpus-level means all errors are summed before dividing, so long calls
    weigh more than short ones (averaging per-call WERs would over-weight short calls).
    Pairs whose normalized reference is empty are skipped (nothing to recognize).
    """
    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(references, hypotheses)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return {"wer": None, "ref_words": 0, "substitutions": 0, "deletions": 0,
                "insertions": 0, "n_pairs": 0}

    refs, hyps = zip(*pairs)
    out = jiwer.process_words(list(refs), list(hyps))
    return {
        "wer": round(out.wer, 4),
        "ref_words": out.hits + out.substitutions + out.deletions,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "n_pairs": len(pairs),
    }


def reference_text(utterances: pd.DataFrame) -> str:
    """Join the human transcript of the given utterances in time order."""
    return " ".join(utterances.sort_values("start_s")["human_transcript"])


def overlap_seconds(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    """Seconds of [start, end] covered by any of the intervals."""
    return sum(max(0.0, min(end, e) - max(start, s)) for s, e in intervals)

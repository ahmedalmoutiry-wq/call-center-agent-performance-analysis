"""Turn recognized words into a conversation with speaker turns.

Each audio channel holds one speaker, so we already know WHO said each word.
What we still need is the conversation ORDER: who spoke first, when the
speaker changed, and which words belong together in one turn.
"""

import pandas as pd

ROLE_NAMES = {"caller": "Customer", "agent": "Agent"}

# Phrases that, in this dataset, only the bank agent says
AGENT_PHRASES = ("harper valley", "how can i help", "how may i help")


def words_to_utterances(words: pd.DataFrame, max_pause_s: float = 1.0) -> pd.DataFrame:
    """Group one speaker's consecutive words into utterances.

    A new utterance starts when the same speaker pauses longer than `max_pause_s`.
    Expects columns: call_id, channel_index, speaker_role, word, start_s, end_s, probability.
    """
    rows = []
    for (call_id, channel, role), w in words.groupby(["call_id", "channel_index", "speaker_role"]):
        w = w.sort_values("start_s")
        gap_before = w["start_s"] - w["end_s"].shift()      # silence before each word (NaN for the first)
        new_utterance = gap_before.gt(max_pause_s)
        for _, g in w.groupby(new_utterance.cumsum()):
            rows.append({
                "call_id": call_id, "channel_index": channel, "speaker_role": role,
                "start_s": g["start_s"].min(), "end_s": g["end_s"].max(),
                "text": " ".join(g["word"].astype(str)).strip(),
                "n_words": len(g), "mean_word_probability": g["probability"].mean(),
            })
    return pd.DataFrame(rows)


def utterances_to_turns(utterances: pd.DataFrame) -> pd.DataFrame:
    """Order utterances by start time and merge neighbours from the same speaker into turns."""
    turns = []
    for call_id, u in utterances.sort_values(["call_id", "start_s"]).groupby("call_id", sort=False):
        current = None
        for row in u.itertuples(index=False):
            if current is not None and current["channel_index"] == row.channel_index:
                # same speaker continues: extend the current turn
                total = current["n_words"] + row.n_words
                current["mean_word_probability"] = (current["mean_word_probability"] * current["n_words"]
                                                    + row.mean_word_probability * row.n_words) / total
                current["n_words"] = total
                current["end_s"] = max(current["end_s"], row.end_s)
                current["text"] += " " + row.text
            else:
                if current is not None:
                    turns.append(current)
                current = row._asdict()
        if current is not None:
            turns.append(current)

    turns = pd.DataFrame(turns)
    turns["turn_index"] = turns.groupby("call_id").cumcount() + 1
    turns["speaker"] = "Speaker " + turns["channel_index"].astype(str)
    turns["role"] = turns["speaker_role"].map(ROLE_NAMES)
    turns["overlaps_other_speaker"] = _overlap_flags(turns)
    return turns


def _overlap_flags(turns: pd.DataFrame, min_overlap_s: float = 0.5) -> list[bool]:
    """True when a turn overlaps a turn of the other speaker by more than `min_overlap_s`.

    Word timestamps are approximate, so tiny overlaps at turn boundaries are ignored.
    """
    flags = []
    for row in turns.itertuples():
        others = turns[(turns["call_id"] == row.call_id) & (turns["channel_index"] != row.channel_index)]
        overlap = (others["end_s"].clip(upper=row.end_s) - others["start_s"].clip(lower=row.start_s)).clip(lower=0)
        flags.append(bool((overlap > min_overlap_s).any()))
    return flags


def role_from_content(turns: pd.DataFrame, first_n: int = 3) -> dict:
    """Guess which channel is the agent from what is said, without using the metadata.

    Returns {call_id: agent_channel or None}. Used only to CHECK the metadata roles.
    """
    guesses = {}
    for call_id, t in turns.groupby("call_id"):
        agent_channels = set()
        for channel, ct in t.groupby("channel_index"):
            opening = " ".join(ct.sort_values("start_s")["text"].head(first_n)).lower()
            if any(phrase in opening for phrase in AGENT_PHRASES):
                agent_channels.add(channel)
        guesses[call_id] = agent_channels.pop() if len(agent_channels) == 1 else None
    return guesses


def format_conversation(turns: pd.DataFrame) -> str:
    """Readable transcript: one line per turn, with time and role."""
    lines = []
    for t in turns.sort_values("turn_index").itertuples():
        flag = "  [low confidence]" if getattr(t, "low_confidence", False) else ""
        lines.append(f"[{t.start_s:6.1f}s] {t.speaker} ({t.role}): {t.text}{flag}")
    return "\n".join(lines)

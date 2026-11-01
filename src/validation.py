"""Helpers for the validation notebook and the final per-call output."""

import math

import pandas as pd


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score interval).

    With small samples (for example 8 of 10), the true rate is uncertain.
    This interval shows that uncertainty instead of reporting "80%" as if it were exact.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def rate_with_interval(values: pd.Series, positive: str = "yes") -> str:
    """'8/10 (80%, 95% CI 49%-94%)' for a column of yes/no answers. Empty answers are ignored."""
    answers = values.astype(str).str.strip().str.lower()
    answers = answers[answers.isin(["yes", "no"])]
    if answers.empty:
        return "not reviewed yet"
    k, n = int((answers == positive).sum()), len(answers)
    low, high = wilson_interval(k, n)
    return f"{k}/{n} ({k / n:.0%}, 95% CI {low:.0%}-{high:.0%})"


def build_call_record(call_id: str, turns: pd.DataFrame, summary: pd.Series, sentiment: pd.Series,
                      wer_rows: pd.DataFrame, info: dict) -> dict:
    """One call in the format required by the assessment:
    call_id, speakers, transcript (segmented turns), summary, sentiment, notes / limitations."""
    turns = turns.sort_values("turn_index")
    speakers = turns.drop_duplicates("speaker").set_index("speaker")["role"].sort_index().to_dict()

    notes = [f"Simulated (role-play) call from the HarperValleyBank dataset; transcript produced by Whisper {info['asr_model']}."]
    for row in wer_rows.itertuples():
        notes.append(f"Word error rate ({'customer' if row.speaker_role == 'caller' else 'agent'} side): {row.wer:.1%}.")
    n_low = int(turns["low_confidence"].sum())
    if n_low:
        notes.append(f"{n_low} turn(s) with low recognition confidence; their text may be wrong.")
    n_overlap = int(turns["overlaps_other_speaker"].sum())
    if n_overlap:
        notes.append(f"{n_overlap} turn(s) overlap the other speaker; their order is approximate.")
    if not bool(summary.get("issue_matches_task", True)):
        notes.append("Automatic check: the summarized issue does not mention the known task type.")
    if str(summary.get("ungrounded_numbers", "") or "") not in ("", "nan"):
        notes.append(f"Automatic check: number(s) not found in the transcript: {summary['ungrounded_numbers']}.")
    if not bool(sentiment.get("evidence_is_customer_quote", True)):
        notes.append("Automatic check: the sentiment evidence is not an exact customer quote.")

    return {
        "call_id": call_id,
        "speakers": speakers,
        "transcript": [
            {"turn": int(t.turn_index), "speaker": t.speaker, "role": t.role,
             "start_s": float(t.start_s), "end_s": float(t.end_s), "text": t.text}
            for t in turns.itertuples()
        ],
        "summary": {
            "customer_issue": summary["customer_issue"],
            "agent_action": summary["agent_action"],
            "outcome": summary["outcome"],
        },
        "sentiment": {
            "label": sentiment["sentiment"],
            "justification": sentiment["justification"],
            "evidence": sentiment["evidence"],
            "beginning": sentiment.get("start_label"),
            "end": sentiment.get("end_label"),
            "change": sentiment.get("start_to_end"),
        },
        "notes": notes,
    }

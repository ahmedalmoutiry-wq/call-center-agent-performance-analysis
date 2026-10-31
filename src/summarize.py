"""Call summarization: prompt, output parsing and automatic checks.

Approach: ABSTRACTIVE summarization with an instruction-tuned language model.
The model writes new sentences (it does not copy sentences from the call),
so we add checks that the summary stays grounded in the conversation.
"""

import json
import re

import pandas as pd

from src.text_normalization import normalize_text

SUMMARY_FIELDS = ["customer_issue", "agent_action", "outcome"]
NOT_STATED = "not stated"

SYSTEM_PROMPT = (
    "You summarize customer-service phone calls for a quality team. "
    "Use ONLY information that is stated in the transcript. Never guess or add details. "
    f'If something is not stated in the call, write "{NOT_STATED}".'
)

USER_TEMPLATE = """Transcript of a call between a bank Agent and a Customer:

{transcript}

Write a short summary as JSON with exactly these keys:
- "customer_issue": what the customer wanted or the problem they had
- "agent_action": what the agent did to help
- "outcome": how the call ended, or the next step

Each value is one short sentence (at most 20 words). Include amounts, dates or times only if they were said.
Return only the JSON object."""


def transcript_text(turns: pd.DataFrame) -> str:
    """Format a call's turns as 'Agent: ...' / 'Customer: ...' lines."""
    return "\n".join(f"{t.role}: {t.text}" for t in turns.sort_values("turn_index").itertuples())


def build_messages(turns: pd.DataFrame) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript_text(turns))}]


def parse_summary(raw: str) -> dict:
    """Extract the three fields from the model output. parse_ok=False if no valid JSON was found."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    result = {field: str(data.get(field, "")).strip() for field in SUMMARY_FIELDS}
    result["parse_ok"] = all(result[field] for field in SUMMARY_FIELDS)
    return result


# Words that show the summary names the right customer request (dataset task types)
TASK_KEYWORDS = {
    "check balance": ["balance"],
    "get branch hours": ["hour", "open", "close"],
    "replace card": ["card"],
    "reset password": ["password"],
    "pay bill": ["bill", "pay"],
    "order checks": ["check", "checkbook", "cheque"],
    "schedule appointment": ["appointment"],
    "transfer money": ["transfer"],
}


def issue_matches_task(issue: str, task_type: str) -> bool:
    """True if the summarized customer issue mentions the call's known task type."""
    issue = issue.lower()
    return any(keyword in issue for keyword in TASK_KEYWORDS.get(task_type, []))


def ungrounded_numbers(summary: str, transcript: str) -> list[str]:
    """Numbers, amounts and times in the summary that do NOT appear in the transcript.

    Both sides are normalized the same way as for WER, so "$159" in the summary
    matches "one hundred and fifty nine dollars" or "$159" in the transcript.
    A non-empty list means the model may have invented a detail.
    """
    transcript_norm = f" {normalize_text(transcript)} "
    missing = []
    for token in re.findall(r"\$?\d[\d,]*(?::\d{2})?(?:\s?[ap]\.?m\.?)?", summary, flags=re.IGNORECASE):
        token_norm = normalize_text(token).replace(" dollars", "")
        if token_norm and f" {token_norm} " not in transcript_norm:
            missing.append(token.strip())
    return missing

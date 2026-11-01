"""Customer sentiment analysis: prompt, parsing and helpers.

Two methods are used together:
  1. A sentiment CLASSIFIER (RoBERTa fine-tuned on tweets) labels every customer
     turn on its own. Fast and consistent, but sees no context.
  2. The instruction-tuned LANGUAGE MODEL from notebook 04 reads the whole call
     and gives ONE label for the customer, with a reason and a quote as evidence.
The language-model label is the final one; the classifier gives a second opinion
and the beginning-vs-end comparison.
"""

import json
import re

import pandas as pd

from src.text_normalization import normalize_text

LABELS = ["negative", "neutral", "positive"]
CLASSIFIER_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # CC-BY-4.0

SYSTEM_PROMPT = (
    "You analyse customer-service phone calls for a quality team. "
    "Judge only the CUSTOMER's sentiment, not the agent's. "
    "Base your answer only on what the customer says in the transcript."
)

USER_TEMPLATE = """Transcript of a call between a bank Agent and a Customer:

{transcript}

Classify the CUSTOMER's overall sentiment in this call as "positive", "neutral" or "negative".

Guidance:
- Sentiment is the emotion the customer expresses (for example frustration, relief, thanks). It is not whether the problem was solved.
- Plain, polite requests with no clear emotion are "neutral".
- A "no" that only answers a question (for example "no, that's all") is not negative.

Return only a JSON object with these keys:
- "sentiment": "positive", "neutral" or "negative"
- "justification": one short sentence explaining why (at most 25 words)
- "evidence": the customer's exact words that best support the label (a short quote)"""


def build_messages(transcript: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)}]


def parse_sentiment(raw: str) -> dict:
    """Extract label, justification and evidence. parse_ok=False if the output is unusable."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    label = str(data.get("sentiment", "")).strip().lower()
    return {
        "sentiment": label if label in LABELS else "",
        "justification": str(data.get("justification", "")).strip(),
        "evidence": str(data.get("evidence", "")).strip(),
        "parse_ok": label in LABELS,
    }


def evidence_is_customer_quote(evidence: str, customer_text: str) -> bool:
    """True if the quoted evidence really appears in what the customer said (after normalization)."""
    quote = normalize_text(evidence)
    return bool(quote) and f" {quote} " in f" {normalize_text(customer_text)} "


def load_classifier(model_id: str = CLASSIFIER_ID):
    """Return a function texts -> DataFrame with p_negative, p_neutral, p_positive and label."""
    import torch
    from transformers import pipeline

    clf = pipeline("text-classification", model=model_id, top_k=None,
                   device=0 if torch.cuda.is_available() else -1)

    def classify(texts: list[str]) -> pd.DataFrame:
        rows = []
        for scores in clf(list(texts), truncation=True):
            probs = {s["label"].lower(): s["score"] for s in scores}
            rows.append({f"p_{label}": probs.get(label, 0.0) for label in LABELS})
        out = pd.DataFrame(rows)
        out["label"] = out[[f"p_{l}" for l in LABELS]].idxmax(axis=1).str.replace("p_", "", regex=False)
        return out

    return classify


def label_from_probabilities(probs: pd.DataFrame) -> str:
    """Average the per-turn probabilities and return the class with the highest mean."""
    means = probs[[f"p_{l}" for l in LABELS]].mean()
    return means.idxmax().replace("p_", "")

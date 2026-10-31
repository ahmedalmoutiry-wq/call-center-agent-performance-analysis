"""Load the HarperValleyBank (HVB) files into pandas tables.

Used by every notebook, so the loading logic lives in one place.
Run scripts/download_data.py first so that data/raw/ exists.
"""

import json
from pathlib import Path

import pandas as pd

NOISE_TAG = "[noise]"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` until we find the folder that holds requirements.txt.

    Notebooks run from notebooks/, scripts from the root, so a fixed relative
    path would break. Searching upwards works from anywhere inside the project.
    """
    path = (start or Path.cwd()).resolve()
    for folder in [path, *path.parents]:
        if (folder / "requirements.txt").exists():
            return folder
    raise FileNotFoundError("Project root not found (no requirements.txt above the current folder).")


PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_call_ids() -> list[str]:
    """Return all call IDs saved by the download script."""
    ids_file = RAW_DIR / "call_ids.txt"
    if not ids_file.exists():
        raise FileNotFoundError(f"{ids_file} is missing. Run: python3 scripts/download_data.py")
    return ids_file.read_text().split()


def load_json(kind: str, call_id: str):
    """Read data/raw/<kind>/<call_id>.json, where kind is 'metadata' or 'transcript'."""
    with open(RAW_DIR / kind / f"{call_id}.json") as f:
        return json.load(f)


def audio_path(call_id: str, channel_index: int) -> Path:
    """Return the WAV file that holds one audio channel of a call.

    Checked in notebook 01: the caller/ folder always holds channel 1 and the
    agent/ folder always holds channel 2. The speaker ROLE of a channel must be
    read from the transcript, because in 37 calls the agent is on channel 1.
    """
    folder = {1: "caller", 2: "agent"}[channel_index]
    return RAW_DIR / "audio" / folder / f"{call_id}.wav"


def build_calls_table(call_ids: list[str]) -> pd.DataFrame:
    """One row per call: task, duration, quality ratings and channel layout."""
    rows = []
    for call_id in call_ids:
        meta = load_json("metadata", call_id)
        utterances = load_json("transcript", call_id)

        # which channel(s) each role was recorded on, according to the transcript
        role_channels: dict[str, set] = {"agent": set(), "caller": set()}
        for u in utterances:
            role_channels.setdefault(u["speaker_role"], set()).add(u["channel_index"])

        rows.append({
            "call_id": call_id,
            "task_type": meta["tasks"][0]["task_type"] if meta.get("tasks") else None,
            "n_tasks": len(meta.get("tasks", [])),
            "duration_s": (meta["end_time_ms"] - meta["start_time_ms"]) / 1000,
            "session": meta.get("session"),
            "caller_audio_quality": meta["labels"].get("caller_mos"),
            "agent_audio_quality": meta["labels"].get("agent_mos"),
            "script_adherence": meta["labels"].get("lhvb_script"),
            "n_utterances": len(utterances),
            "agent_channels": sorted(role_channels["agent"]),
            "caller_channels": sorted(role_channels["caller"]),
        })

    calls = pd.DataFrame(rows)
    # Normal layout: caller on channel 1, agent on channel 2.
    calls["agent_on_channel_1"] = calls["agent_channels"].apply(lambda c: c == [1])
    return calls


def build_utterances_table(call_ids: list[str]) -> pd.DataFrame:
    """One row per utterance (segment) with timing, role, text and emotion scores."""
    rows = []
    for call_id in call_ids:
        for u in load_json("transcript", call_id):
            emotion = u.get("emotion", {})
            human_text = u["human_transcript"].strip()
            rows.append({
                "call_id": call_id,
                "index": u["index"],
                "speaker_role": u["speaker_role"],
                "channel_index": u["channel_index"],
                "start_s": u["start_ms"] / 1000,
                "end_s": (u["start_ms"] + u["duration_ms"]) / 1000,
                "human_transcript": human_text,
                "provider_asr": u["transcript"].strip(),   # old machine output, NOT our result
                "is_empty": human_text == "",
                "is_noise_only": human_text == NOISE_TAG,
                "p_negative": emotion.get("negative"),
                "p_neutral": emotion.get("neutral"),
                "p_positive": emotion.get("positive"),
            })

    utts = pd.DataFrame(rows)
    # emotion label = the class with the highest model probability
    probs = utts[["p_negative", "p_neutral", "p_positive"]]
    utts["emotion_label"] = probs.idxmax(axis=1).str.replace("p_", "", regex=False)
    utts["n_words"] = utts["human_transcript"].str.split().str.len()
    return utts

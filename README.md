# Call Center Agent Performance Analysis

AI pipeline that analyzes customer-service phone calls: **speech-to-text**, **speaker segmentation**, **call summarization** and **customer sentiment analysis**, with validation at every stage.

Built for the Maharah GDP Trainee Assessment.

## Results at a glance

25 calls from the HarperValleyBank corpus (simulated bank calls), processed end to end.

| Stage | What we measured | Result |
|---|---|---|
| Speech-to-text | Word Error Rate (WER) vs human reference | **4.9%** overall (agent 2.9%, customer 8.1%); the dataset's own ASR scores 7.5% |
| Speaker segmentation | Recognized words attributed to the right speaker | **99.4%**; content-based role check agreed with the metadata in every call where it could decide |
| Summaries | Manual review of 10 calls | **8/10 faithful** (95% CI 49%–94%), **9/10 complete** (95% CI 60%–98%) |
| Sentiment | Manual review of 10 calls | **9/10 labels correct** (95% CI 60%–98%); evidence quotes are real customer words in 24/25 calls |
| Final output | Required fields per call | All 25 calls complete in [`results/calls.json`](results/calls.json) |

Full numbers, charts, successes and failures: [`notebooks/06_validation_and_results.ipynb`](notebooks/06_validation_and_results.ipynb).

The human reference transcripts are used **only to evaluate** our outputs. They are never used as the pipeline's output.

## Presentation

Slides: [`presentation/slides.pdf`](presentation/slides.pdf) · Recording: https://drive.google.com/file/d/1FIAuClMLyRRpVieYVxIyZWMowDM3cxpz/view?usp=sharing (Arabic, about 10 minutes)

## Dataset

**HarperValleyBank** (Gridspace and Stanford University): 1,446 simulated consumer-banking calls, 23.7 hours.

- Source: https://github.com/cricketclub/gridspace-stanford-harper-valley
- Paper: Wu, Nafziger, Scodary and Maas, *HarperValleyBank: A Domain-Specific Spoken Dialog Corpus*, 2020. https://arxiv.org/abs/2010.13929
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Redistribution is allowed with attribution. Raw audio is not committed because of its size; [`scripts/download_data.py`](scripts/download_data.py) downloads it.

**Why this dataset.** 12 datasets and 8 papers were compared in [`research/dataset_survey.xlsx`](research/dataset_survey.xlsx). HarperValleyBank is the only candidate that passed all four required criteria (customer-service conversations, audio, a license that allows our use, free access) and also met all six desirable ones: human reference transcripts for both speakers, speaker-separated audio, role metadata, a published paper, extra labels for validation, and a manageable size. Real call-center audio is rarely public: datasets with real calls either withholdi the audio for privacy, provide no transcripts, or are paid.


**Important notes**
- The calls are **simulated** (scripted role-play by crowd workers), not real customer calls.
- **Data-quality finding:** in 37 of 1,446 calls (2.6%), the agent's voice is stored in the `caller/` audio folder. We verified the roles from the transcript content and the audio energy, and map *audio file → channel → role* instead of trusting folder names (notebook 01).
- The dataset's emotion labels are produced by a commercial model, not by humans, so they are used only as a noisy reference.

**Subset.** 25 calls: 3 per task type (8 types), chosen at random with a fixed seed (`SEED = 42`), plus 1 deliberate edge case from the 37 calls above. No filtering on audio quality.

## Repository structure

```
├── README.md
├── requirements.txt         # local notebooks (01, 03, 06)
├── requirements-gpu.txt     # GPU notebooks (02, 04, 05), run on Google Colab
├── research/
│   └── dataset_survey.xlsx  # datasets, papers, selection criteria, search method
├── scripts/
│   └── download_data.py     # downloads the dataset (metadata, transcripts, subset audio)
├── src/                     # reusable code used by the notebooks
│   ├── data_loading.py      # load calls/utterances, map audio channel to file
│   ├── asr.py               # Whisper loading and transcription
│   ├── text_normalization.py# text normalization for WER (with built-in test examples)
│   ├── evaluation.py        # WER and overlap helpers
│   ├── segmentation.py      # words -> utterances -> speaker turns, role check
│   ├── llm.py               # language model loading and greedy generation
│   ├── summarize.py         # summary prompt, parsing and grounding checks
│   ├── sentiment.py         # sentiment prompt, classifier, evidence check
│   └── validation.py        # confidence intervals, final per-call record
├── notebooks/               # run in order 01 -> 06
├── data/
│   └── README.md            # how to download the data (raw data is not committed)
├── results/                 # all outputs (CSV / JSON) and figures
└── presentation/            # slides and recording link
```

## Setup and how to run

### 1. Local setup (notebooks 01, 03, 06)

```bash
git clone https://github.com/ahmedalmoutiry-wq/call-center-agent-performance-analysis.git
cd call-center-agent-performance-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# metadata + transcripts for all 1,446 calls (about 30 MB)
python scripts/download_data.py
```

Run `notebooks/01_data_exploration.ipynb`. It writes `results/subset_ids.csv`. Then download the audio of the subset only:

```bash
python scripts/download_data.py --audio results/subset_ids.csv
```

macOS note: if you see `CERTIFICATE_VERIFY_FAILED`, see [`data/README.md`](data/README.md).

### 2. GPU notebooks (02, 04, 05) on Google Colab

Open each notebook in Colab, select **Runtime → Change runtime type → T4 GPU**, and **Run all**. The first cell clones this repository, installs the GPU libraries and downloads the data.

| Notebook | Open in Colab |
|---|---|
| 02 Speech-to-text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedalmoutiry-wq/call-center-agent-performance-analysis/blob/main/notebooks/02_transcription.ipynb) |
| 04 Summarization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedalmoutiry-wq/call-center-agent-performance-analysis/blob/main/notebooks/04_summarization.ipynb) |
| 05 Sentiment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedalmoutiry-wq/call-center-agent-performance-analysis/blob/main/notebooks/05_sentiment.ipynb) |

Each GPU notebook ends by downloading a zip of its result files; copy them into `results/`. All result files are already committed, so the local notebooks can be run without a GPU.

### 3. Run order

| # | Notebook | Runs on | Main output |
|---|---|---|---|
| 01 | `01_data_exploration` | laptop | `subset_ids.csv`, `data_quality_report.csv`, figures |
| 02 | `02_transcription` | Colab GPU | `asr_segments.csv`, `asr_words.csv`, `wer_summary.csv`, `wer_per_call.csv`, `asr_vad_experiment.csv` |
| 03 | `03_speaker_segmentation` | laptop | `segmented_turns.csv`, `transcripts/<call_id>.txt` |
| 04 | `04_summarization` | Colab GPU | `summaries.csv`, `summary_review.csv` |
| 05 | `05_sentiment` | Colab GPU | `sentiment.csv`, `sentiment_turns.csv`, `sentiment_review.csv` |
| 06 | `06_validation_and_results` | laptop | `calls.json`, `calls_overview.csv`, figures |

## Methodology

### Data exploration (01)
Completeness checks on all 1,446 calls (no missing files, one task and both speakers per call), reference quality (22% of reference segments are only `[noise]` and are excluded from WER), the channel-layout check described above, and a stratified subset.

### Speech-to-text (02)
- **Model:** faster-whisper (CTranslate2 implementation of OpenAI Whisper). Three sizes were compared on 3 calls; `medium.en` had the lowest WER (2.9%, vs 4.7% `small.en`, 4.3% `large-v3`).
- **Each speaker's audio file is transcribed separately**, with word timestamps.
- **WER** is computed against the human reference after the same normalization on both sides: lower case, no punctuation, numbers written as words, phone numbers read digit by digit, annotation tags and fillers removed.
- **Error analysis → experiment:** the worst customer transcript (81.8% WER) was almost all deletions. Its audio was very quiet, and the voice activity detector (VAD) kept only 31% of the customer's speech. Lowering the VAD threshold trades deletions for insertions; 0.2 gave the lowest customer WER (8.1%, vs 11.0% at the default 0.5). This setting was used for the final run.

### Speaker segmentation (03)
No diarization model is needed: each speaker has their own audio channel. Words are grouped into utterances (a new one after a pause over 1 s), then sorted by time and merged into turns. **Speaker 1 / Speaker 2** = audio channel 1 / 2; **Customer / Agent** come from the transcript metadata and are checked against the content (only the agent says the bank greeting). Turns are flagged for low recognition confidence and for overlapping speech.

### Summarization (04)
Abstractive summarization with `Qwen3-4B-Instruct-2507`, greedy decoding, JSON output with three fields (customer issue, agent action, outcome) and an instruction to write "not stated" for missing information. Checks: format, issue vs known task type, numbers present in the transcript, and a manual review of 10 calls.

### Customer sentiment (05)
Customer turns only. Final label from the same language model reading the whole call, with a one-sentence justification and a quoted piece of the customer's words as evidence (checked to be a real customer quote). A RoBERTa classifier (`cardiffnlp/twitter-roberta-base-sentiment-latest`) scores every customer turn as a second opinion and gives the beginning-vs-end comparison.

### Validation (06)
All checks in one place, 95% confidence intervals (Wilson) for the manual reviews, one success and one failure shown end to end, and the final per-call output with a completeness check.

## Output format

[`results/calls.json`](results/calls.json) holds one record per call (a flat version is in `results/calls_overview.csv`):

```json
{
  "call_id": "08f360b3f3b844ef",
  "speakers": {"Speaker 1": "Customer", "Speaker 2": "Agent"},
  "transcript": [
    {"turn": 1, "speaker": "Speaker 2", "role": "Agent", "start_s": 1.84, "end_s": 7.26,
     "text": "Hello, this is Harbor Valley National Bank. My name is Robert. How can I help you today?"}
  ],
  "summary": {
    "customer_issue": "The customer wanted to check her checking account balance.",
    "agent_action": "The agent confirmed the customer's identity and provided her checking account balance of $90.",
    "outcome": "The customer expressed satisfaction and stated there was nothing else needed."
  },
  "sentiment": {
    "label": "positive",
    "justification": "Customer expresses satisfaction and closes the call politely.",
    "evidence": "Fantastic. That's all I needed.",
    "beginning": "neutral", "end": "positive", "change": "improved"
  },
  "notes": [
    "Simulated (role-play) call from the HarperValleyBank dataset; transcript produced by Whisper medium.en.",
    "Word error rate (customer side): 7.1%.",
    "Word error rate (agent side): 2.3%.",
    "1 turn(s) with low recognition confidence; their text may be wrong.",
    "8 turn(s) overlap the other speaker; their order is approximate."
  ]
}
```

## Models and tools

| Purpose | Model / library | Version | License |
|---|---|---|---|
| Speech-to-text | Whisper `medium.en` via faster-whisper | faster-whisper 1.2.1 | MIT |
| Summaries and sentiment | `Qwen/Qwen3-4B-Instruct-2507` via transformers | transformers 5.21.0 | Apache-2.0 |
| Sentiment (second opinion) | `cardiffnlp/twitter-roberta-base-sentiment-latest` | — | CC BY 4.0 |
| WER | jiwer | 4.0.0 | Apache-2.0 |
| Data and plots | pandas, numpy, matplotlib | see `requirements.txt` | — |
| GPU runtime | Google Colab, Tesla T4 | torch 2.11.0+cu130 | — |

No paid APIs were used. All models run in the notebook; no call data is sent to external services.

## Limitations

- **Simulated data.** Scripted role-play with a small vocabulary and clear speech. Results will be worse on real call-center audio (accents, noise, long calls).
- **Small samples.** 25 processed calls, 10 manually reviewed per task; the confidence intervals above are wide.
- **Tuned on the evaluation set.** The VAD threshold was chosen on the same 25 calls, so the final WER is slightly optimistic. The two best thresholds (0.35 and 0.2) differ by a single word error.
- **Summaries can invent a typical ending** when a call is incomplete (one of 10 reviewed calls), and never used "not stated".
- **Sentiment.** No negative customers in the subset, so the negative class is untested on real calls; the language model is not fully consistent on polite closing phrases; sentiment is not the same as satisfaction.
- **Agreement is not accuracy.** The dataset's emotion labels come from another model.
- **Number formats.** Some numbers are ambiguous without context (for example "734 Main Street").

## Reproducibility

- Fixed random seed for subset selection; same subset on every run.
- Greedy decoding for the language model; Whisper with fixed settings in `src/asr.py`.
- Pinned versions in `requirements.txt` and `requirements-gpu.txt` (recorded from the Colab runtime used).
- Notebook 02 was run twice and gave identical WER results.
- Data is downloaded by script from the original source; nothing private or restricted is in the repository.

## Acknowledgements

Dataset: HarperValleyBank by Gridspace and Stanford University (CC BY 4.0).

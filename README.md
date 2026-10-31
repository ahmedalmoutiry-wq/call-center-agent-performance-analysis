# Call Center Agent Performance Analysis

AI pipeline that analyzes customer-service phone calls: speech-to-text, speaker segmentation, call summarization and customer sentiment analysis.

> **Status:** Work in progress. Dataset research and selection are complete. The pipeline is being implemented.

## Objective

Build a simple, reproducible solution that takes customer-service call recordings and produces, for each call:

- A transcript with speaker turns labelled **Customer** and **Agent**
- A short summary: customer issue, agent action, outcome
- A customer sentiment label with a short justification

## Dataset

**HarperValleyBank** (Gridspace & Stanford University): 1,446 simulated consumer-banking calls with speaker-separated audio and human reference transcripts.

- Source: https://github.com/cricketclub/gridspace-stanford-harper-valley
- Paper: Wu et al., *HarperValleyBank: A Domain-Specific Spoken Dialog Corpus*, 2020. https://arxiv.org/abs/2010.13929
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Note: the calls are **simulated** (scripted role-play), not real customer calls.

The dataset comparison and selection criteria are in [`research/dataset_survey.xlsx`](research/dataset_survey.xlsx).

## Repository Structure

| Folder | Contents |
|---|---|
| `research/` | Dataset and literature survey (Excel) |
| `notebooks/` | Ordered notebooks, one per pipeline stage |
| `src/` | Reusable Python functions used by the notebooks |
| `scripts/` | Helper scripts (for example, data download) |
| `data/` | Download instructions only; raw audio is not committed |
| `results/` | Transcripts, segmented turns, summaries and sentiment outputs |
| `docs/` | Diagrams and extra documentation |
| `presentation/` | Slides and recording link |

## Setup and Usage

To be added.
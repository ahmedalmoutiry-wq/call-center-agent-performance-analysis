# Data

This project uses the **HarperValleyBank (HVB)** corpus. The data is **not stored in this repository**: run the download script to get it.

## Source and license

- Repository: https://github.com/cricketclub/gridspace-stanford-harper-valley
- Paper: Wu, Nafziger, Scodary and Maas, *HarperValleyBank: A Domain-Specific Spoken Dialog Corpus*, 2020. https://arxiv.org/abs/2010.13929
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Redistribution is allowed with attribution. Raw audio is still kept out of Git because of its size (several GB).

## Important notes about the data

- **Simulated calls:** crowd workers role-played a bank agent and a customer from a script. These are not real customer calls.
- **Speaker-separated audio:** each call has two mono WAV files (agent and caller), 8 kHz telephone quality.
- **Reference transcripts:** `human_transcript` is written by human transcribers and is used only as the ground truth for evaluation. The `transcript` field is an older machine (ASR) output from the dataset provider and is not used as our output.
- **Emotion labels:** produced by a model (Gridspace API), not by humans, so they are treated as a noisy reference.

## How to download

Run from the project root:

```bash
# Stage 1: metadata + transcripts for all 1,446 calls (about 30 MB)
python3 scripts/download_data.py

# Stage 2: audio for the selected subset only (about 1.5 MB per call)
python3 scripts/download_data.py --audio results/subset_ids.csv
```

`results/subset_ids.csv` is created by `notebooks/01_data_exploration.ipynb`.

The script skips files that already exist, so it can be re-run safely after a network error.

**macOS note:** if you see `CERTIFICATE_VERIFY_FAILED`, Python cannot find its SSL certificates. Install them with:

```bash
python3 -m pip install --upgrade certifi
ln -sf "$(python3 -m certifi)" "$(python3 -c 'import ssl; print(ssl.get_default_verify_paths().openssl_cafile)')"
```

## Folder layout after download

```
data/raw/
├── call_ids.txt              # list of all 1,446 call IDs
├── metadata/<call_id>.json   # task, speaker IDs, quality ratings
├── transcript/<call_id>.json # utterances with timestamps, roles, emotion
└── audio/
    ├── agent/<call_id>.wav
    └── caller/<call_id>.wav
```
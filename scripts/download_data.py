"""Download the HarperValleyBank (HVB) files used in this project.

The dataset lives in a public GitHub repository (CC BY 4.0):
https://github.com/cricketclub/gridspace-stanford-harper-valley

The full audio is several gigabytes, so we download in two stages:

    # Stage 1: metadata + transcripts for ALL calls (small JSON files)
    python scripts/download_data.py

    # Stage 2: audio ONLY for the calls we selected in notebook 01
    python scripts/download_data.py --audio results/subset_ids.csv

Files are saved under data/raw/, which is ignored by Git (see .gitignore).
Running the script again skips files that were already downloaded.
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO = "cricketclub/gridspace-stanford-harper-valley"
BRANCH = "master"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CALL_IDS_FILE = RAW_DIR / "call_ids.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_call_ids() -> list[str]:
    """Return every call ID (sid) in the dataset.

    One request to the GitHub API lists all file paths in the repository.
    Each call has a metadata file named data/metadata/<sid>.json, so the
    file names give us the list of call IDs.
    """
    if CALL_IDS_FILE.exists():  # reuse the list from a previous run
        return CALL_IDS_FILE.read_text().split()

    print("Listing call IDs from GitHub ...")
    with urllib.request.urlopen(TREE_API, timeout=60) as response:
        tree = json.load(response)

    if tree.get("truncated"):
        sys.exit("ERROR: GitHub returned a truncated file list.")

    prefix, suffix = "data/metadata/", ".json"
    call_ids = sorted(
        item["path"][len(prefix):-len(suffix)]
        for item in tree["tree"]
        if item["path"].startswith(prefix) and item["path"].endswith(suffix)
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CALL_IDS_FILE.write_text("\n".join(call_ids) + "\n")
    return call_ids


def download_file(url: str, destination: Path, retries: int = 3) -> bool:
    """Download one file. Return True on success, False after all retries fail.

    The file is first written as <name>.part and renamed at the end, so an
    interrupted download never leaves a half-written file behind.
    """
    if destination.exists() and destination.stat().st_size > 0:
        return True  # already downloaded

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".part")

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                temp_path.write_bytes(response.read())
            temp_path.rename(destination)
            return True
        except Exception as error:  # network errors, HTTP 404, timeouts ...
            if attempt == retries:
                print(f"  FAILED {url}: {error}")
                return False
            time.sleep(2 * attempt)  # wait a little longer after each failure
    return False


def download_many(jobs: list[tuple[str, Path]], workers: int = 16) -> list[str]:
    """Download many (url, destination) pairs in parallel. Return failed URLs."""
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, url, dest): url for url, dest in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            if not future.result():
                failed.append(futures[future])
            if done % 250 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files")
    return failed


def read_subset_ids(csv_path: Path) -> list[str]:
    """Read the call IDs from the 'call_id' column of a CSV file."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "call_id" not in (reader.fieldnames or []):
            sys.exit(f"ERROR: {csv_path} has no 'call_id' column.")
        return [row["call_id"] for row in reader]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, metavar="CSV",
                        help="CSV with a call_id column: download audio for these calls only")
    args = parser.parse_args()

    if args.audio:
        call_ids = read_subset_ids(args.audio)
        jobs = [
            (f"{RAW_BASE}/audio/{side}/{sid}.wav", RAW_DIR / "audio" / side / f"{sid}.wav")
            for sid in call_ids
            for side in ("agent", "caller")
        ]
        print(f"Downloading audio for {len(call_ids)} calls ({len(jobs)} files) ...")
    else:
        call_ids = list_call_ids()
        jobs = [
            (f"{RAW_BASE}/{kind}/{sid}.json", RAW_DIR / kind / f"{sid}.json")
            for sid in call_ids
            for kind in ("metadata", "transcript")
        ]
        print(f"Downloading metadata + transcripts for {len(call_ids)} calls ({len(jobs)} files) ...")

    failed = download_many(jobs)

    if failed:
        print(f"\n{len(failed)} files failed. Run the same command again to retry them.")
        sys.exit(1)
    print(f"\nDone. Files are in {RAW_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()

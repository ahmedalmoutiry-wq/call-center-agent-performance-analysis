"""Speech-to-text with Whisper (faster-whisper implementation).

faster-whisper runs OpenAI's Whisper models with the CTranslate2 engine:
same model weights, faster and lighter than the original implementation.
"""

import time
from pathlib import Path

# Settings used for every transcription (kept in one place for the README)
ASR_SETTINGS = {
    "language": "en",                      # the dataset is English: skip language detection
    "beam_size": 5,                        # keep the 5 best partial transcripts while decoding
    "vad_filter": True,                    # skip silence -> fewer hallucinated words
    "condition_on_previous_text": False,   # avoid repeating earlier mistakes in long silences
}


def preload_cuda12_libraries() -> int:
    """Load CUDA 12 cuBLAS and cuDNN from the `nvidia-*-cu12` pip packages, if installed.

    faster-whisper runs on CTranslate2 4.x, which is built for CUDA 12. Newer
    Colab images ship a newer CUDA, so `libcublas.so.12` is missing. Loading the
    CUDA 12 libraries from pip into the process (RTLD_GLOBAL) lets CTranslate2
    find them by name. Returns how many libraries were loaded.
    """
    import ctypes
    import sys

    lib_files = []
    for site_dir in sys.path:
        for package in ("cublas", "cudnn"):
            lib_files += sorted(Path(site_dir, "nvidia", package, "lib").glob("lib*.so*"))

    loaded, pending = 0, lib_files
    for _ in range(3):  # some libraries need another one loaded first, so retry the failures
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                failed.append(lib)
        pending = failed
    return loaded


def load_whisper(model_size: str):
    """Load a Whisper model on the GPU if it really works, otherwise on the CPU.

    float16 on GPU is fast and accurate; int8 on CPU is the practical fallback.
    A one-second test transcription checks the GPU, because missing CUDA
    libraries only fail when the model first runs, not when it loads.
    Returns (model, device_description).
    """
    import numpy as np
    from faster_whisper import WhisperModel  # imported here so other notebooks don't need it

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        test_segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        list(test_segments)  # forces the model to run once on the GPU
        return model, "cuda/float16"
    except Exception as error:  # no GPU, or CUDA libraries missing
        print(f"GPU not usable ({type(error).__name__}: {error}); using CPU int8 (slower).")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        return model, "cpu/int8"


def transcribe_file(model, audio_file: Path) -> tuple[list[dict], float]:
    """Transcribe one WAV file. Return (segments, seconds_taken).

    Each segment is a dict with start/end time (seconds), text, and two
    confidence signals from Whisper:
      avg_logprob    - average log-probability of the words (closer to 0 = more confident)
      no_speech_prob - probability that the segment is actually silence/noise
    """
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"{audio_file} not found. Download the subset audio first.")

    started = time.perf_counter()
    segments, _info = model.transcribe(str(audio_file), **ASR_SETTINGS)
    # `segments` is a generator: decoding actually happens while we loop over it
    rows = [{
        "start_s": round(seg.start, 2),
        "end_s": round(seg.end, 2),
        "asr_text": seg.text.strip(),
        "avg_logprob": round(seg.avg_logprob, 3),
        "no_speech_prob": round(seg.no_speech_prob, 3),
    } for seg in segments]
    return rows, time.perf_counter() - started

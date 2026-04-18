"""Central path constants. All large data lives on /nas/pro-craft."""

from pathlib import Path
import os

NAS_ROOT = Path(os.environ.get("PROCRAFT_NAS_ROOT", "/nas/pro-craft"))

RAW = NAS_ROOT / "raw"
RAW_SLAKH = RAW / "slakh2100"
RAW_GIGAMIDI = RAW / "gigamidi"
RAW_LAKH = RAW / "lakh_midi"
RAW_MIDICAPS = RAW / "midicaps"

SOUNDFONTS = NAS_ROOT / "soundfonts"
RENDERED = NAS_ROOT / "rendered"       # per-stem audio, grouped by source/track
MIXTURES = NAS_ROOT / "mixtures"       # 10s mixture clips (the original_audio entries)
TRACES = NAS_ROOT / "traces"           # Qwen3 JSONL outputs (motivation + think + tool_calls)
MODIFIED = NAS_ROOT / "modified"       # post-tool-execution audio
DATASET = NAS_ROOT / "dataset"         # final sharded dataset (train/val/test)
CACHE = NAS_ROOT / "cache"
DAC_CACHE = CACHE / "dac_tokens"
PIANOROLL_CACHE = CACHE / "pianoroll"
HF_CACHE = CACHE / "hf"
MODELS = NAS_ROOT / "models"
LOGS = NAS_ROOT / "logs"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE / "hub"))

SAMPLE_RATE = 48000
CLIP_SECONDS = 30
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

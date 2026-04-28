"""Scan generated Lakh dataset for failures.

Walks /nas/pro-craft/dataset/lakh/ and validates each ``entry.json``.
Writes the list of failed paths (one md5 per line) to a text file,
ready for re-rendering by ``regenerate_failures.py``.

Failure conditions:
    - missing entry.json
    - missing audio (original.wav / modified.wav) or MIDI sidecars
    - empty motivation
    - motivation > 800 chars OR contains chain-of-thought / prompt
      leak patterns
    - executed_ok = false
    - attempt_count > max_retries default (3)
    - audio is silent (RMS < 1e-5)

Output line format: ``<md5>\\t<reason>`` so the regenerator can either
replay the exact track OR filter by reason.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf


_LEAK_PATTERNS = [
    re.compile(r"\bokay,\s+let'?s\s+tackle\b", re.I),
    re.compile(r"\bthe\s+user\s+(?:is|wants|specifies|asks)\b", re.I),
    re.compile(r"\bPRIMARY\s+MOVE\b"),
    re.compile(r"<tool_call>"),
    re.compile(r"</?think>"),
    re.compile(r"\bI\s+(?:need\s+to|should|must|have\s+to)\b", re.I),
    re.compile(r"\bWait\b,\s*(?:the|but|no|maybe)\b", re.I),
    re.compile(r"\bLet me\s+(?:check|think|see|plan)\b", re.I),
    re.compile(r"\bemit\s+each\s+call\s+below\s+verbatim\b", re.I),
    re.compile(r"\bDROPS\s*\(.*?\):", re.I),
    re.compile(r"\bSWAPS\s*\(.*?\):", re.I),
    re.compile(r"\bADDS\s*\(.*?\):", re.I),
]


def _looks_like_leak(motivation: str) -> bool:
    if len(motivation) > 800:
        return True
    for pat in _LEAK_PATTERNS:
        if pat.search(motivation):
            return True
    return False


def _audio_is_silent(path: Path, threshold: float = 1e-5) -> bool:
    try:
        x, _ = sf.read(str(path), dtype="float32")
        if x.size == 0:
            return True
        rms = float((x ** 2).mean() ** 0.5)
        return rms < threshold
    except Exception:
        return True   # unreadable → treat as failure


def validate_entry(entry_dir: Path, max_retries: int = 3) -> str | None:
    """Return None when entry passes; an error reason string otherwise."""
    entry_json = entry_dir / "entry.json"
    if not entry_json.exists():
        return "missing_entry_json"
    try:
        e = json.loads(entry_json.read_text())
    except Exception as ex:
        return f"unparseable_entry_json:{type(ex).__name__}"

    motivation = (e.get("motivation") or "").strip()
    if not motivation:
        return "empty_motivation"
    if _looks_like_leak(motivation):
        return "motivation_leak"

    for fname in ("original.wav", "modified.wav",
                  "original.mid", "modified.mid"):
        if not (entry_dir / fname).exists():
            return f"missing_artifact:{fname}"

    if not e.get("executed_ok", False):
        return "executed_ok_false"

    if int(e.get("attempt_count", 0)) > max_retries:
        return "exhausted_retries"

    if _audio_is_silent(entry_dir / "original.wav"):
        return "original_audio_silent"
    if _audio_is_silent(entry_dir / "modified.wav"):
        return "modified_audio_silent"

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        default="/nas/pro-craft/dataset/lakh",
    )
    ap.add_argument(
        "--out",
        default="/nas/pro-craft/dataset/lakh.failures.txt",
    )
    ap.add_argument(
        "--summary",
        default="/nas/pro-craft/dataset/lakh.failures.summary.txt",
    )
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--report-every", type=int, default=2000)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        raise SystemExit(f"dataset root missing: {root}")

    failures: list[tuple[str, str]] = []   # (md5, reason)
    n_scanned = 0
    n_pass = 0
    from collections import Counter
    reason_hist: Counter[str] = Counter()

    for first_dir in sorted(root.iterdir()):
        if not first_dir.is_dir():
            continue
        for entry_dir in sorted(first_dir.iterdir()):
            if not entry_dir.is_dir():
                continue
            md5 = entry_dir.name
            reason = validate_entry(entry_dir, max_retries=args.max_retries)
            n_scanned += 1
            if reason is None:
                n_pass += 1
            else:
                failures.append((md5, reason))
                reason_hist[reason.split(":", 1)[0]] += 1
            if n_scanned % args.report_every == 0:
                print(f"[scan] scanned={n_scanned} pass={n_pass} "
                      f"fail={len(failures)}", flush=True)

    print(f"\n[scan] DONE: scanned={n_scanned} "
          f"pass={n_pass} fail={len(failures)}")
    if reason_hist:
        print("[scan] failure reasons:")
        for reason, count in reason_hist.most_common():
            print(f"  {reason:30s}  {count:>7d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for md5, reason in sorted(failures):
            f.write(f"{md5}\t{reason}\n")
    with open(args.summary, "w") as f:
        f.write(f"scanned\t{n_scanned}\n")
        f.write(f"pass\t{n_pass}\n")
        f.write(f"fail\t{len(failures)}\n")
        for reason, count in reason_hist.most_common():
            f.write(f"reason:{reason}\t{count}\n")
    print(f"[scan] wrote {args.out}  ({len(failures)} entries)")
    print(f"[scan] wrote {args.summary}")


if __name__ == "__main__":
    main()

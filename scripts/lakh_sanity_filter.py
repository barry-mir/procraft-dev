"""Apply rule-based sanity filter on top of the Jeong et al. dedup list.

Reads the dedup-kept list at /nas/pro-craft/raw/lakh_midi/kept_paths.txt
(140,427 paths after the duplicate filter) and writes a clean subset to
/nas/pro-craft/raw/lakh_midi/kept_paths_clean.txt.

Filter rules (all must pass):
    - File parses without exception.
    - Total duration >= 30 s.
    - >= 3 instruments with >= 16 notes each.
    - Not all melodic stems on GM 0 (the unset-program default that
      indicates the producer never assigned an instrument).
    - <= 2 drum-flagged stems (>=3 is an LMD tagging oddity).

Anything that fails is logged with a reason in
``kept_paths_clean.skipped.txt`` for audit.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import pretty_midi


def is_clean(midi_path: Path) -> tuple[bool, str | None]:
    """Apply the sanity filter; return (ok, reason_if_bad)."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        return False, f"parse_error:{type(e).__name__}"

    duration = pm.get_end_time()
    if duration < 30.0:
        return False, f"duration_lt_30s:{duration:.1f}"

    valid_inst = [i for i in pm.instruments if len(i.notes) >= 16]
    if len(valid_inst) < 3:
        return False, f"valid_stems_lt_3:{len(valid_inst)}"

    melodic = [i for i in valid_inst if not i.is_drum]
    # Default-program detection: a Lakh file where every melodic stem is
    # GM 0 (Acoustic Grand Piano) is almost always one where the source
    # never set program-change events; treat as broken.
    if melodic and all(i.program == 0 for i in melodic):
        return False, "all_melodic_program_0"

    # Excessive drum tags often indicate LMD tagging noise.
    drum_count = sum(1 for i in pm.instruments if i.is_drum)
    if drum_count >= 3:
        return False, f"drum_stems_gte_3:{drum_count}"

    return True, None


def _worker(args: tuple[int, str]) -> tuple[int, bool, str | None, str]:
    idx, line = args
    p = Path(line)
    ok, reason = is_clean(p)
    return idx, ok, reason, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="/nas/pro-craft/raw/lakh_midi/kept_paths.txt")
    ap.add_argument("--out", dest="out",
                    default="/nas/pro-craft/raw/lakh_midi/kept_paths_clean.txt")
    ap.add_argument("--skipped", dest="skipped",
                    default="/nas/pro-craft/raw/lakh_midi/kept_paths_clean.skipped.txt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N paths (0 = all)")
    args = ap.parse_args()

    with open(args.inp) as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        paths = paths[: args.limit]
    print(f"[sanity] {len(paths)} input paths, workers={args.workers}")

    kept = []
    skipped: list[tuple[str, str]] = []
    chunk = 2000
    with mp.Pool(args.workers) as pool:
        for i, (_, ok, reason, line) in enumerate(
            pool.imap_unordered(_worker, enumerate(paths), chunksize=64)
        ):
            if ok:
                kept.append(line)
            else:
                skipped.append((line, reason or ""))
            if (i + 1) % chunk == 0:
                print(f"[sanity] {i + 1}/{len(paths)}  kept={len(kept)}  "
                      f"skipped={len(skipped)}", flush=True)

    print(f"[sanity] DONE: kept={len(kept)} / skipped={len(skipped)} "
          f"= {len(kept) / max(1, len(paths)) * 100:.1f}% kept")

    with open(args.out, "w") as f:
        f.write("\n".join(sorted(kept)) + "\n")
    with open(args.skipped, "w") as f:
        for path, reason in sorted(skipped):
            f.write(f"{reason}\t{path}\n")

    # Reason histogram.
    from collections import Counter
    reasons = Counter(r.split(":", 1)[0] for _, r in skipped)
    print("\n[sanity] skip reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason:30s}  {count:>7d}")


if __name__ == "__main__":
    main()

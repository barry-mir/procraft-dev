"""Regenerate Lakh dataset entries that failed validation.

Reads ``lakh.failures.txt`` (md5\\treason), maps each md5 back to its
source MIDI under ``/nas/pro-craft/raw/lakh_midi/lmd_full/<first>/<md5>.mid``,
deletes the existing failed-entry directory, and re-dispatches
generation through ``scripts.generate_lakh._build_one``. The generator
is idempotent (skips tracks with a valid ``entry.json``), but we delete
explicitly so the new run starts clean.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.pipeline.trace_client import VLLMClient
from procraft_data.rendering.fluidsynth_render import find_default_soundfont

# Import the generator's per-task function so we share the same code path.
from scripts.generate_lakh import _build_one, _entry_dir_for


LAKH_RAW_ROOT = Path("/nas/pro-craft/raw/lakh_midi/lmd_full")


def _midi_path_for(md5: str) -> Path:
    return LAKH_RAW_ROOT / md5[:1] / f"{md5}.mid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--failures",
        default="/nas/pro-craft/dataset/lakh.failures.txt",
    )
    ap.add_argument("--server", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--model",
                    default="cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--tool-count-min", type=int, default=10)
    ap.add_argument("--tool-count-max", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=20)
    args = ap.parse_args()

    sf_path = find_default_soundfont()
    if sf_path is None:
        raise SystemExit("No soundfont.")

    client = VLLMClient(base_url=args.server, model=args.model)
    client.wait_ready(max_wait_sec=30)

    with open(args.failures) as f:
        rows = [ln.strip().split("\t", 1) for ln in f if ln.strip()]
    rows = [(r[0], r[1] if len(r) > 1 else "") for r in rows]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[regen] {len(rows)} failed entries to retry")

    # Wipe each failed entry dir + map md5 → MIDI source path.
    paths_to_run: list[str] = []
    missing_src = 0
    for md5, _reason in rows:
        midi = _midi_path_for(md5)
        if not midi.exists():
            missing_src += 1
            continue
        # Wipe the existing entry dir so the generator regenerates.
        entry_dir = _entry_dir_for(md5)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        paths_to_run.append(str(midi))

    if missing_src:
        print(f"[regen] WARNING: {missing_src} md5s missing from raw "
              f"Lakh corpus (filter list / source mismatch).")

    print(f"[regen] retrying {len(paths_to_run)} tracks with workers={args.workers}")
    t0 = time.time()
    n_done = 0
    n_ok = 0
    n_err = 0
    seeds = [args.seed + i for i in range(len(paths_to_run))]
    tool_count_range = (args.tool_count_min, args.tool_count_max)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(_build_one, p, s, client, sf_path,
                      paths.CLIP_SECONDS, args.max_retries, tool_count_range)
            for p, s in zip(paths_to_run, seeds)
        ]
        for fut in as_completed(futures):
            n_done += 1
            try:
                r = fut.result()
            except Exception as exc:
                n_err += 1
                print(f"  [{n_done}/{len(paths_to_run)}] FUTURE_ERROR: {exc}",
                      flush=True)
                continue
            if r["status"].startswith("ok") or r["status"] == "cached":
                n_ok += 1
            else:
                n_err += 1
            if n_done % args.report_every == 0:
                wall = time.time() - t0
                rate = n_done / max(0.001, wall)
                print(f"[regen] {n_done}/{len(paths_to_run)} "
                      f"ok={n_ok} err={n_err} rate={rate:.2f}/s",
                      flush=True)
    print(f"\n[regen] DONE. ok={n_ok} err={n_err} "
          f"wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

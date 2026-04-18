"""Batch-ingest every BabySlakh track into /nas/pro-craft/rendered/slakh/.

Writes per-stem FLACs + mix.wav + meta.json for each track, plus a top-level
``index.jsonl`` (one row per track) to simplify later dataset construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.pipeline.ingest import ingest_one
from procraft_data.rendering.fluidsynth_render import (
    FluidSynthRenderer, find_default_soundfont,
)
from procraft_data.sources.slakh import iter_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-root", default=str(paths.RAW_SLAKH / "babyslakh_16k"))
    ap.add_argument("--out-root", default=str(paths.RENDERED / "slakh"))
    ap.add_argument("--duration", type=float, default=paths.CLIP_SECONDS)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    slakh_root = Path(args.slakh_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sf_path = find_default_soundfont()
    if sf_path is None:
        raise SystemExit("No soundfont.")
    renderer = FluidSynthRenderer(sf_path, sample_rate=paths.SAMPLE_RATE)

    index_path = out_root / "index.jsonl"
    ok, failed, skipped = 0, 0, 0
    t0 = time.time()
    with index_path.open("w") as idx:
        for track in iter_tracks(slakh_root):
            out_meta = out_root / track.track_id / "meta.json"
            if args.skip_existing and out_meta.exists():
                skipped += 1
                continue
            try:
                rec = ingest_one(
                    track, renderer, out_root,
                    sample_rate=paths.SAMPLE_RATE,
                    duration_sec=args.duration,
                    source_tag=f"slakh/{slakh_root.name}",
                )
                idx.write(json.dumps(asdict(rec)) + "\n")
                idx.flush()
                ok += 1
                print(f"  {rec.track_id}: stems={len(rec.stems):2d}  "
                      f"peak_norm={rec.peak_normalize_divisor or 1.0:.2f}")
            except Exception as e:
                failed += 1
                print(f"  FAIL {track.track_id}: {e}")

    print(f"done in {time.time()-t0:.1f}s — ok={ok} failed={failed} skipped={skipped}")
    print(f"index: {index_path}")
    renderer.close()


if __name__ == "__main__":
    main()

"""Ingest one Slakh track. Wrapper around ``procraft_data.pipeline.ingest``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.pipeline.ingest import ingest_one
from procraft_data.rendering.fluidsynth_render import (
    FluidSynthRenderer, find_default_soundfont,
)
from procraft_data.sources.slakh import iter_tracks, load_track


DEFAULT_BABYSLAKH = paths.RAW_SLAKH / "babyslakh_16k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-root", default=str(DEFAULT_BABYSLAKH))
    ap.add_argument("--track", default=None,
                    help="Track id to ingest (default: first track found)")
    ap.add_argument("--start", type=float, default=None,
                    help="Window start in seconds (default: auto-centered)")
    ap.add_argument("--duration", type=float, default=paths.CLIP_SECONDS)
    ap.add_argument("--out-root", default=str(paths.RENDERED / "slakh"))
    args = ap.parse_args()

    slakh_root = Path(args.slakh_root)
    track = (load_track(slakh_root / args.track) if args.track
             else next(iter_tracks(slakh_root)))

    sf_path = find_default_soundfont()
    if sf_path is None:
        raise SystemExit("No soundfont found. Set PROCRAFT_SOUNDFONT or drop "
                         f"a .sf2 into {paths.SOUNDFONTS}/")
    renderer = FluidSynthRenderer(sf_path, sample_rate=paths.SAMPLE_RATE)

    rec = ingest_one(
        track, renderer, Path(args.out_root),
        sample_rate=paths.SAMPLE_RATE,
        start_sec=args.start, duration_sec=args.duration,
        source_tag=f"slakh/{slakh_root.name}",
    )
    print(f"wrote {args.out_root}/{rec.track_id}/  "
          f"stems={len(rec.stems)}  total={rec.midi_total_duration_sec:.1f}s  "
          f"window=[{rec.window_start_sec:.1f}s, "
          f"{rec.window_start_sec + rec.window_duration_sec:.1f}s]")
    for s in rec.stems:
        tag = "drums" if s.is_drum else f"GM{s.program:3d}"
        print(f"  {s.name:22s} {tag:8s} inst={s.inst_class!r:24s} plugin={s.plugin_name}")
    renderer.close()


if __name__ == "__main__":
    main()

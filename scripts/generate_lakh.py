"""Generate dataset entries from Lakh MIDI tracks.

Reads ``kept_paths_clean.txt`` (sanity-filtered subset of the Jeong et
al. dedup list) and dispatches generation across N workers. One random
PRIMARY_INTENT per track. Output layout:

    /nas/pro-craft/dataset/lakh/<first-char>/<md5>/
        entry.json
        original.wav   modified.wav
        original.mid   modified.mid

Idempotent: skips tracks whose ``entry.json`` already exists.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.pipeline.generate_traces import generate_one
from procraft_data.pipeline.trace_client import VLLMClient
from procraft_data.pipeline.trace_prompts import (
    PRIMARY_INTENTS, build_spec, sample_primary_intent,
)
from procraft_data.rendering.fluidsynth_render import (
    FluidSynthRenderer, find_default_soundfont,
)
from procraft_data.sources import lakh as lakh_src
from procraft_data.sources.slakh import build_mixture_state, describe_mixture


DATASET_ROOT = Path("/nas/pro-craft/dataset/lakh")


# Per-thread renderer (libfluidsynth is not thread-safe after sfload).
_TLS = threading.local()


def _get_renderer(sf_path: Path) -> FluidSynthRenderer:
    r = getattr(_TLS, "renderer", None)
    if r is None:
        r = FluidSynthRenderer(soundfont_path=sf_path)
        _TLS.renderer = r
    return r


def _entry_dir_for(md5: str) -> Path:
    return DATASET_ROOT / md5[:1] / md5


def _preview_metadata(track, duration: float) -> str:
    import pretty_midi
    longest = max(pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time()
                  for sid in track.stems)
    start = max(0.0, (longest - duration) / 2.0)
    state = build_mixture_state(track, paths.SAMPLE_RATE, start, duration)
    return describe_mixture(state, track)


def _build_one(midi_path_str: str, seed: int, client: VLLMClient,
               sf_path: Path, duration: float, max_retries: int,
               tool_count_range: tuple[int, int]) -> dict:
    rng = random.Random(seed)
    midi_path = Path(midi_path_str)
    md5 = midi_path.stem
    out_dir = _entry_dir_for(md5)

    if (out_dir / "entry.json").exists():
        return {"track_id": md5, "status": "cached", "wall": 0.0}

    t0 = time.time()
    try:
        track = lakh_src.load_track(midi_path)
        if len(track.stems) < 3:
            return {"track_id": md5, "status": "skip:too_few_stems",
                    "wall": time.time() - t0}

        intent_name = rng.choice(PRIMARY_INTENTS)
        meta = _preview_metadata(track, duration)
        intent = sample_primary_intent(
            meta, forced_intent=intent_name,
            seed=rng.randint(0, 2**31 - 1),
        )
        spec = build_spec(
            meta, intent=intent,
            tool_count_range=tool_count_range,
            seed=rng.randint(0, 2**31 - 1),
        )
        renderer = _get_renderer(sf_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        entry = generate_one(
            track, spec, client, renderer, out_dir,
            duration_sec=duration, entry_idx=0,
            withhold_for_add=None, max_retries=max_retries,
            stem_override="entry",
        )
        # generate_one writes ``entry.json`` + ``entry_original.wav`` /
        # ``entry_modified.wav`` / ``entry_original.mid`` /
        # ``entry_modified.mid``. Rename to the canonical flat layout
        # we promise on disk: original.wav, modified.wav, etc.
        for src_name, dst_name in (
            ("entry_original.wav", "original.wav"),
            ("entry_modified.wav", "modified.wav"),
            ("entry_original.mid", "original.mid"),
            ("entry_modified.mid", "modified.mid"),
        ):
            src = out_dir / src_name
            if src.exists():
                src.replace(out_dir / dst_name)
        # Patch the filename fields on the persisted entry.json so the
        # paths there match what's actually on disk.
        entry_json_path = out_dir / "entry.json"
        if entry_json_path.exists():
            blob = json.loads(entry_json_path.read_text())
            blob["original_wav"] = "original.wav"
            blob["modified_wav"] = "modified.wav"
            blob["original_midi"] = "original.mid"
            blob["modified_midi"] = "modified.mid"
            entry_json_path.write_text(json.dumps(blob, indent=2))
        return {
            "track_id": md5,
            "status": "ok" if entry.executed_ok else "ok_with_errors",
            "intent": entry.primary_intent,
            "wall": time.time() - t0,
            "n_calls": len(entry.tool_calls),
            "attempts": entry.attempt_count,
        }
    except Exception as e:
        return {"track_id": md5, "status": "error",
                "error": f"{type(e).__name__}:{e!s:.200}",
                "trace": traceback.format_exc()[-400:],
                "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paths-file",
        default="/nas/pro-craft/raw/lakh_midi/kept_paths_clean.txt",
    )
    ap.add_argument("--server", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--model",
                    default="cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N paths (0 = all).")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N paths (for sharded runs).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=paths.CLIP_SECONDS)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--tool-count-min", type=int, default=10)
    ap.add_argument("--tool-count-max", type=int, default=15)
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args()

    sf_path = find_default_soundfont()
    if sf_path is None:
        raise SystemExit("No soundfont.")

    client = VLLMClient(base_url=args.server, model=args.model)
    client.wait_ready(max_wait_sec=30)

    with open(args.paths_file) as f:
        all_paths = [ln.strip() for ln in f if ln.strip()]
    paths_list = all_paths[args.offset: args.offset + args.limit] if args.limit else all_paths[args.offset:]
    print(f"[lakh-gen] {len(paths_list)} paths "
          f"(offset={args.offset}, limit={args.limit or 'all'})")
    print(f"[lakh-gen] workers={args.workers}, "
          f"tool_count={args.tool_count_min}-{args.tool_count_max}, "
          f"out={DATASET_ROOT}")

    rng = random.Random(args.seed)
    seeds = [rng.randint(0, 2**31 - 1) for _ in paths_list]
    tool_count_range = (args.tool_count_min, args.tool_count_max)

    n_done = 0
    n_ok = 0
    n_cached = 0
    n_err = 0
    sum_wall_per_entry = 0.0
    t_run = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(_build_one, p, s, client, sf_path,
                      args.duration, args.max_retries, tool_count_range)
            for p, s in zip(paths_list, seeds)
        ]
        for fut in as_completed(futures):
            n_done += 1
            try:
                r = fut.result()
            except Exception as e:
                n_err += 1
                print(f"  [{n_done}/{len(paths_list)}] FUTURE_ERROR: {e}",
                      flush=True)
                continue
            status = r["status"]
            if status.startswith("ok"):
                n_ok += 1
                sum_wall_per_entry += r["wall"]
            elif status == "cached":
                n_cached += 1
            else:
                n_err += 1
            if n_done % args.report_every == 0:
                wall = time.time() - t_run
                rate = n_done / wall if wall > 0 else 0
                eta = (len(paths_list) - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"[lakh-gen] {n_done}/{len(paths_list)}  "
                    f"ok={n_ok} cached={n_cached} err={n_err}  "
                    f"rate={rate:.2f} entries/s  "
                    f"avg-per-entry={(sum_wall_per_entry / max(1, n_ok)):.1f}s  "
                    f"eta={eta / 3600:.2f}h",
                    flush=True,
                )

    wall = time.time() - t_run
    print(
        f"\n[lakh-gen] DONE. processed={n_done}  ok={n_ok}  cached={n_cached}  "
        f"err={n_err}  wall={wall:.1f}s  "
        f"avg-wall-per-entry={(wall / max(1, n_done)):.1f}s"
    )


if __name__ == "__main__":
    main()

"""Smoke test: run one Qwen3 generation per role on BabySlakh Track00001.

Parallelism:
- vLLM server runs without --enforce-eager, with --max-num-seqs 6 to enable
  continuous batching of up to 6 concurrent requests.
- This script dispatches N worker threads (default 4). Each thread owns its
  own ``FluidSynthRenderer`` (libfluidsynth is NOT thread-safe after sfload),
  shares the single ``VLLMClient`` (``requests`` is thread-safe), and runs
  the full per-entry generate_one path.

Prereq: vLLM server must be running. Start with:

    bash scripts/serve_qwen3.sh

Then:

    python scripts/smoke_qwen_all_motivations.py
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.pipeline.generate_traces import generate_one
from procraft_data.pipeline.trace_client import VLLMClient
from procraft_data.pipeline.trace_prompts import smoke_plan
from procraft_data.rendering.fluidsynth_render import (
    FluidSynthRenderer, find_default_soundfont,
)
from procraft_data.sources.slakh import (
    build_mixture_state, describe_mixture, load_track,
)


def _preview_mixture_metadata(track, duration: float) -> str:
    import pretty_midi
    longest = max(pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time()
                  for sid in track.stems)
    start = max(0.0, (longest - duration) / 2.0)
    state = build_mixture_state(track, paths.SAMPLE_RATE, start, duration)
    return describe_mixture(state, track)


# Thread-local FluidSynthRenderer: each worker gets its own instance, created
# lazily on first use. libfluidsynth holds state in the Synth struct that is
# not safe to share across threads once sfload() has been called.
_thread_local = threading.local()


def _get_renderer(soundfont_path: Path, sample_rate: int) -> FluidSynthRenderer:
    r = getattr(_thread_local, "renderer", None)
    if r is None:
        r = FluidSynthRenderer(soundfont_path, sample_rate=sample_rate)
        _thread_local.renderer = r
    return r


def _close_thread_renderer():
    r = getattr(_thread_local, "renderer", None)
    if r is not None:
        r.close()
        _thread_local.renderer = None


def _generate_one_thread(idx: int, spec, track, args, out_dir: Path,
                         client: VLLMClient, soundfont_path: Path):
    renderer = _get_renderer(soundfont_path, paths.SAMPLE_RATE)
    # Random withhold is OFF until intent-guided generation lands — otherwise
    # a silently-withheld track never gets added back and the original/modified
    # pair is mismatched (see review of iter5 case 12).
    t0 = time.time()
    try:
        entry = generate_one(
            track, spec, client, renderer, out_dir,
            duration_sec=args.duration,
            entry_idx=idx, withhold_for_add=None,
        )
        return idx, entry, None, time.time() - t0
    except Exception as e:
        return idx, None, repr(e), time.time() - t0


def _print_entry_block(idx: int, plan_len: int, spec, entry, err: str | None,
                       wall_time: float):
    print("=" * 78)
    print(f"[{idx+1}/{plan_len}] intent={spec.primary_intent}  role={spec.role}  "
          f"abstraction={spec.abstraction_level}  "
          f"temp={spec.temperature}  tool_count={spec.tool_count_hint}")
    if spec.target_track:
        prog = f" program={spec.target_program}" if spec.target_program is not None else ""
        print(f"  primary_tool={spec.primary_tool}  target={spec.target_track!r}{prog}")
    else:
        print(f"  primary_tool={spec.primary_tool}")
    print(f"  hook: {spec.hook}")
    if err is not None:
        print(f"  FAILED after {wall_time:.1f}s: {err}")
        return
    print(f"\n<think>")
    print(textwrap.indent((entry.think or "(no reasoning)")[:800], "  "))
    if entry.think and len(entry.think) > 800:
        print(f"  ... [truncated, {len(entry.think)} chars total]")
    print(f"\nmotivation: {entry.motivation or '(empty)'}")
    print(f"\ntool_calls ({len(entry.tool_calls)}):")
    for tc, eff in zip(entry.tool_calls, entry.tool_effects):
        name = tc.get("name")
        args_str = str(tc.get("arguments", {}))[:140]
        mix_d = eff.get("mix_rms_delta", 0.0)
        trk_d = eff.get("track_rms_delta")
        status = eff.get("status", "?")
        err_str = (" err=" + eff["error"]) if eff.get("error") else ""
        trk_str = f" Δtrack={trk_d:.4f}" if isinstance(trk_d, float) else ""
        print(f"  - [{status}] Δmix={mix_d:.4f}{trk_str}  {name}: {args_str}{err_str}")
    exec_str = "ok" if entry.executed_ok else f"errors={entry.executed_errors}"
    pm = "✓" if entry.primary_move_executed else "✗ MISSING"
    print(f"\nexecuted: {exec_str}  primary_move={pm}  "
          f"mix_rms orig={entry.mix_rms_original:.4f} mod={entry.mix_rms_modified:.4f} "
          f"Δ={entry.mix_rms_delta:.4f}  "
          f"usage={entry.usage.get('completion_tokens','?')}tok  "
          f"latency={entry.latency_sec:.1f}s  wall={wall_time:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-root", default=str(paths.RAW_SLAKH / "babyslakh_16k"))
    ap.add_argument("--track", default="Track00001")
    ap.add_argument("--out-dir", default=str(paths.TRACES / "smoke" / "Track00001"))
    ap.add_argument("--server", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--model",
                    default="cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit")
    ap.add_argument("--duration", type=float, default=paths.CLIP_SECONDS)
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of concurrent LLM requests. vLLM must be "
                         "started with --max-num-seqs >= this value.")
    args = ap.parse_args()

    track = load_track(Path(args.slakh_root) / args.track)
    meta_str = _preview_mixture_metadata(track, args.duration)
    print(f"[track] {track.track_id}  {meta_str}\n")

    plan = smoke_plan(meta_str, seed=42)
    client = VLLMClient(base_url=args.server, model=args.model)
    client.wait_ready(max_wait_sec=10)

    sfp = find_default_soundfont()
    if sfp is None:
        raise SystemExit("No soundfont.")

    out_dir = Path(args.out_dir)
    print(f"[out]   {out_dir}  (workers={args.workers})\n")

    # Preserve original order in the printed output but dispatch in parallel.
    results: dict[int, tuple] = {}
    t_run = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_generate_one_thread, idx, spec, track, args, out_dir,
                      client, sfp): idx
            for idx, spec in enumerate(plan)
        }
        for fut in as_completed(futures):
            idx, entry, err, wall = fut.result()
            results[idx] = (entry, err, wall)
            spec = plan[idx]
            tag = "OK " if err is None else "ERR"
            print(f"[done {idx+1}/{len(plan)} {tag}] role={spec.role} "
                  f"abstraction={spec.abstraction_level} wall={wall:.1f}s")

    total_wall = time.time() - t_run
    print(f"\nall entries done in {total_wall:.1f}s (wall).\n")

    for idx, spec in enumerate(plan):
        entry, err, wall = results[idx]
        _print_entry_block(idx, len(plan), spec, entry, err, wall)

    # Close any renderers created by worker threads we started. The executor
    # already exited, but libfluidsynth keeps its own resources until explicit
    # close — safe to skip; they'll be GC'd on process exit.


if __name__ == "__main__":
    main()

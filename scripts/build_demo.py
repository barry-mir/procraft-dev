"""Generate the ProCraft demo page.

Picks 10 different BabySlakh tracks, samples one random production
motivation per track (different role/abstraction/hook/intent each),
generates the original + modified audio pair, transcodes both WAVs to
MP3, and emits a static HTML page with all 10 cases.

Output lives under ``demo/`` in the repo root so it can be committed
and served from GitHub Pages.

Prereqs:
- vLLM server running on http://127.0.0.1:8765
- conda env ``pro-craft`` active
- ffmpeg available on PATH (or in /home/barrycheng/anaconda3/envs/mixing/bin)
"""

from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import subprocess
import sys
import threading
import time
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
from procraft_data.sources.slakh import (
    build_mixture_state, describe_mixture, load_track,
)


DEFAULT_TRACKS = [f"Track{i:05d}" for i in range(1, 11)]


# Thread-local FluidSynth
_tl = threading.local()


def _get_renderer(sf_path: Path) -> FluidSynthRenderer:
    r = getattr(_tl, "renderer", None)
    if r is None:
        r = FluidSynthRenderer(sf_path, sample_rate=paths.SAMPLE_RATE)
        _tl.renderer = r
    return r


def _ffmpeg_path() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    bundled = "/home/barrycheng/anaconda3/envs/mixing/bin/ffmpeg"
    if Path(bundled).exists():
        return bundled
    raise RuntimeError("ffmpeg not found")


def _wav_to_mp3(src: Path, dst: Path, bitrate: str = "192k") -> None:
    ffmpeg = _ffmpeg_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-codec:a", "libmp3lame", "-b:a", bitrate,
        str(dst),
    ], check=True)


def _preview_metadata(track, duration: float) -> str:
    import pretty_midi
    longest = max(pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time()
                  for sid in track.stems)
    start = max(0.0, (longest - duration) / 2.0)
    state = build_mixture_state(track, paths.SAMPLE_RATE, start, duration)
    return describe_mixture(state, track)


def _build_one(track_id: str, slakh_root: Path, seed: int,
               client: VLLMClient, sf_path: Path,
               work_dir: Path, duration: float) -> dict:
    renderer = _get_renderer(sf_path)
    track = load_track(slakh_root / track_id)
    meta = _preview_metadata(track, duration)

    rng = random.Random(seed)
    # Random intent per track (uniform over 10)
    intent_name = rng.choice(PRIMARY_INTENTS)
    intent = sample_primary_intent(
        meta, forced_intent=intent_name,
        seed=rng.randint(0, 2**31 - 1),
    )
    spec = build_spec(
        meta,
        intent=intent,
        tool_count_range=(4, 7),
        seed=rng.randint(0, 2**31 - 1),
    )

    t0 = time.time()
    entry = generate_one(
        track, spec, client, renderer, work_dir,
        duration_sec=duration,
        entry_idx=0, withhold_for_add=None,
    )
    wall = time.time() - t0
    return {
        "track_id": track_id,
        "entry": entry,
        "wall": wall,
    }


def _render_html(cases: list[dict]) -> str:
    cards = []
    for i, c in enumerate(cases, 1):
        e = c["entry"]
        orig_rel = c["original_mp3_rel"]
        mod_rel = c["modified_mp3_rel"]
        tool_lines = []
        for tc in e.tool_calls:
            name = tc["name"]
            a = tc.get("arguments", {})
            if name == "apply_fx":
                t = a.get("track") or a.get("track_name") or "?"
                call = a.get("call", {})
                eff = call.get("effect", "?")
                params = json.dumps(call.get("params", {}), separators=(",", ":"))
                tool_lines.append(
                    f"apply_fx({html.escape(t)}, {html.escape(eff)}, {html.escape(params)})"
                )
            else:
                t = a.get("track") or a.get("track_name") or "?"
                arg_str = json.dumps({k: v for k, v in a.items() if k not in ("track", "track_name")},
                                     separators=(",", ":"))
                tool_lines.append(
                    f"{html.escape(name)}({html.escape(str(t))}, {html.escape(arg_str)})"
                )
        tools_html = "<br>".join(tool_lines) or "<em>none</em>"

        cards.append(f"""
<section class="card">
  <header>
    <span class="idx">Case {i:02d}</span>
    <span class="track">{html.escape(c['track_id'])}</span>
    <span class="intent">{html.escape(e.primary_intent)}</span>
    <span class="role">{html.escape(e.role)}</span>
  </header>
  <p class="motivation">{html.escape(e.motivation)}</p>
  <div class="players">
    <div class="player">
      <div class="label">Original</div>
      <audio controls preload="none" src="{html.escape(orig_rel)}"></audio>
    </div>
    <div class="player">
      <div class="label">Modified</div>
      <audio controls preload="none" src="{html.escape(mod_rel)}"></audio>
    </div>
  </div>
  <details class="toolcalls">
    <summary>tool_calls ({len(e.tool_calls)})</summary>
    <code>{tools_html}</code>
  </details>
</section>
""")

    css = """
:root {
  --bg: #fbfaf7;
  --fg: #1b1b1b;
  --muted: #7a7a7a;
  --accent: #b94f1f;
  --line: #e6e4de;
  --card-bg: #ffffff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.55;
  font-size: 15px;
}
.wrap { max-width: 820px; margin: 0 auto; padding: 48px 24px 96px; }
h1 { font-size: 28px; font-weight: 600; margin: 0 0 8px; letter-spacing: -0.01em; }
.lede { color: var(--muted); margin: 0 0 40px; }
.card {
  background: var(--card-bg);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 20px 22px;
  margin: 0 0 18px;
}
.card header {
  display: flex; gap: 10px; align-items: baseline;
  font-size: 12px; color: var(--muted); margin-bottom: 12px;
  flex-wrap: wrap;
}
.idx { color: var(--accent); font-weight: 600; letter-spacing: 0.03em; }
.track, .intent, .role { font-family: 'SF Mono', ui-monospace, monospace; }
.intent { background: #f0eee8; padding: 1px 8px; border-radius: 4px; }
.motivation {
  font-size: 16px;
  margin: 0 0 16px;
  color: var(--fg);
}
.players { display: flex; gap: 18px; flex-wrap: wrap; }
.player { flex: 1 1 340px; min-width: 300px; }
.label {
  font-size: 11px; color: var(--muted); margin-bottom: 4px;
  text-transform: uppercase; letter-spacing: 0.08em;
}
audio { width: 100%; }
.toolcalls { margin-top: 12px; font-size: 12px; }
.toolcalls summary { cursor: pointer; color: var(--muted); }
.toolcalls code {
  display: block; font-family: 'SF Mono', ui-monospace, monospace;
  white-space: pre-wrap; word-break: break-word;
  padding: 8px 10px; background: #faf9f6; border-radius: 4px;
  margin-top: 6px; color: #444;
}
footer {
  margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 12px;
}
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ProCraft — 10-case audio demo</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
  <h1>ProCraft — 10-case audio demo</h1>
  <p class="lede">
    Ten BabySlakh tracks, each paired with an LLM-generated production
    motivation and the resulting modified mix. Every case was produced end-to-end
    by the ProCraft-Data pipeline: role × abstraction × hook × intent sampling,
    Qwen3-30B-A3B-Thinking generation, MultiAFX tool execution, and
    shared-peak-normalized stereo rendering at 48&nbsp;kHz.
  </p>
  {''.join(cards)}
  <footer>
    30&nbsp;s center-cropped clips. Audio compressed to 192&nbsp;kbps MP3 for loading speed.
  </footer>
</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-root", default=str(paths.RAW_SLAKH / "babyslakh_16k"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "demo"))
    ap.add_argument("--server", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--model",
                    default="cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit")
    ap.add_argument("--duration", type=float, default=paths.CLIP_SECONDS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    work_dir = out_dir / "_work"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    slakh_root = Path(args.slakh_root)
    sf_path = find_default_soundfont()
    if sf_path is None:
        raise SystemExit("No soundfont.")

    client = VLLMClient(base_url=args.server, model=args.model)
    client.wait_ready(max_wait_sec=10)

    rng = random.Random(args.seed)
    seeds = [rng.randint(0, 2**31 - 1) for _ in args.tracks]
    print(f"Generating {len(args.tracks)} demo cases…")
    t0 = time.time()
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_build_one, tid, slakh_root, s, client, sf_path,
                      work_dir, args.duration): i
            for i, (tid, s) in enumerate(zip(args.tracks, seeds))
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
                results[i] = r
                e = r["entry"]
                print(f"  done [{i+1}/{len(args.tracks)}] {r['track_id']}: "
                      f"intent={e.primary_intent} role={e.role} wall={r['wall']:.1f}s")
            except Exception as exc:
                print(f"  FAIL {i}: {exc}")

    print(f"\nGeneration wall: {time.time()-t0:.1f}s")

    # Transcode WAVs → MP3 and collect cases in order.
    cases: list[dict] = []
    for i in range(len(args.tracks)):
        if i not in results:
            continue
        r = results[i]
        e = r["entry"]
        # generate_one writes under work_dir using the track_id folder; find the pair.
        src_orig = work_dir / e.original_wav
        src_mod = work_dir / e.modified_wav
        if not src_orig.exists() or not src_mod.exists():
            # fall back: search any wav whose name matches
            candidates = list(work_dir.glob(f"*{r['track_id']}*original.wav"))
            if candidates:
                src_orig = candidates[0]
            candidates = list(work_dir.glob(f"*{r['track_id']}*modified.wav"))
            if candidates:
                src_mod = candidates[0]

        out_orig = audio_dir / f"case_{i+1:02d}_{r['track_id']}_original.mp3"
        out_mod = audio_dir / f"case_{i+1:02d}_{r['track_id']}_modified.mp3"
        _wav_to_mp3(src_orig, out_orig)
        _wav_to_mp3(src_mod, out_mod)
        r["original_mp3_rel"] = f"audio/{out_orig.name}"
        r["modified_mp3_rel"] = f"audio/{out_mod.name}"
        cases.append(r)

    # Render HTML.
    html_out = _render_html(cases)
    (out_dir / "index.html").write_text(html_out)

    # Save an index JSON for reproducibility.
    manifest = [
        {
            "track_id": c["track_id"],
            "role": c["entry"].role,
            "abstraction_level": c["entry"].abstraction_level,
            "hook": c["entry"].hook,
            "primary_intent": c["entry"].primary_intent,
            "primary_tool": c["entry"].primary_tool,
            "target_track": c["entry"].target_track,
            "motivation": c["entry"].motivation,
            "tool_calls": c["entry"].tool_calls,
            "original_mp3": c["original_mp3_rel"],
            "modified_mp3": c["modified_mp3_rel"],
            "attempt_count": c["entry"].attempt_count,
            "executed_ok": c["entry"].executed_ok,
        }
        for c in cases
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Clean up intermediate WAVs (keep only MP3s in the published demo).
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nDemo built at {out_dir}/")
    print(f"  {len(cases)} cases, {len(list(audio_dir.glob('*.mp3')))} MP3s")
    total_mb = sum(p.stat().st_size for p in audio_dir.glob('*.mp3')) / 1e6
    print(f"  total audio: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()

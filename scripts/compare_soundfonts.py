"""Render ONE reference Slakh track with every available soundfont and
build a static comparison page under ``docs/soundfonts/``.

Pipeline:
- For each ``*.sf2`` in ``/nas/pro-craft/soundfonts/``, spin up a
  FluidSynthRenderer, render every stem of the reference track for a
  30 s center-cropped window, mix-down, peak-normalize, transcode to
  MP3 at 192 kbps.
- Emit ``docs/soundfonts/index.html`` with one card per soundfont
  (name / size / license-tier / player) so the user can A/B compare.
- Also emit ``docs/soundfonts/manifest.json`` with per-file metadata.

The page is reachable at GitHub Pages as
``https://<user>.github.io/<repo>/soundfonts/``.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.rendering.fluidsynth_render import FluidSynthRenderer
from procraft_data.sources.slakh import build_mixture_state, load_track
from procraft_data.tools.executors import _mixdown


# Human-friendly metadata keyed by filename stem. Size / license tier is
# display-only — the render process doesn't care.
SOUNDFONT_META: dict[str, dict] = {
    "FluidR3_GM":                      {"display": "FluidR3_GM (current default)", "license": "MIT",          "tier": "A"},
    "GeneralUser-GS-v1.471":           {"display": "GeneralUser GS 1.471",         "license": "Permissive",   "tier": "A"},
    "MuseScore_General":               {"display": "MuseScore_General 0.2",        "license": "MIT",          "tier": "A"},
    "Unison":                          {"display": "Unison",                       "license": "CC0",          "tier": "A"},
    "Jnsgm2":                          {"display": "Jnsgm2",                       "license": "CC0",          "tier": "A"},
    "Arachno":                         {"display": "Arachno 1.0",                  "license": "Freeware",     "tier": "B"},
    "MagicSF":                         {"display": "MagicSFver2",                  "license": "Freeware",     "tier": "B"},
    "airfont_380":                     {"display": "Airfont 380 Final",            "license": "Public Domain","tier": "A"},
    "OPL3-FM":                         {"display": "OPL-3 FM 128M (SB16)",         "license": "Freeware",     "tier": "B"},
    "SGM-v2.01":                       {"display": "SGM v2.01",                    "license": "CC-BY 3.0",    "tier": "A"},
    "TimbresOfHeaven_3.4":             {"display": "Timbres of Heaven 3.4 (raw)",  "license": "Author-free",  "tier": "A"},
    "TimbresOfHeaven_3.4_vorbis":      {"display": "Timbres of Heaven 3.4 (SF3/Vorbis)", "license": "Author-free", "tier": "A"},
    "Musyng-Kite":                     {"display": "Musyng Kite (1 GB composite)", "license": "WTFPL*",       "tier": "B"},
    "default-GM":                      {"display": "Debian fluid-soundfont-gm",    "license": "MIT (FluidR3 mirror)", "tier": "A"},
    "TimGM6mb":                        {"display": "TimGM6mb (tiny)",              "license": "GPL",          "tier": "A"},
}


def _human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    bundled = "/home/barrycheng/anaconda3/envs/mixing/bin/ffmpeg"
    if Path(bundled).exists():
        return bundled
    raise RuntimeError("ffmpeg not found")


def _render_with_soundfont(sf_path: Path, track, duration: float) -> np.ndarray:
    """Render every stem in ``track`` with ``sf_path``, return peak-normalized mix."""
    import pretty_midi
    longest = max(pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time()
                  for sid in track.stems)
    start = max(0.0, (longest - duration) / 2.0)
    state = build_mixture_state(track, paths.SAMPLE_RATE, start, duration)

    renderer = FluidSynthRenderer(sf_path, sample_rate=paths.SAMPLE_RATE)
    try:
        for ts in state.tracks.values():
            ts.audio = renderer.render_track(ts, duration)
    finally:
        renderer.close()

    mix = _mixdown(state)
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 0.95:
        mix = mix / peak * 0.95
    return mix


def _write_wav_and_mp3(audio: np.ndarray, wav_path: Path, mp3_path: Path,
                       sample_rate: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio.T, sample_rate)
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)],
        check=True,
    )


def _render_html(cases: list[dict], track_id: str) -> str:
    cards = []
    for i, c in enumerate(cases, 1):
        size_str = _human_size(c["size_bytes"])
        tier = c.get("tier", "?")
        license_ = c.get("license", "?")
        cards.append(f"""
<section class="card">
  <header>
    <span class="idx">{i:02d}</span>
    <span class="name">{html.escape(c['display'])}</span>
    <span class="size">{size_str}</span>
    <span class="tier tier-{tier.lower()}">tier {tier}</span>
    <span class="license">{html.escape(license_)}</span>
  </header>
  <audio controls preload="none" src="{html.escape(c['mp3_rel'])}"></audio>
  <p class="filename"><code>{html.escape(c['filename'])}</code></p>
</section>
""")

    css = """
:root {
  --bg: #fbfaf7; --fg: #1b1b1b; --muted: #7a7a7a;
  --accent: #b94f1f; --line: #e6e4de; --card: #fff;
  --tier-a: #2e7d32; --tier-b: #e17a00;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
  background: var(--bg); color: var(--fg);
  line-height: 1.55; font-size: 15px;
}
.wrap { max-width: 780px; margin: 0 auto; padding: 48px 24px 96px; }
h1 { font-size: 26px; font-weight: 600; margin: 0 0 8px; letter-spacing: -0.01em; }
.lede { color: var(--muted); margin: 0 0 32px; }
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 16px 20px; margin: 0 0 12px;
}
.card header {
  display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap;
  margin-bottom: 10px;
}
.idx { color: var(--accent); font-weight: 600; font-size: 13px; }
.name { font-weight: 600; flex: 1; }
.size, .license { color: var(--muted); font-size: 12px; font-family: 'SF Mono', ui-monospace, monospace; }
.tier {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
  padding: 2px 8px; border-radius: 3px; font-weight: 600;
}
.tier-a { background: #e7f2e8; color: var(--tier-a); }
.tier-b { background: #fdf1de; color: var(--tier-b); }
audio { width: 100%; }
.filename { margin: 8px 0 0; }
.filename code {
  font-size: 11px; color: var(--muted);
  font-family: 'SF Mono', ui-monospace, monospace;
}
.back {
  display: inline-block; margin-bottom: 24px; color: var(--muted);
  text-decoration: none; font-size: 13px;
}
.back:hover { color: var(--accent); }
footer {
  margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 12px;
}
"""
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ProCraft — Soundfont Comparison</title>
<style>{css}</style>
</head><body>
<div class="wrap">
  <a class="back" href="../">&larr; back to 10-case demo</a>
  <h1>Soundfont comparison</h1>
  <p class="lede">
    Same Slakh reference track ({html.escape(track_id)}, 30&nbsp;s center-cropped,
    same mixing / peak-normalize pipeline) rendered with each soundfont. A/B
    by ear to pick which one(s) the main dataset should use. Tier A = clean
    redistribution license; Tier B = freeware / mixed-source (OK for internal
    rendering, check before public release).
  </p>
  {''.join(cards)}
  <footer>
    {len(cases)} soundfonts · 48&nbsp;kHz stereo · 192&nbsp;kbps MP3
  </footer>
</div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-root", default=str(paths.RAW_SLAKH / "babyslakh_16k"))
    ap.add_argument("--track-id", default="Track00001")
    ap.add_argument("--soundfont-dir", default=str(paths.SOUNDFONTS))
    ap.add_argument("--out-dir",
                    default=str(Path(__file__).resolve().parents[1]
                                / "docs" / "soundfonts"))
    ap.add_argument("--duration", type=float, default=paths.CLIP_SECONDS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    work_dir = out_dir / "_work"
    audio_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    track = load_track(Path(args.slakh_root) / args.track_id)
    sf_dir = Path(args.soundfont_dir)
    # Resolve symlinks to their targets and prefer real files.
    candidates = sorted(sf_dir.glob("*.sf2"))
    print(f"Found {len(candidates)} soundfonts under {sf_dir}:")
    for p in candidates:
        real = p.resolve()
        size = real.stat().st_size if real.exists() else 0
        print(f"  - {p.name}  ({_human_size(size)})")

    cases: list[dict] = []
    for i, p in enumerate(candidates):
        stem = p.stem
        meta = SOUNDFONT_META.get(stem, {
            "display": stem, "license": "unknown", "tier": "?"
        })
        print(f"\n[{i+1}/{len(candidates)}] rendering {stem}…")
        t0 = time.time()
        try:
            mix = _render_with_soundfont(p.resolve(), track, args.duration)
        except Exception as e:
            print(f"   FAILED: {e}")
            continue
        wav = work_dir / f"{stem}.wav"
        mp3 = audio_dir / f"{stem}.mp3"
        _write_wav_and_mp3(mix, wav, mp3, paths.SAMPLE_RATE)
        size = p.resolve().stat().st_size
        print(f"   done in {time.time()-t0:.1f}s  mp3={_human_size(mp3.stat().st_size)}")
        cases.append({
            "filename": p.name,
            "display": meta["display"],
            "license": meta["license"],
            "tier": meta["tier"],
            "size_bytes": size,
            "mp3_rel": f"audio/{mp3.name}",
        })

    # Sort: Tier A first, then by size ascending for consistent presentation
    cases.sort(key=lambda c: (c["tier"] != "A", c["size_bytes"]))

    html_out = _render_html(cases, args.track_id)
    (out_dir / "index.html").write_text(html_out)
    (out_dir / "manifest.json").write_text(json.dumps(cases, indent=2))

    shutil.rmtree(work_dir, ignore_errors=True)

    total_mb = sum((audio_dir / c["mp3_rel"].split("/")[-1]).stat().st_size
                   for c in cases) / 1e6
    print(f"\nBuilt {out_dir}/ with {len(cases)} soundfont cards "
          f"({total_mb:.1f} MB audio).")


if __name__ == "__main__":
    main()

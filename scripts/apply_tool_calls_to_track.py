"""Validate the full path: Slakh track → MixtureState → tool_calls → modified audio.

Feeds a canned Hermes-format response (as if from Qwen3) into the parser and
executors, then writes ``original.wav``, ``modified.wav``, and the resolved
tool-call log side-by-side so we can listen and diff.

Usage:
    python scripts/apply_tool_calls_to_track.py                         # Track00001, canned script
    python scripts/apply_tool_calls_to_track.py --track Track00003      # pick a different track
    python scripts/apply_tool_calls_to_track.py --response path.txt     # real Qwen3 output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.rendering.fluidsynth_render import (
    FluidSynthRenderer, find_default_soundfont,
)
from procraft_data.sources.slakh import (
    build_mixture_state, describe_mixture, load_track,
)
from procraft_data.tools.executors import EXECUTORS, _mixdown
from procraft_data.tools.parse import parse_response


CANNED_RESPONSE = """<think>
This track has piano plus multiple guitars and organs. The distorted guitars
are dominating the midrange. A gentle low-shelf cut on the mix tames the
mud, then compressing the drums tightens the rhythm section. Finally, I'll
add some air with a high shelf boost on the piano.
</think>

Production motivation: tighten the rhythm section and declutter the midrange.

<tool_call>
{"name": "apply_fx", "arguments": {"track": "drums", "call": {"effect": "sox_compand", "params": {"attack_time": 0.005, "decay_time": 0.15, "soft_knee_db": 6.0}}}}
</tool_call>
<tool_call>
{"name": "apply_fx", "arguments": {"track": "mix", "call": {"effect": "sox_equalizer", "params": {"frequency": 250.0, "width_q": 1.2, "gain_db": -4.0}}}}
</tool_call>
<tool_call>
{"name": "apply_fx", "arguments": {"track": "piano", "call": {"effect": "ta_treble_biquad", "params": {"gain_db": 3.0, "central_freq": 5000.0, "Q": 0.8}}}}
</tool_call>
"""


def _peak_normalize(x: np.ndarray, headroom: float = 0.99) -> np.ndarray:
    p = float(np.max(np.abs(x))) or 1.0
    return x / p * headroom if p > headroom else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-root", default=str(paths.RAW_SLAKH / "babyslakh_16k"))
    ap.add_argument("--track", default="Track00001")
    ap.add_argument("--duration", type=float, default=paths.CLIP_SECONDS)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--response", type=str, default=None,
                    help="Path to a Qwen3-style response file. Uses a canned "
                         "response if omitted.")
    ap.add_argument("--out-root", default=str(paths.MODIFIED / "slakh"))
    args = ap.parse_args()

    track = load_track(Path(args.slakh_root) / args.track)
    import pretty_midi
    longest = max(pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time()
                  for sid in track.stems)
    start = args.start if args.start is not None else max(0.0, (longest - args.duration) / 2.0)

    state = build_mixture_state(track, paths.SAMPLE_RATE, start, args.duration)
    print(f"[input]  {describe_mixture(state, track)}")

    sf_path = find_default_soundfont()
    if sf_path is None:
        raise SystemExit("No soundfont.")
    renderer = FluidSynthRenderer(sf_path, sample_rate=paths.SAMPLE_RATE)

    for ts in state.tracks.values():
        ts.audio = renderer.render_track(ts, args.duration)

    original_mix = _mixdown(state)

    if args.response:
        text = Path(args.response).read_text()
    else:
        text = CANNED_RESPONSE
    parsed = parse_response(text)
    if not parsed.is_valid():
        raise SystemExit(f"no valid tool calls in response\n{text!r}")
    print(f"[motivation] {parsed.motivation}")
    for tc in parsed.tool_calls:
        print(f"[call] {tc['name']}  {json.dumps(tc['arguments'])[:100]}")
        EXECUTORS[tc["name"]](tc["arguments"], state, renderer, args.duration)
        state.executed.append(tc)

    modified_mix = _mixdown(state)

    out_dir = Path(args.out_root) / track.track_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "original.wav",
             _peak_normalize(original_mix).T, paths.SAMPLE_RATE)
    sf.write(out_dir / "modified.wav",
             _peak_normalize(modified_mix).T, paths.SAMPLE_RATE)
    log = {
        "track_id": track.track_id,
        "window_start_sec": start,
        "duration_sec": args.duration,
        "motivation": parsed.motivation,
        "think": parsed.think,
        "tool_calls": parsed.tool_calls,
        "input_tracks": list(state.tracks.keys()),
    }
    (out_dir / "trace.json").write_text(json.dumps(log, indent=2))

    diff = modified_mix - original_mix
    print(f"[output] wrote {out_dir}")
    print(f"[output] RMS(original)={float(np.sqrt(np.mean(original_mix**2))):.4f}  "
          f"RMS(modified)={float(np.sqrt(np.mean(modified_mix**2))):.4f}  "
          f"RMS(diff)={float(np.sqrt(np.mean(diff**2))):.4f}")
    renderer.close()


if __name__ == "__main__":
    main()

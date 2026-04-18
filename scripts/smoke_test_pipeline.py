"""End-to-end smoke test: generate a synthetic 2-track MIDI, render, apply
a canned tool-call sequence, save original+modified audio side by side.

Run with the ``pro-craft`` conda env active:

    python scripts/smoke_test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import paths
from procraft_data.rendering.fluidsynth_render import (
    FluidSynthRenderer, find_default_soundfont,
)
from procraft_data.tools import schemas as tool_schemas
from procraft_data.tools.executors import (
    EXECUTORS, MixtureState, TrackState, _mixdown,
)
from procraft_data.tools.parse import parse_response


def make_demo_mixture() -> MixtureState:
    """Piano + bass, 4-bar loop at 92 bpm."""
    state = MixtureState(sample_rate=paths.SAMPLE_RATE)

    piano = pretty_midi.Instrument(program=0, is_drum=False, name="piano")
    for bar in range(4):
        t0 = bar * 2.6
        for step, pitch in enumerate([60, 64, 67, 72]):
            s = t0 + step * 0.5
            piano.notes.append(pretty_midi.Note(80, pitch, s, s + 0.45))
    state.tracks["piano"] = TrackState("piano", 0, False, piano)

    bass = pretty_midi.Instrument(program=33, is_drum=False, name="bass")
    for bar in range(4):
        t0 = bar * 2.6
        for step, pitch in enumerate([36, 36, 40, 43]):
            s = t0 + step * 0.65
            bass.notes.append(pretty_midi.Note(95, pitch, s, s + 0.6))
    state.tracks["bass"] = TrackState("bass", 33, False, bass)

    return state


def main():
    sf_path = find_default_soundfont()
    if sf_path is None:
        raise SystemExit("No soundfont found. Set PROCRAFT_SOUNDFONT or drop "
                         f"a .sf2 into {paths.SOUNDFONTS}/")
    print(f"soundfont: {sf_path}")
    renderer = FluidSynthRenderer(sf_path, sample_rate=paths.SAMPLE_RATE)

    out_dir = paths.LOGS / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    state = make_demo_mixture()
    duration = paths.CLIP_SECONDS
    for t in state.tracks.values():
        t.audio = renderer.render_track(t, duration)

    original_mix = _mixdown(state)
    sf.write(out_dir / "original.wav", original_mix.T, paths.SAMPLE_RATE)

    # Simulate a Qwen3 response we expect to see in production.
    simulated = """<think>
The piano is too bright; swapping to an electric piano (GM 4) warms it.
Then a gentle tape saturation on the piano adds analog character, and a
light reverb on the mix glues the two stems together.
</think>

Production motivation: warm up the piano and add mix glue for a late-night jazz feel.

<tool_call>
{"name": "change_instrument", "arguments": {"track": "piano", "to_program": 4}}
</tool_call>
<tool_call>
{"name": "apply_fx", "arguments": {"track": "piano", "call": {"effect": "sox_overdrive", "params": {"gain_db": 8.0, "colour": 40.0}}}}
</tool_call>
<tool_call>
{"name": "apply_fx", "arguments": {"track": "mix", "call": {"effect": "sox_reverb", "params": {"reverberance": 30.0, "high_freq_damping": 50.0, "room_scale": 50.0, "stereo_depth": 80.0, "pre_delay": 12.0, "wet_gain": -10.0}}}}
</tool_call>
"""
    parsed = parse_response(simulated)
    assert parsed.is_valid(), "parsed response invalid"
    print(f"motivation: {parsed.motivation!r}")
    print(f"think: {parsed.think!r}")
    for tc in parsed.tool_calls:
        print(f"  call: {tc['name']}  args={json.dumps(tc['arguments'])[:120]}")

    for tc in parsed.tool_calls:
        EXECUTORS[tc["name"]](tc["arguments"], state, renderer, duration)
        state.executed.append(tc)

    modified_mix = _mixdown(state)
    sf.write(out_dir / "modified.wav", modified_mix.T, paths.SAMPLE_RATE)

    print(f"wrote {out_dir/'original.wav'}  shape={original_mix.shape}")
    print(f"wrote {out_dir/'modified.wav'}  shape={modified_mix.shape}")
    renderer.close()


if __name__ == "__main__":
    main()

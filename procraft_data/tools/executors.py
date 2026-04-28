"""Executors for the 10 tool schemas.

Each executor consumes a parsed ``{"name": ..., "arguments": ...}`` dict and
operates on a mutable ``MixtureState`` holding the per-track MIDI + per-stem
audio. The thinking model never sees these classes; it only sees the schemas
in ``schemas.py``.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import pretty_midi
from multiafx import FXChain, registry as _mafx_registry


# ---------------------------------------------------------------------------
# State shared across tool calls within a single dataset entry
# ---------------------------------------------------------------------------
@dataclass
class TrackState:
    name: str
    program: int            # General MIDI program (0-127); -1 for drums
    is_drum: bool
    midi: pretty_midi.Instrument    # mutable
    audio: np.ndarray | None = None  # (2, T) float32 stem, or None if not yet rendered


@dataclass
class MixtureState:
    sample_rate: int
    tracks: dict[str, TrackState] = field(default_factory=dict)
    # Withheld tracks for add_track: keyed by track_name → TrackState-with-midi-only.
    pending_tracks: dict[str, TrackState] = field(default_factory=dict)
    # Tool-call log for the dataset entry.
    executed: list[dict] = field(default_factory=list)

    def clone(self) -> "MixtureState":
        """Deep copy (for branching / failed-tool-call rollback)."""
        new = MixtureState(sample_rate=self.sample_rate)
        new.tracks = {k: TrackState(
            name=v.name, program=v.program, is_drum=v.is_drum,
            midi=copy.deepcopy(v.midi),
            audio=None if v.audio is None else v.audio.copy(),
        ) for k, v in self.tracks.items()}
        new.pending_tracks = {k: TrackState(
            name=v.name, program=v.program, is_drum=v.is_drum,
            midi=copy.deepcopy(v.midi), audio=None,
        ) for k, v in self.pending_tracks.items()}
        new.executed = list(self.executed)
        return new


# ---------------------------------------------------------------------------
# Renderer protocol (dependency-injected so executors stay lib-agnostic)
# ---------------------------------------------------------------------------
class Renderer:
    """Renders a pretty_midi.Instrument to stereo float32 (2, T)."""
    def render_track(self, track: TrackState, duration_sec: float) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------
def _require_track(state: MixtureState, name: str) -> TrackState:
    if name not in state.tracks:
        raise KeyError(f"Track {name!r} not in mixture (tracks: {list(state.tracks)})")
    return state.tracks[name]


def _get_track_arg(args: dict) -> str:
    """Accept either ``track`` or ``track_name`` — Qwen3 swaps them constantly.

    The tool schemas are intentionally inconsistent (``change_instrument``
    uses ``track`` while ``add_track`` uses ``track_name`` to match the
    proposal's tool-category conventions), and the model picks the wrong key
    about 10% of the time. Rather than surface that as an executor error we
    coerce: whichever alias is present wins, with ``track_name`` preferred.
    """
    v = args.get("track_name")
    if isinstance(v, str):
        return v
    v = args.get("track")
    if isinstance(v, str):
        return v
    raise KeyError(f"tool call missing 'track' or 'track_name': {args!r}")


def change_instrument(args: dict, state: MixtureState, renderer: Renderer,
                      duration_sec: float) -> None:
    track = _require_track(state, _get_track_arg(args))
    track.program = int(args["to_program"])
    track.midi.program = track.program
    track.audio = renderer.render_track(track, duration_sec)


def layer_instrument(args: dict, state: MixtureState, renderer: Renderer,
                     duration_sec: float) -> None:
    base = _require_track(state, _get_track_arg(args))
    add_prog = int(args["additional_program"])
    mix_ratio = float(args["mix_ratio"])

    layer = TrackState(
        name=f"{base.name}__layer{add_prog}",
        program=add_prog, is_drum=False,
        midi=pretty_midi.Instrument(program=add_prog, is_drum=False,
                                    name=f"{base.name}__layer{add_prog}"),
    )
    layer.midi.notes = copy.deepcopy(base.midi.notes)
    layer_audio = renderer.render_track(layer, duration_sec)
    if base.audio is None:
        base.audio = renderer.render_track(base, duration_sec)
    base.audio = base.audio + mix_ratio * layer_audio


def _coerce_fx_params(effect_name: str, params: dict | None) -> dict:
    """Map caller-supplied arg names onto the canonical multiafx names.

    Qwen3 occasionally cross-contaminates case between effects — e.g. emits
    ``"Q"`` (from ``ta_equalizer_biquad``) for ``am_peaking_filter`` which
    expects ``"q"``. Look up the registered param names for the effect and
    rewrite any case-insensitive matches to their canonical form. Unknown
    keys are dropped silently (model hallucinated a param this effect
    doesn't take).
    """
    params = params or {}
    try:
        eff = _mafx_registry.get(effect_name)
    except KeyError:
        return dict(params)
    canonical = {p.lower(): p for p in eff.param_ranges}
    out: dict = {}
    for k, v in params.items():
        cname = canonical.get(str(k).lower())
        if cname is None:
            continue
        out[cname] = v
    return out


_LEVEL_PRESERVING_EFFECTS: frozenset[str] = frozenset({
    # Distortion / saturation / drive — artistic intent is character/timbre,
    # but the DSP implementation also pushes the output level up substantially.
    # Without compensation, "add some grit" silently becomes "add grit AND
    # +6-12 dB". Compensate by peak-matching output to input.
    # Deliberately EXCLUDED: gain / loudness / vol / compand — those effects
    # exist specifically to change level.
    "sox_overdrive",
    "ta_overdrive",
    "am_tanh_distortion",
    "am_clipping_distortion",
    "am_bit_crush",
    "sox_contrast",
    "ta_contrast",
})


def _match_loudness(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Scale ``after`` so its RMS equals ``before``'s RMS (character-only).

    RMS-matching is the right operation for distortion/saturation: these
    effects flatten the waveform, raising the crest-factor — peak-matching
    leaves perceived loudness ~3-5× above input. RMS-matching drops the peak
    (sometimes below unity, that's fine) while preserving perceived loudness,
    so a "add some grit" call changes character only, not volume.
    """
    rms_before = float(np.sqrt(np.mean(before.astype(np.float64) ** 2)))
    rms_after = float(np.sqrt(np.mean(after.astype(np.float64) ** 2)))
    if rms_before <= 0.0 or rms_after <= 0.0:
        return after
    return (after * (rms_before / rms_after)).astype(np.float32)


def apply_fx(args: dict, state: MixtureState, renderer: Renderer,
             duration_sec: float) -> None:
    # apply_fx schema uses ``track`` but accept ``track_name`` too for
    # consistency with arrangement tools.
    target = args.get("track") or args.get("track_name")
    if not isinstance(target, str):
        raise KeyError(f"apply_fx missing 'track'/'track_name': {args!r}")
    call = args["call"]
    effect_name = call["effect"]
    coerced_params = _coerce_fx_params(effect_name, call.get("params", {}))
    chain = FXChain([{"effect": effect_name, "params": coerced_params}])

    if target == "mix":
        # Full mix-bus effect.
        mix = _mixdown(state)
        processed = chain(mix, state.sample_rate)
        if effect_name in _LEVEL_PRESERVING_EFFECTS:
            processed = _match_loudness(mix, processed)
        _apply_mix_delta(state, processed - mix)
        return
    track = _require_track(state, target)
    if track.audio is None:
        track.audio = renderer.render_track(track, duration_sec)
    before = track.audio
    after = chain(before, state.sample_rate)
    if effect_name in _LEVEL_PRESERVING_EFFECTS:
        after = _match_loudness(before, after)
    track.audio = after


def humanize_timing(args: dict, state: MixtureState, renderer: Renderer,
                    duration_sec: float, rng: random.Random | None = None) -> None:
    """Gaussian timing jitter with light coupled velocity variation.

    Algorithm:
      - per-note Gaussian offset dt ~ N(0, sigma) where sigma = max_offset_ms/2
        (clipped to ±max_offset_ms so extreme outliers don't wreck the groove)
      - start and end shift together by dt (duration preserved)
      - coupled velocity jitter: each note's velocity multiplied by
        (1 + N(0, 0.08)) clipped to [0.6, 1.25], so humanized timing also
        brings humanized dynamics. This matches how real human performance
        varies — timing and velocity are weakly correlated, not independent.
      - RNG seed derived from the current wall-clock time (plus id(track))
        so repeated calls produce independently humanized takes.
    """
    import time as _time
    if rng is None:
        rng = random.Random(_time.time_ns() ^ id(args) ^ id(state))
    track = _require_track(state, _get_track_arg(args))
    max_offset = float(args["max_offset_ms"]) / 1000.0
    sigma = max_offset * 0.5
    for note in track.midi.notes:
        dt = max(-max_offset, min(max_offset, rng.gauss(0.0, sigma)))
        note.start = max(0.0, note.start + dt)
        note.end = max(note.start + 1e-3, note.end + dt)
        vel_mult = 1.0 + rng.gauss(0.0, 0.08)
        vel_mult = max(0.6, min(1.25, vel_mult))
        note.velocity = max(1, min(127, int(round(note.velocity * vel_mult))))
    track.audio = renderer.render_track(track, duration_sec)


def change_articulation(args: dict, state: MixtureState, renderer: Renderer,
                        duration_sec: float) -> None:
    track = _require_track(state, _get_track_arg(args))
    style = args["style"]
    # Stretch/compress each note's duration around its midpoint.
    factor = {"legato": 1.2, "staccato": 0.6, "tenuto": 1.0}[style]
    for note in track.midi.notes:
        dur = note.end - note.start
        new_dur = max(0.02, dur * factor)
        note.end = note.start + new_dur
    track.audio = renderer.render_track(track, duration_sec)


def add_track(args: dict, state: MixtureState, renderer: Renderer,
              duration_sec: float) -> None:
    name = _get_track_arg(args)
    if name not in state.pending_tracks:
        raise KeyError(f"No pending track {name!r} available to add "
                       f"(available: {list(state.pending_tracks)})")
    track = state.pending_tracks.pop(name)
    track.program = int(args["program"])
    track.midi.program = track.program
    gain = 10 ** (float(args.get("gain_db", 0.0)) / 20.0)
    track.audio = gain * renderer.render_track(track, duration_sec)
    state.tracks[name] = track


def remove_track(args: dict, state: MixtureState, renderer: Renderer,
                 duration_sec: float) -> None:
    name = _get_track_arg(args)
    if name not in state.tracks:
        return
    del state.tracks[name]


def double_track(args: dict, state: MixtureState, renderer: Renderer,
                 duration_sec: float) -> None:
    base = _require_track(state, _get_track_arg(args))
    offset_s = float(args["offset_ms"]) / 1000.0
    detune = float(args["detune_cents"])

    twin = TrackState(
        name=f"{base.name}__dbl",
        program=base.program, is_drum=base.is_drum,
        midi=copy.deepcopy(base.midi),
    )
    # Apply detune via pitch_bend on all notes (crude but MIDI-native).
    bend_value = int(round(detune / 100.0 * 8192))   # ±8192 == ±2 semitones typically
    twin.midi.pitch_bends.append(pretty_midi.PitchBend(pitch=bend_value, time=0.0))
    for note in twin.midi.notes:
        note.start += offset_s
        note.end += offset_s
    twin.audio = renderer.render_track(twin, duration_sec)
    if base.audio is None:
        base.audio = renderer.render_track(base, duration_sec)
    base.audio = base.audio + twin.audio


def mute_and_replace(args: dict, state: MixtureState, renderer: Renderer,
                     duration_sec: float) -> None:
    name = _get_track_arg(args)
    old = _require_track(state, name)
    # Accept ``new_program``, ``program``, or ``to_program`` — model confuses all three.
    new_program = int(args.get("new_program") or args.get("program")
                      or args.get("to_program"))
    replacement = TrackState(
        name=name, program=new_program, is_drum=False,
        midi=copy.deepcopy(old.midi),
    )
    replacement.midi.program = new_program
    replacement.audio = renderer.render_track(replacement, duration_sec)
    state.tracks[name] = replacement


EXECUTORS = {
    "change_instrument": change_instrument,
    "layer_instrument": layer_instrument,
    "apply_fx": apply_fx,
    "humanize_timing": humanize_timing,
    "change_articulation": change_articulation,
    "add_track": add_track,
    "remove_track": remove_track,
    "double_track": double_track,
    "mute_and_replace": mute_and_replace,
}


# ---------------------------------------------------------------------------
# Mix helpers
# ---------------------------------------------------------------------------
def _mixdown(state: MixtureState) -> np.ndarray:
    """Sum all track stems. Raises if any track lacks audio."""
    stems = []
    length = 0
    for t in state.tracks.values():
        if t.audio is None:
            raise RuntimeError(f"Track {t.name!r} has no audio rendered yet.")
        stems.append(t.audio)
        length = max(length, t.audio.shape[-1])
    if not stems:
        return np.zeros((2, 1), dtype=np.float32)
    padded = [np.pad(s, ((0, 0), (0, length - s.shape[-1]))) if s.shape[-1] < length else s
              for s in stems]
    return np.sum(padded, axis=0).astype(np.float32)


def _apply_mix_delta(state: MixtureState, delta: np.ndarray) -> None:
    """Bake a mix-bus effect delta onto an arbitrary carrier stem.

    Shortcut: a true bus chain would re-apply the effect after any later stem
    mutation. We keep it simple — store the delta on the first stem. This is
    correct when the model places mix-bus moves after per-stem moves, which is
    the conventional engineer sequencing.
    """
    if not state.tracks:
        return
    carrier = next(iter(state.tracks.values()))
    if carrier.audio is None:
        carrier.audio = np.zeros_like(delta)
    L = min(carrier.audio.shape[-1], delta.shape[-1])
    carrier.audio[:, :L] = carrier.audio[:, :L] + delta[:, :L]

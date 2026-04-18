"""Single reusable ingest function used by both single-track and batch scripts.

Writes, per Slakh track, to ``<out_root>/<track_id>/``:

    mix.wav                — peak-normalized stereo mix
    meta.json              — full metadata (track-level + per-stem), see MetaRecord

Stems are **not** persisted — FluidSynth rendering is deterministic given the
MIDI + soundfont + seed, so stems can be regenerated on demand. The meta.json
still records every stem's identity (GM program, ``is_drum``, Slakh ``inst_class``,
Slakh ``plugin_name``) so a timbre-embedding trainer has the label side without
the audio duplication.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from procraft_data.rendering.fluidsynth_render import FluidSynthRenderer
from procraft_data.sources.slakh import (
    TrackMeta, _dedup_names, build_mixture_state,
)
from procraft_data.tools.executors import MixtureState, _mixdown


@dataclass(frozen=True)
class StemRecord:
    name: str               # dedup'd track name used by tool-call 'track' field
    stem_id: str            # Slakh S00/S01/... (stable across runs)
    program: int            # GM program 0-127; 128 for drums (Slakh sentinel)
    is_drum: bool
    inst_class: str         # Slakh's curated class — training target for [TMB]
    plugin_name: str        # Slakh's source VSTi (proxy for fine-grained timbre)
    midi_program_name: str  # GM-standard human-readable program
    integrated_loudness: float | None


@dataclass
class TrackRecord:
    track_id: str
    source: str             # "slakh/babyslakh" or "slakh/full" etc.
    uuid: str
    sample_rate: int
    window_start_sec: float
    window_duration_sec: float
    midi_total_duration_sec: float
    peak_normalize_divisor: float | None
    mix_path: str
    stems: list[StemRecord]


def midi_duration(track: TrackMeta) -> float:
    longest = 0.0
    for sid in track.stems:
        longest = max(longest, pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time())
    return longest


def ingest_one(
    track: TrackMeta,
    renderer: FluidSynthRenderer,
    out_root: Path,
    *,
    sample_rate: int = 48000,
    start_sec: float | None = None,
    duration_sec: float = 10.0,
    source_tag: str = "slakh/babyslakh",
) -> TrackRecord:
    """Render one Slakh track and persist mix + meta. Stems are not saved."""
    total = midi_duration(track)
    start = start_sec if start_sec is not None else max(0.0, (total - duration_sec) / 2.0)

    state = build_mixture_state(track, sample_rate, start, duration_sec)
    name_by_sid = _dedup_names(list(track.stems.values()))

    for ts in state.tracks.values():
        ts.audio = renderer.render_track(ts, duration_sec)

    out_dir = out_root / track.track_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_records: list[StemRecord] = []
    for sid, stem_meta in track.stems.items():
        tname = name_by_sid[sid]
        ts = state.tracks.get(tname) or state.pending_tracks.get(tname)
        if ts is None:
            continue
        stem_records.append(StemRecord(
            name=tname,
            stem_id=sid,
            program=stem_meta.program_num,
            is_drum=stem_meta.is_drum,
            inst_class=stem_meta.inst_class,
            plugin_name=stem_meta.plugin_name,
            midi_program_name=stem_meta.midi_program_name,
            integrated_loudness=stem_meta.integrated_loudness,
        ))

    mix = _mixdown(state)
    peak = float(np.max(np.abs(mix))) or 1.0
    divisor = peak if peak > 1.0 else None
    if divisor:
        mix = mix / divisor * 0.99
    sf.write(out_dir / "mix.wav", mix.T, sample_rate)

    record = TrackRecord(
        track_id=track.track_id,
        source=source_tag,
        uuid=track.uuid,
        sample_rate=sample_rate,
        window_start_sec=start,
        window_duration_sec=duration_sec,
        midi_total_duration_sec=total,
        peak_normalize_divisor=divisor,
        mix_path="mix.wav",
        stems=stem_records,
    )
    (out_dir / "meta.json").write_text(_dumps(record))
    return record


def _dumps(rec: TrackRecord) -> str:
    d = asdict(rec)
    return json.dumps(d, indent=2)

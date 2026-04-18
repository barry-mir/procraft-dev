"""Slakh2100 / BabySlakh loader.

Track layout (from ethman/slakh-utils):

    Track00001/
      all_src.mid          # unified MIDI from Lakh (all stems)
      metadata.yaml        # per-stem metadata (see below)
      MIDI/
        S01.mid, S02.mid, …   # one file per stem
      mix.flac             # Slakh's own VSTi-rendered mix (we ignore; we re-render)
      stems/
        S01.flac, S02.flac, … # Slakh's own per-stem audio (we ignore)

metadata.yaml shape (one entry per stem id, e.g. "S01"):

    stems:
      S01:
        inst_class: Piano           # curated class — what we use for [TMB] supervision
        midi_program_name: "Acoustic Grand Piano"
        program_num: 0              # GM program 0-127
        is_drum: false
        plugin_name: "grand-piano-YDP-20160804.sf2"
        integrated_loudness: -18.3
        audio_rendered: true
        midi_saved: true

The loader ignores Slakh's FLAC audio entirely — proposal §3.1.1 commits to re-rendering with
FluidSynth + GM soundfonts so we have per-mixture control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pretty_midi
import yaml

from procraft_data.tools.executors import MixtureState, TrackState


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StemMeta:
    stem_id: str           # "S01"
    inst_class: str        # "Piano", "Guitar Clean", "Drums", ...
    program_num: int       # GM program 0-127 (0 for drums; see is_drum)
    midi_program_name: str
    is_drum: bool
    plugin_name: str
    integrated_loudness: float | None
    midi_saved: bool


@dataclass(frozen=True)
class TrackMeta:
    track_id: str          # "Track00001"
    root: Path             # /nas/pro-craft/raw/slakh2100/babyslakh/Track00001
    uuid: str
    stems: dict[str, StemMeta]

    def midi_path(self, stem_id: str) -> Path:
        return self.root / "MIDI" / f"{stem_id}.mid"

    def all_src_midi_path(self) -> Path:
        return self.root / "all_src.mid"


# ---------------------------------------------------------------------------
# Discovery / parsing
# ---------------------------------------------------------------------------
def iter_tracks(slakh_root: str | Path) -> Iterator[TrackMeta]:
    """Yield every ``TrackMeta`` under ``slakh_root``, sorted by track id."""
    root = Path(slakh_root)
    for track_dir in sorted(root.iterdir()):
        if not track_dir.is_dir() or not track_dir.name.startswith("Track"):
            continue
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        yield _parse_track(track_dir, meta_path)


def load_track(track_dir: str | Path) -> TrackMeta:
    track_dir = Path(track_dir)
    return _parse_track(track_dir, track_dir / "metadata.yaml")


def _parse_track(track_dir: Path, meta_path: Path) -> TrackMeta:
    with meta_path.open() as f:
        blob = yaml.safe_load(f)

    stems_blob = blob.get("stems", {}) or {}
    stems: dict[str, StemMeta] = {}
    for sid, s in stems_blob.items():
        # BabySlakh's metadata.yaml ships with midi_saved=false for every stem even
        # though the .mid files are present. Trust the filesystem, not the flag.
        midi_path = track_dir / "MIDI" / f"{sid}.mid"
        if not midi_path.exists():
            continue
        stems[sid] = StemMeta(
            stem_id=sid,
            inst_class=s.get("inst_class", "Unknown") or "Unknown",
            program_num=int(s.get("program_num", 0) or 0),
            midi_program_name=s.get("midi_program_name", "") or "",
            is_drum=bool(s.get("is_drum", False)),
            plugin_name=s.get("plugin_name", "") or "",
            integrated_loudness=(float(s["integrated_loudness"])
                                 if s.get("integrated_loudness") is not None else None),
            midi_saved=True,
        )

    return TrackMeta(
        track_id=track_dir.name,
        root=track_dir,
        uuid=str(blob.get("UUID", "")),
        stems=stems,
    )


# ---------------------------------------------------------------------------
# Track-name dedup for tool-call readability
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


def _dedup_names(stems: list[StemMeta]) -> dict[str, str]:
    """Return stem_id → human-readable track name, deduplicating on inst_class.

    E.g. two "Piano" stems become "piano_1" / "piano_2"; a single "Bass" stays "bass".
    Drums always become "drums" (or "drums_N") regardless of ``inst_class``.
    """
    base_counts: dict[str, int] = {}
    raw_bases: dict[str, str] = {}
    for s in stems:
        base = "drums" if s.is_drum else _slug(s.inst_class)
        raw_bases[s.stem_id] = base
        base_counts[base] = base_counts.get(base, 0) + 1

    running: dict[str, int] = {}
    out: dict[str, str] = {}
    for s in stems:
        base = raw_bases[s.stem_id]
        if base_counts[base] == 1:
            out[s.stem_id] = base
        else:
            running[base] = running.get(base, 0) + 1
            out[s.stem_id] = f"{base}_{running[base]}"
    return out


# ---------------------------------------------------------------------------
# MixtureState construction
# ---------------------------------------------------------------------------
def build_mixture_state(
    track: TrackMeta,
    sample_rate: int,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    withhold: list[str] | None = None,
) -> MixtureState:
    """Load per-stem MIDI, window to ``[start_sec, start_sec+duration_sec]``,
    and return a ``MixtureState`` ready for rendering.

    ``withhold``: list of track names (post-dedup) to move into
    ``pending_tracks`` — the ``add_track`` tool will materialize them on demand.
    Use this to create natural (incomplete-mix, complete-mix) pairs for
    Category D training (proposal §3.2).
    """
    withhold = set(withhold or [])
    stems = list(track.stems.values())
    names = _dedup_names(stems)

    state = MixtureState(sample_rate=sample_rate)
    for stem in stems:
        inst = _load_stem_window(track.midi_path(stem.stem_id), stem,
                                 start_sec, duration_sec)
        # Silent-track filter: if a stem has no (or very few) notes landing
        # inside this window, drop it entirely. Otherwise the metadata would
        # advertise a track the listener can't hear, and the LLM may try to
        # operate on it.
        if len(inst.notes) < 2:
            continue
        ts = TrackState(
            name=names[stem.stem_id],
            program=stem.program_num,
            is_drum=stem.is_drum,
            midi=inst,
        )
        if ts.name in withhold:
            state.pending_tracks[ts.name] = ts
        else:
            state.tracks[ts.name] = ts
    return state


def _load_stem_window(midi_path: Path, stem: StemMeta,
                      start_sec: float, duration_sec: float) -> pretty_midi.Instrument:
    """Load one stem MIDI, trim to the requested window, shift to zero."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    end_sec = start_sec + duration_sec

    src_inst = pm.instruments[0] if pm.instruments else pretty_midi.Instrument(
        program=stem.program_num, is_drum=stem.is_drum)

    inst = pretty_midi.Instrument(program=stem.program_num, is_drum=stem.is_drum,
                                  name=src_inst.name or stem.inst_class)
    for note in src_inst.notes:
        if note.end <= start_sec or note.start >= end_sec:
            continue
        n_start = max(0.0, note.start - start_sec)
        n_end = min(duration_sec, note.end - start_sec)
        if n_end <= n_start:
            continue
        inst.notes.append(pretty_midi.Note(
            velocity=int(note.velocity), pitch=int(note.pitch),
            start=n_start, end=n_end,
        ))
    for cc in src_inst.control_changes:
        if start_sec <= cc.time <= end_sec:
            inst.control_changes.append(pretty_midi.ControlChange(
                number=int(cc.number), value=int(cc.value),
                time=cc.time - start_sec,
            ))
    for pb in src_inst.pitch_bends:
        if start_sec <= pb.time <= end_sec:
            inst.pitch_bends.append(pretty_midi.PitchBend(
                pitch=int(pb.pitch), time=pb.time - start_sec,
            ))
    return inst


# ---------------------------------------------------------------------------
# System-prompt metadata string for Qwen3
# ---------------------------------------------------------------------------
def natural_names(state: MixtureState, track: TrackMeta) -> dict[str, str]:
    """Return ``{track_identifier: english phrase}`` for motivation prose.

    Disambiguates when multiple tracks share the same ``midi_program_name``
    by prepending "first" / "second" / "third". Drums become "the drums"
    or "the drum kit" (picked deterministically). The phrase NEVER contains
    the internal identifier, so the model can use it as a drop-in replacement
    in motivation sentences.
    """
    names_by_sid = _dedup_names(list(track.stems.values()))
    sid_by_name = {v: k for k, v in names_by_sid.items()}

    # How many active tracks share each program name?
    prog_for_name: dict[str, str] = {}
    for name, ts in {**state.tracks, **state.pending_tracks}.items():
        if ts.is_drum:
            prog_for_name[name] = "Drum Kit"
            continue
        sid = sid_by_name.get(name)
        stem = track.stems.get(sid) if sid else None
        prog_for_name[name] = (stem.midi_program_name if stem and stem.midi_program_name
                               else "Instrument")

    # Count duplicates.
    dup_index: dict[str, int] = {}
    seen: dict[str, int] = {}
    for name, prog in prog_for_name.items():
        seen[prog] = seen.get(prog, 0) + 1
    ordinals = ["first", "second", "third", "fourth", "fifth"]
    counter: dict[str, int] = {}

    def phrase(name: str, prog: str, is_drum: bool) -> str:
        if is_drum:
            return "the drums"
        prog_l = prog.lower()
        if seen[prog] > 1:
            i = counter.get(prog, 0)
            counter[prog] = i + 1
            ord_word = ordinals[i] if i < len(ordinals) else f"{i + 1}th"
            return f"the {ord_word} {prog_l}"
        return f"the {prog_l}"

    out: dict[str, str] = {}
    for name in {**state.tracks, **state.pending_tracks}:
        ts = state.tracks.get(name) or state.pending_tracks.get(name)
        out[name] = phrase(name, prog_for_name[name], ts.is_drum)
    return out


def describe_mixture(state: MixtureState, track: TrackMeta) -> str:
    """Build the Mixture metadata string Qwen3 sees in the system prompt.

    Format per track: ``<identifier> "<midi_program_name>" (GM N)``. The
    identifier (e.g. ``guitar_1``) is the exact string tool_calls must use;
    the quoted human program name is what the model should cite in the
    English motivation. A drums track reads ``drums "Drum Kit"``.
    """
    names_by_sid = _dedup_names(list(track.stems.values()))
    sid_by_name = {v: k for k, v in names_by_sid.items()}

    def human_for(name: str, ts: TrackState) -> str:
        sid = sid_by_name.get(name)
        stem = track.stems.get(sid) if sid else None
        if ts.is_drum:
            return "Drum Kit"
        return (stem.midi_program_name if stem and stem.midi_program_name
                else "Unknown")

    parts = []
    for name, ts in state.tracks.items():
        human = human_for(name, ts)
        if ts.is_drum:
            parts.append(f'{name} "{human}"')
        else:
            parts.append(f'{name} "{human}" (GM {ts.program})')

    pending_desc = []
    for name, ts in state.pending_tracks.items():
        human = human_for(name, ts)
        if ts.is_drum:
            pending_desc.append(f'{name} "{human}" (available-to-add)')
        else:
            pending_desc.append(f'{name} "{human}" (GM {ts.program}, available-to-add)')

    tracks_str = ", ".join(parts) if parts else "(empty)"
    out = f"tracks: {tracks_str}"
    if pending_desc:
        out += f"; available_to_add: {', '.join(pending_desc)}"
    out += f"; source: slakh/{track.track_id}"
    return out

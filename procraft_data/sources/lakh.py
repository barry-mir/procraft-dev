"""Lakh MIDI Dataset (LMD-full) loader.

Lakh ships a single multi-instrument MIDI per track (no per-stem files
the way Slakh does). This loader splits each file by
``pretty_midi.Instrument`` and writes the result into a per-stem MIDI
cache so the rest of the pipeline can consume Lakh tracks through the
same ``TrackMeta`` interface as Slakh:

    /nas/pro-craft/cache/lakh_stems/<first-char>/<md5>/MIDI/S00.mid
    /nas/pro-craft/cache/lakh_stems/<first-char>/<md5>/MIDI/S01.mid
    ...

Returns a Slakh-shaped ``TrackMeta`` whose ``midi_path(sid)`` resolves
to those per-stem cache files. Sanity-filter rules are in
``scripts/lakh_sanity_filter.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pretty_midi

from procraft_data.sources.slakh import StemMeta, TrackMeta


CACHE_ROOT = Path("/nas/pro-craft/cache/lakh_stems")


# ---------------------------------------------------------------------------
# GM program → broad inst_class slug. Mirrors Slakh's inst_class taxonomy
# at a coarse granularity. Used by the existing _dedup_names / natural_names
# helpers to build identifier slugs (``piano_1``, ``guitar_2``, ...).
# ---------------------------------------------------------------------------
_GM_FAMILY = [
    (0,  7,   "Piano"),
    (8,  15,  "Chromatic Percussion"),
    (16, 23,  "Organ"),
    (24, 31,  "Guitar"),
    (32, 39,  "Bass"),
    (40, 47,  "Strings"),
    (48, 55,  "Strings"),       # Ensemble — fold into Strings for naming
    (56, 63,  "Brass"),
    (64, 71,  "Reed"),
    (72, 79,  "Pipe"),
    (80, 87,  "Synth Lead"),
    (88, 95,  "Synth Pad"),
    (96, 103, "Synth Effects"),
    (104, 111, "Ethnic"),
    (112, 119, "Percussive"),
    (120, 127, "Sound Effects"),
]


def _inst_class(program: int, is_drum: bool) -> str:
    if is_drum:
        return "Drums"
    for lo, hi, name in _GM_FAMILY:
        if lo <= program <= hi:
            return name
    return "Unknown"


def load_track(midi_path: str | Path) -> TrackMeta:
    """Parse a Lakh MIDI and return a Slakh-shaped ``TrackMeta``.

    On first call, splits each non-empty ``pretty_midi.Instrument`` into
    its own .mid file under ``CACHE_ROOT/<first>/<md5>/MIDI/<sid>.mid``.
    Subsequent calls hit the cache.
    """
    midi_path = Path(midi_path)
    md5 = midi_path.stem            # e.g. "fcc9210e5d67edd1d5080e75b175ca3c"
    first = md5[:1] or "0"
    cache_dir = CACHE_ROOT / first / md5
    midi_dir = cache_dir / "MIDI"
    midi_dir.mkdir(parents=True, exist_ok=True)

    # Parse the source file once.
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    stems: dict[str, StemMeta] = {}
    written_any = False
    for idx, inst in enumerate(pm.instruments):
        if not inst.notes:
            continue
        sid = f"S{idx:02d}"
        cache_file = midi_dir / f"{sid}.mid"
        if not cache_file.exists() or cache_file.stat().st_size == 0:
            single = pretty_midi.PrettyMIDI()
            # Reset channel — pretty_midi assigns channels at write time
            # but we want each per-stem file to have just one melodic
            # (channel 0) or drum (channel 9) instrument so downstream
            # parsers don't hit conflicts.
            new_inst = pretty_midi.Instrument(
                program=0 if inst.is_drum else max(0, min(127, int(inst.program))),
                is_drum=inst.is_drum,
                name=inst.name or "",
            )
            new_inst.notes = list(inst.notes)
            new_inst.control_changes = list(inst.control_changes)
            new_inst.pitch_bends = list(inst.pitch_bends)
            single.instruments.append(new_inst)
            try:
                single.write(str(cache_file))
                written_any = True
            except Exception:
                # Skip pathological instruments that pretty_midi can't
                # round-trip cleanly; the stem just won't be available.
                if cache_file.exists():
                    cache_file.unlink()
                continue
        program = 0 if inst.is_drum else max(0, min(127, int(inst.program)))
        try:
            midi_program_name = (
                "Drum Kit" if inst.is_drum
                else pretty_midi.program_to_instrument_name(program)
            )
        except Exception:
            midi_program_name = "Unknown"
        stems[sid] = StemMeta(
            stem_id=sid,
            inst_class=_inst_class(program, inst.is_drum),
            program_num=128 if inst.is_drum else program,
            midi_program_name=midi_program_name,
            is_drum=inst.is_drum,
            plugin_name="lakh",
            integrated_loudness=None,
            midi_saved=True,
        )

    return TrackMeta(
        track_id=md5,
        root=cache_dir,
        uuid=md5,
        stems=stems,
    )


def iter_paths_from_list(list_path: str | Path) -> Iterator[Path]:
    """Yield Lakh MIDI paths from a kept_paths.txt-style file."""
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield Path(line)

"""Tier-2 renderer — FluidSynth + GM soundfont.

Each worker process owns a dedicated ``fluidsynth.Synth`` with the soundfont
loaded once. Renders a single ``pretty_midi.Instrument`` to stereo float32
at ``sample_rate``. We synthesize the MIDI event stream explicitly (noteon,
noteoff, program change) rather than going through pretty_midi's
``fluidsynth()`` because that path assumes mono and allocates fresh Synth
objects per call.
"""

from __future__ import annotations

import os
from pathlib import Path

import fluidsynth
import numpy as np
import pretty_midi

from procraft_data.tools.executors import Renderer, TrackState


class FluidSynthRenderer(Renderer):
    """Stereo FluidSynth renderer with a persistent Synth instance.

    One instance per worker process — do not share across processes (libfluidsynth
    is not fork-safe once a soundfont is loaded).
    """

    def __init__(self, soundfont_path: str | Path, sample_rate: int = 48000,
                 gain: float = 0.5):
        self.soundfont_path = str(soundfont_path)
        self.sample_rate = sample_rate
        self._synth: fluidsynth.Synth | None = None
        self._sfid: int | None = None
        self._gain = gain

    def _ensure_synth(self) -> fluidsynth.Synth:
        if self._synth is None:
            synth = fluidsynth.Synth(samplerate=float(self.sample_rate), gain=self._gain)
            self._sfid = synth.sfload(self.soundfont_path)
            self._synth = synth
        return self._synth

    def render_track(self, track: TrackState, duration_sec: float) -> np.ndarray:
        synth = self._ensure_synth()
        channel = 9 if track.is_drum else 0
        # Slakh uses program_num=128 as a sentinel for drums; FluidR3_GM.sf2
        # stores drum kits at bank=128 preset=0 (standard kit). Clamp to valid GM.
        if track.is_drum:
            program, bank = 0, 128
        else:
            program = max(0, min(127, int(track.program)))
            bank = 0
        synth.program_select(channel, self._sfid, bank, program)

        # Apply any static pitch bends before first noteon
        for pb in track.midi.pitch_bends:
            if pb.time <= 0.0:
                synth.pitch_bend(channel, int(pb.pitch))

        events: list[tuple[float, str, tuple]] = []
        for note in track.midi.notes:
            events.append((note.start, "on", (channel, int(note.pitch), int(note.velocity))))
            events.append((note.end, "off", (channel, int(note.pitch))))
        for cc in track.midi.control_changes:
            events.append((cc.time, "cc", (channel, int(cc.number), int(cc.value))))
        events.sort(key=lambda e: e[0])

        total_samples = int(duration_sec * self.sample_rate)
        out = np.zeros((2, total_samples), dtype=np.float32)
        cursor = 0
        for t, kind, args in events:
            target = min(total_samples, int(t * self.sample_rate))
            if target > cursor:
                seg = synth.get_samples(target - cursor)
                out[:, cursor:target] = _interleaved_to_stereo(seg)
                cursor = target
            if cursor >= total_samples:
                break
            if kind == "on":
                synth.noteon(*args)
            elif kind == "off":
                synth.noteoff(*args)
            elif kind == "cc":
                synth.cc(*args)

        # Tail: flush remaining samples (release envelopes, reverb tails).
        if cursor < total_samples:
            seg = synth.get_samples(total_samples - cursor)
            out[:, cursor:] = _interleaved_to_stereo(seg)

        # All-notes-off at end for next call.
        synth.all_notes_off(channel)
        return out

    def close(self) -> None:
        if self._synth is not None:
            self._synth.delete()
            self._synth = None


def _interleaved_to_stereo(buf: np.ndarray) -> np.ndarray:
    """FluidSynth returns interleaved int16 stereo; convert to (2, T) float32."""
    if buf.dtype == np.int16:
        buf = buf.astype(np.float32) / 32768.0
    else:
        buf = buf.astype(np.float32)
    # Interleaved LRLRLR... → (2, T)
    if buf.ndim == 1:
        if buf.size % 2 != 0:
            buf = buf[:-1]
        return buf.reshape(-1, 2).T.copy()
    return buf


def find_default_soundfont() -> Path | None:
    """Locate a usable GM soundfont.

    Priority:
      1. ``$PROCRAFT_SOUNDFONT`` env var
      2. ``/nas/pro-craft/soundfonts/*.sf2``
      3. Standard system locations (fluid-soundfont-gm etc.)
    """
    env = os.environ.get("PROCRAFT_SOUNDFONT")
    if env and Path(env).exists():
        return Path(env)

    from configs import paths
    candidates = list(paths.SOUNDFONTS.glob("*.sf2"))
    if candidates:
        return sorted(candidates)[0]

    for p in [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default-GM.sf2",
        "/usr/share/soundfonts/default.sf2",
    ]:
        if Path(p).exists():
            return Path(p)
    return None

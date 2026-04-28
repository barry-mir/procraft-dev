"""Sanity checks on prompt construction — no LLM needed."""

import json

from procraft_data.pipeline.trace_prompts import (
    ABSTRACTION_LEVELS, PRIMARY_INTENTS, ROLES,
    build_spec, coverage_plan, sample_primary_intent, smoke_plan,
)


EXAMPLE_META = (
    "tracks: piano \"Acoustic Grand Piano\" (GM 0), drums \"Drum Kit\", "
    "bass \"Electric Bass (finger)\" (GM 33), strings \"String Ensemble 1\" (GM 48), "
    "guitar_1 \"Distortion Guitar\" (GM 30); "
    "source: slakh/Track00001"
)


def test_roles_and_levels_defined():
    assert len(ROLES) >= 10
    assert len(ABSTRACTION_LEVELS) == 6


def test_build_spec_samples_role_level_hook():
    spec = build_spec(EXAMPLE_META, seed=0)
    assert spec.role in {r.name for r in ROLES}
    assert spec.abstraction_level in ABSTRACTION_LEVELS
    assert spec.hook
    assert spec.category_focus == "free"
    assert "Hard rules" in spec.system
    assert "INSTRUMENTAL ONLY" in spec.system
    assert "Effect cookbook" in spec.system
    assert "<tools>" in spec.system
    assert EXAMPLE_META in spec.system


def test_build_spec_respects_pins():
    spec = build_spec(
        EXAMPLE_META,
        role="non_musician_client",
        abstraction_level="scene_metaphor",
        seed=42,
    )
    assert spec.role == "non_musician_client"
    assert spec.abstraction_level == "scene_metaphor"
    assert "non_musician_client" in spec.user
    assert "scene_metaphor" in spec.user


def test_session_roles_skipped_when_target_missing():
    # Mixture with NO drums → session_drummer must not be sampled.
    no_drums = "tracks: piano (GM program 1), bass (GM program 33); source: slakh/x"
    for _ in range(50):
        spec = build_spec(no_drums, seed=_)
        assert spec.role != "session_drummer"


def test_smoke_plan_covers_every_primary_intent_once():
    plan = smoke_plan(EXAMPLE_META, seed=0)
    intents_in_plan = [s.primary_intent for s in plan]
    assert sorted(intents_in_plan) == sorted(PRIMARY_INTENTS)


def test_coverage_plan_grid_size():
    plan = coverage_plan(
        EXAMPLE_META,
        temperatures=(0.7, 0.9),
        intents_per_temp=4,
        roles_per_intent=2,
        seed=0,
    )
    assert len(plan) == 2 * 4 * 2


def test_sample_primary_intent_pre_commits_target_where_expected():
    for intent_name in PRIMARY_INTENTS:
        ic = sample_primary_intent(EXAMPLE_META, forced_intent=intent_name, seed=1)
        if intent_name in {"effects", "arrangement_add"}:
            # effects is free-choice; arrangement_add target is decided by the
            # pipeline at withhold time, not by sample_primary_intent.
            continue
        if intent_name == "remix":
            # remix pre-commits via ``forced_calls`` + ``plan`` (a partition
            # plus per-track decisions), not via a single ``target_track``.
            assert ic.forced_calls, "remix must pre-commit forced_calls"
            assert ic.plan, "remix must pre-commit a plan"
            continue
        assert ic.target_track is not None, f"{intent_name} must pre-commit a target"


def test_build_spec_includes_primary_move_block():
    spec = build_spec(EXAMPLE_META, seed=0)
    assert "PRIMARY MOVE" in spec.user
    assert spec.primary_intent in PRIMARY_INTENTS
    assert spec.primary_tool


def test_system_prompt_is_parseable_tools_json():
    spec = build_spec(EXAMPLE_META, seed=0)
    blob = spec.system.split("<tools>", 1)[1].split("</tools>", 1)[0]
    tools = json.loads(blob)
    assert any(t["function"]["name"] == "apply_fx" for t in tools)

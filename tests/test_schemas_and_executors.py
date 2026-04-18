"""Sanity checks: schemas round-trip, every executor has a schema and vice versa."""

import json
import pytest

from procraft_data.tools import schemas
from procraft_data.tools.executors import EXECUTORS
from procraft_data.tools.parse import parse_response


def test_schema_executor_parity():
    schema_names = {t["function"]["name"] for t in schemas.all_tool_schemas()}
    executor_names = set(EXECUTORS)
    assert schema_names == executor_names, (
        f"missing executors: {schema_names - executor_names}; "
        f"orphan executors: {executor_names - schema_names}"
    )


def test_apply_fx_oneof_covers_all_multiafx_effects():
    from multiafx import registry
    apply_fx = next(t for t in schemas.all_tool_schemas()
                    if t["function"]["name"] == "apply_fx")
    branches = apply_fx["function"]["parameters"]["properties"]["call"]["oneOf"]
    listed = {b["properties"]["effect"]["const"] for b in branches}
    assert listed == set(registry.list_effects())


def test_system_prompt_is_serializable():
    prompt = schemas.build_hermes_system_prompt("tracks: piano, bass")
    # tools should be valid JSON
    blob = prompt.split("<tools>", 1)[1].split("</tools>", 1)[0]
    assert isinstance(json.loads(blob), list)


def test_parse_response_extracts_think_and_calls():
    raw = """<think>reasoning here</think>
motivation text
<tool_call>{"name": "remove_track", "arguments": {"track_name": "drums"}}</tool_call>"""
    out = parse_response(raw)
    assert out.is_valid()
    assert out.think == "reasoning here"
    assert "motivation text" in out.motivation
    assert out.tool_calls == [{"name": "remove_track", "arguments": {"track_name": "drums"}}]


def test_parse_response_handles_malformed_json():
    raw = '<tool_call>{not json}</tool_call><tool_call>{"name":"remove_track","arguments":{"track_name":"x"}}</tool_call>'
    out = parse_response(raw)
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["name"] == "remove_track"

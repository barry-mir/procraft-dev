"""Dump the Hermes tool-call schemas Qwen3 sees at generation time.

Usage:
    python scripts/dump_tool_schemas.py              # pretty-print summary
    python scripts/dump_tool_schemas.py --json       # full JSON to stdout
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procraft_data.tools import schemas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="Dump full Hermes system prompt.")
    args = ap.parse_args()

    tools = schemas.all_tool_schemas()
    if args.json:
        example_meta = ("tracks: piano (program 0), bass (program 33); "
                        "tempo 92 bpm; duration 10s")
        print(schemas.build_hermes_system_prompt(example_meta))
        return

    print(f"{len(tools)} top-level tools:")
    for t in tools:
        fn = t["function"]
        props = fn["parameters"].get("properties", {})
        keys = list(props)
        print(f"  {fn['name']:22s}  params={keys}")
        if fn["name"] == "apply_fx":
            branches = props["call"]["oneOf"]
            print(f"    oneOf branches: {len(branches)} effects "
                  f"(first = {branches[0]['properties']['effect']['const']})")

    full_prompt = schemas.build_hermes_system_prompt("tracks: piano, bass; tempo 92 bpm")
    print(f"\nfull system prompt length: {len(full_prompt)} chars")


if __name__ == "__main__":
    main()

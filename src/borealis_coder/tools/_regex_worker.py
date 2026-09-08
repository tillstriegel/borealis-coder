"""Isolated regular-expression matching for the search tool."""

from __future__ import annotations

import json
import re
import sys


def _reply(value: object) -> None:
    print(json.dumps(value, ensure_ascii=True), flush=True)


def main() -> None:
    configuration = json.loads(sys.stdin.readline())
    try:
        expression = re.compile(configuration["pattern"], configuration["flags"])
    except (re.error, OverflowError, RecursionError) as error:
        _reply({"error": f"Invalid regular expression: {error}"})
        return
    _reply({"ready": True})
    for line in sys.stdin:
        request = json.loads(line)
        matches = []
        for index, text in enumerate(request["lines"]):
            if expression.search(text):
                matches.append(index)
                if len(matches) >= request["limit"]:
                    break
        _reply({"matches": matches})


if __name__ == "__main__":
    main()

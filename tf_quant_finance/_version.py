from __future__ import annotations

import re


_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


def version_tuple(value: str) -> tuple[int, int, int]:
    match = _VERSION_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"invalid_version:{value}")
    return tuple(int(part or 0) for part in match.groups())

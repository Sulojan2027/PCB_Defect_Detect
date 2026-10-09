from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_overrides(args: list[str]) -> dict:
    """Parse Ultralytics-style `key=value` CLI overrides, YAML-typed (`epochs=5` -> int)."""
    overrides = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Override must look like key=value, got {arg!r}")
        key, value = arg.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    return overrides

#!/usr/bin/env python3
"""Initialize developer identity."""

from pathlib import Path
import sys


def init_developer(name: str) -> None:
    """Initialize developer identity file."""
    developer_file = Path(__file__).parent.parent / ".developer"
    developer_file.write_text(name.strip())
    print(f"Initialized developer: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: init_developer.py <name>")
        sys.exit(1)

    init_developer(sys.argv[1])

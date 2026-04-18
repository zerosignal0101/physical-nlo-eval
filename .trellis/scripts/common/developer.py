#!/usr/bin/env python3
"""Get developer identity."""

from pathlib import Path


def get_developer() -> str:
    """Get current developer name."""
    developer_file = Path(__file__).parent.parent / ".developer"

    if developer_file.exists():
        return developer_file.read_text().strip()

    return "unknown"


if __name__ == "__main__":
    print(get_developer())

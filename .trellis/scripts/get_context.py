#!/usr/bin/env python3
"""Get session context."""

import json
from pathlib import Path


def get_context() -> str:
    """Get full context as string."""
    project_root = Path(__file__).parent.parent.parent

    context = {
        "project": "physics-nlo-eval",
        "root": str(project_root),
        "developer": get_developer(),
    }

    return json.dumps(context, indent=2)


def get_context_json() -> str:
    """Get context as JSON string."""
    return get_context()


def get_developer() -> str:
    """Get current developer name."""
    developer_file = Path(__file__).parent.parent / ".developer"

    if developer_file.exists():
        return developer_file.read_text().strip()

    return "unknown"


if __name__ == "__main__":
    print(get_context())

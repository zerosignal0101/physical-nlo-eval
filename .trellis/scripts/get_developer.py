#!/usr/bin/env python3
"""Get developer identity - entry point."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from common.developer import get_developer

if __name__ == "__main__":
    print(get_developer())

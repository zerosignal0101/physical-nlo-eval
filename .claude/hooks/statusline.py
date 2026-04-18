#!/usr/bin/env python3
"""Status line hook - shows current task in status bar."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from trellis.scripts.get_developer import get_developer
    developer = get_developer()
    print(f"[{developer}]")
except Exception:
    print("[physics-nlo-eval]")

#!/usr/bin/env python3
"""Ralph loop hook for check agent."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from trellis.scripts.task import get_current_task
    task = get_current_task()
    if task:
        print(f"Continuing task: {task}")
except Exception:
    pass

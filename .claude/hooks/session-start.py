#!/usr/bin/env python3
"""Session start hook - displays context on session start."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from trellis.scripts.get_context import get_context
    context = get_context()
    print(context)
except Exception as e:
    print(f"Warning: Could not load session context: {e}")

#!/usr/bin/env python3
"""Inject context for subagent calls."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from trellis.scripts.get_context import get_context_json
    context = get_context_json()
    print(context)
except Exception as e:
    print(json.dumps({"error": str(e)}))

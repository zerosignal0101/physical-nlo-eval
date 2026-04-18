#!/usr/bin/env python3
"""Add session to journal."""

import sys
from pathlib import Path
from datetime import datetime


def add_session(title: str, commit: str | None = None, summary: str | None = None) -> None:
    """Add session to journal."""
    project_root = Path(__file__).parent.parent.parent
    workspace_dir = project_root / ".trellis" / "workspace"

    developer = (project_root / ".trellis" / ".developer").read_text().strip()
    developer_dir = workspace_dir / developer

    if not developer_dir.exists():
        developer_dir.mkdir(parents=True)

    # Find latest journal
    journals = sorted(developer_dir.glob("journal-*.md"))
    current_journal = journals[-1] if journals else None

    if current_journal is None or current_journal.stat().st_size > 100000:
        journal_num = len(journals) + 1
        current_journal = developer_dir / f"journal-{journal_num}.md"
        current_journal.write_text(f"# Session Journal - {developer}\n\n")

    # Append session
    with open(current_journal, "a", encoding="utf-8") as f:
        f.write(f"\n## Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Topic**: {title}\n\n")
        if commit:
            f.write(f"**Commit**: {commit}\n\n")
        if summary:
            f.write(f"**Summary**: {summary}\n\n")


if __name__ == "__main__":
    title = "Untitled"
    commit = None
    summary = None

    args = sys.argv[1:]
    while args:
        if args[0] == "--title" and len(args) > 1:
            title = args[1]
            args = args[2:]
        elif args[0] == "--commit" and len(args) > 1:
            commit = args[1]
            args = args[2:]
        elif args[0] == "--summary" and len(args) > 1:
            summary = args[1]
            args = args[2:]
        else:
            args = args[1:]

    add_session(title, commit, summary)

#!/usr/bin/env python3
"""Task management script."""

import json
import sys
from pathlib import Path
from datetime import datetime


def get_current_task() -> str | None:
    """Get current task path."""
    current_task_file = Path(__file__).parent.parent / ".current-task"

    if current_task_file.exists():
        return current_task_file.read_text().strip()

    return None


def list_tasks() -> list[dict]:
    """List active tasks."""
    tasks_dir = Path(__file__).parent.parent / "tasks"

    if not tasks_dir.exists():
        return []

    tasks = []
    for task_dir in tasks_dir.iterdir():
        if task_dir.is_dir():
            task_file = task_dir / "task.json"
            if task_file.exists():
                tasks.append(json.loads(task_file.read_text()))

    return tasks


def create_task(title: str, slug: str | None = None) -> None:
    """Create a new task."""
    tasks_dir = Path(__file__).parent.parent / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    if slug is None:
        slug = title.lower().replace(" ", "-")

    date_str = datetime.now().strftime("%m-%d")
    task_dir = tasks_dir / f"{date_str}-{slug}"
    task_dir.mkdir(exist_ok=True)

    task = {
        "title": title,
        "slug": slug,
        "created": datetime.now().isoformat(),
        "status": "active",
    }

    (task_dir / "task.json").write_text(json.dumps(task, indent=2))
    print(f"Created task: {title} ({task_dir})")


def finish_task() -> None:
    """Clear current task."""
    current_task_file = Path(__file__).parent.parent / ".current-task"

    if current_task_file.exists():
        current_task_file.unlink()
        print("Task finished")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: task.py [list|create|start|finish]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        tasks = list_tasks()
        for task in tasks:
            print(f"- {task['title']} ({task['slug']})")
    elif command == "create":
        title = sys.argv[2] if len(sys.argv) > 2 else "Untitled"
        slug = sys.argv[3] if len(sys.argv) > 3 else None
        create_task(title, slug)
    elif command == "finish":
        finish_task()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

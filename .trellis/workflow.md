# Development Workflow

> Based on [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

---

## Table of Contents

1. [Quick Start (Do This First)](#quick-start-do-this-first)
2. [Workflow Overview](#workflow-overview)
3. [Session Start Process](#session-start-process)
4. [Development Process](#development-process)
5. [Session End](#session-end)
6. [File Descriptions](#file-descriptions)
7. [Best Practices](#best-practices)

---

## Quick Start (Do This First)

### Step 0: Initialize Developer Identity (First Time Only)

```bash
# Check if already initialized
python3 ./.trellis/scripts/get_developer.py

# If not initialized, run:
python3 ./.trellis/scripts/init_developer.py <your-name>
```

### Step 1: Understand Current Context

```bash
python3 ./.trellis/scripts/get_context.py
```

### Step 2: Read Project Guidelines [MANDATORY]

```bash
python3 ./.trellis/scripts/get_context.py --mode packages
cat .trellis/spec/backend/index.md
```

---

## Workflow Overview

### Core Principles

1. **Read Before Write** - Understand context before starting
2. **Follow Standards** - [!] **MUST read `.trellis/spec/` guidelines before coding**
3. **Incremental Development** - Complete one task at a time
4. **Record Promptly** - Update tracking files immediately after completion
5. **Document Limits** - [!] **Max 2000 lines per journal document**

---

## Session Start Process

### Step 1: Get Session Context

```bash
python3 ./.trellis/scripts/get_context.py
```

### Step 2: Read Development Guidelines [!] REQUIRED

```bash
python3 ./.trellis/scripts/get_context.py --mode packages
cat .trellis/spec/backend/index.md
```

### Step 3: Select Task to Develop

```bash
python3 ./.trellis/scripts/task.py list
python3 ./.trellis/scripts/task.py create "<title>" --slug <task-name>
```

---

## Development Process

### Task Development Flow

```
1. Create or select task
2. Start task
3. Write code according to guidelines
4. Self-test (lint, type check, tests)
5. Commit code
6. Record session
7. Finish task
```

---

## Session End

### One-Click Session Recording

```bash
python3 ./.trellis/scripts/add_session.py \
  --title "Session Title" \
  --commit "abc1234" \
  --summary "Brief summary"
```

---

## File Descriptions

### workspace/ - Developer Workspaces

Record each AI Agent session's work content.

### spec/ - Development Guidelines

Documented standards for consistent development.

### Tasks - Task Tracking

```
tasks/
|-- {MM-DD}-{name}/
|   +-- task.json
+-- archive/
```

---

## Best Practices

### [OK] DO - Should Do

1. **Before session start**: Run `get_context.py`, read guidelines
2. **During development**: Follow spec guidelines, develop one task at a time
3. **After development complete**: Use `/trellis:finish-work`, record session

### [X] DON'T - Should Not Do

1. [!] **Don't** skip reading `.trellis/spec/` guidelines
2. **Don't** let journal exceed 2000 lines
3. **Don't** commit code with lint/test errors
4. [!] **Don't** execute `git commit` - AI should not commit code

---

## Summary

Following this workflow ensures:

- Continuity across multiple sessions
- Consistent code quality
- Trackable progress
- Knowledge accumulation in spec docs

**Core Philosophy**: Read before write, follow standards, record promptly, capture learnings

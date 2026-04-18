# Code Reuse Thinking Guide

> Identify patterns and reduce duplication before coding.

---

## Purpose

Before writing similar code to something that exists, or creating new utility functions, think about reuse. This prevents the "copy-paste修改" spiral that leads to hard-to-maintain code.

---

## When to Apply

Apply this guide when:

- [ ] You're writing similar code to something that exists
- [ ] You see the same pattern repeated 3+ times
- [ ] You're adding a new field to multiple places
- [ ] **You're modifying any constant or config**
- [ ] **You're creating a new utility/helper function** ← Search first!

---

## Checklist

### Step 1: Search Before You Write

```bash
# Search for similar functionality
grep -r "similar_function_name" src/
```

### Step 2: Identify the Pattern

Ask yourself:
- Is this a new pattern or an existing one?
- Can the existing code be generalized?
- Should this be a parameter rather than duplication?

### Step 3: Choose the Right Approach

**Option A: Reuse existing** - If similar code exists, modify it to be more general
**Option B: Extract common** - If 3+ places have similar code, extract to shared function
**Option C: Parameterize** - If only difference is values, use configuration/parameters
**Option D: Create new** - Only if truly novel functionality

---

## Common Cases in Physics Simulation

### Case 1: Physical Constants

**Bad**: Hardcoding `np.pi * 2` in multiple places
**Good**: Define `TWO_PI = np.pi * 2` in `core/constants.py`

### Case 2: Array Operations

**Bad**: Repeating `np.fft.fftshift` + `np.fft.ifft` pattern
**Good**: Create `ifftshift_ifft()` helper in `physics/dsp/freq_domain.py`

### Case 3: Signal Processing Chains

**Bad**: Copy-pasting filter coefficients calculation
**Good**: Create shared `create_gaussian_filter()` function

---

## Red Flags

- [ ] Similar code blocks > 5 lines that differ only in variable names
- [ ] "Just copy this and change X" in comments
- [ ] Same bug fix applied in multiple places
- [ ] Constants defined in multiple files

---

## Refactoring Tips

1. **Small steps** - Don't try to refactor everything at once
2. **Test after each change** - Ensure behavior unchanged
3. **Use types** - Make sure refactored code is well-typed for mypy
4. **Document the why** - If extraction is non-obvious, add comment

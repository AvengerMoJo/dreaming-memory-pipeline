# Optimizer Artifact Policy

## Purpose
Keep the submodule focused on reusable memory/dream implementation code.

## Rule
- `optimizer/` is treated as local experimentation and tuning workspace by default.
- It is ignored by git in this submodule (`.gitignore`) to avoid noisy untracked artifacts.

## If You Need To Version Optimizer Work
1. Move production-worthy optimizer code into a tracked module path under `src/` or `tests/`.
2. Add documentation for the promoted component and its ownership.
3. Keep temporary experiments, ad-hoc logs, and one-off scripts out of tracked history.

## Rationale
This prevents accidental coupling of core module history with local benchmark/tuning iterations,
and keeps modular interfaces stable for independent teams.

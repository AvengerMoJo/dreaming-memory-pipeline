# Memory + Dream Workflow Standard (Submodule Owned)

Status: Active
Owner: `submodules/dreaming-memory-pipeline`

## Intent

All core implementation of memory tier logic and dreaming pipeline logic is owned in this submodule so contributors can improve it independently from the main app shell.

## Ownership Boundary

### Submodule owns implementation

- Memory tier implementations (`working`, `active`, `archival`, `knowledge`, embeddings)
- Memory services (`MemoryService`, `HybridMemoryService`)
- Dreaming pipeline implementation (A→B→C→D)
- Optimization/benchmark utilities directly tied to memory+dream internals

### Main app owns integration shell

- Scheduler orchestration and task wiring
- MCP tool surface and policy gates
- Backward-compatible import shims for legacy paths

## Runtime Contract

Main app imports continue to work through compatibility shims:

- `app.memory.*` -> delegates to `mojo_memory.memory.*`
- `app.services.memory_service` -> delegates to `mojo_memory.services.memory_service`
- `app.services.hybrid_memory_service` -> delegates to `mojo_memory.services.hybrid_memory_service`

This preserves existing call sites while allowing implementation work to happen in one place.

## Contribution Rule

If a change modifies memory/dream behavior, implement it in submodule paths first:

- `src/mojo_memory/**`
- `src/dreaming/**`

Do not place new memory/dream core logic under `app/memory` or `app/services`.

## Migration Note

Legacy modules in main app are now compatibility shims only. Future cleanups can remove shims after all dependent imports are migrated to submodule-native paths.

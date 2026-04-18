# ABCD Gap Review

Date: 2026-04-18

Purpose: record the current implementation gaps between the intended ABCD dream memory architecture and the code that exists today in `dreaming-memory-pipeline`.

This is a gap review, not a final redesign. It should be read before changing the pipeline or making strong benchmark claims.

## Intended Shape

The intended architecture is:

- `A` = original source evidence
- `B` = broken-down atomic units derived from A
- `C` = clustered entity / relationship / timeline / conflict structures over B
- `D` = dynamic surfaced current view over C, with lineage back to A/B/C

The current implementation is much closer to:

- `A` = raw conversation text
- `B` = semantic text chunks with attached metadata
- `C` = LLM-written summaries / topic groupings over those chunks
- `D` = versioned JSON snapshot of B + C

That means the current module is a useful archive-writing foundation, but not yet a full ABCD memory engine.

## High-Impact Gaps

### 1. D is not a real dynamic surfaced view

Impact: critical

Current behavior:
- `process_conversation()` writes a versioned archive.
- `_create_archive_data()` serializes `b_chunks`, `c_clusters`, `entities`, and metadata.
- There is no answer-time `D` selection or surfacing logic.
- There is no “latest valid answer” resolver for versioned or contradictory facts.

Evidence:
- `src/dreaming/pipeline.py:197`
- `src/dreaming/pipeline.py:244`

Why this matters:
- The current `D` is just a stored snapshot.
- Your intended design requires `D` to act like a current surfaced answer layer.
- Example: if ten historical password mentions exist, `D` should surface the latest password, not merely store all prior states.

### 2. B is not truly atomic

Impact: critical

Current behavior:
- The chunker only creates `ChunkType.SEMANTIC`.
- `key_facts` exist only as ad hoc fields on `BChunk.__dict__`.
- `embedding` is left as `None`.
- `token_range` is synthetic (`i * 400`), not derived from real offsets.

Evidence:
- `src/dreaming/chunker.py:260`
- `src/dreaming/models.py:38`

Why this matters:
- `B` should be a durable fact substrate, not just semantic segments.
- Without first-class atomic fact units, identity/version/current-state questions remain brittle.

### 3. C is under-populated and not entity-grade

Impact: critical

Current behavior:
- `CCluster` supports rich fields:
  - `participants`
  - `time_span_start`
  - `time_span_end`
  - `contradictions_resolved`
  - `embedding`
  - `version`
- The synthesizer mostly sets:
  - `id`
  - `cluster_type`
  - `content`
  - `related_chunks`
  - `related_clusters`
  - `theme`
  - `confidence`
- `key_facts` are attached ad hoc via `__dict__`.

Evidence:
- `src/dreaming/models.py:93`
- `src/dreaming/synthesizer.py:252`

Why this matters:
- Your intended `C` is an entity/relationship/timeline layer.
- The current implementation is closer to “clustered summaries.”

### 4. No real contradiction or latest-state resolution

Impact: critical

Current behavior:
- Versioning exists at archive level.
- Contradiction resolution is not actually performed inside the conversation pipeline.
- `contradictions_resolved` exists in the model but is never materially populated.

Evidence:
- `src/dreaming/models.py:119`
- `src/dreaming/pipeline.py:83`

Why this matters:
- ABCD must become useful exactly where simple RAG fails:
  - updated passwords
  - newer configurations
  - changed statuses
  - conflicting facts over time

### 5. Cross-conversation synthesis is missing

Impact: high

Current behavior:
- `process_conversation()` processes one conversation/session at a time.
- `synthesize_chunks()` clusters only the chunks from that one call.
- There is no global clustering pass across conversations.

Evidence:
- `src/dreaming/pipeline.py:142`
- `src/dreaming/synthesizer.py:76`

Why this matters:
- The intended `C` should accumulate meaning across memory growth.
- Right now the pipeline primarily produces per-conversation islands.

### 6. Relationships are effectively unimplemented

Impact: high

Current behavior:
- `_create_archive_data()` initializes `relationships = []`.
- It never fills the list.

Evidence:
- `src/dreaming/pipeline.py:255`

Why this matters:
- The archive shape advertises relationship support.
- The data written to disk does not actually contain those relationships.

## Retrieval / Consumption Gaps

### 7. ABCD output is not consumed as ABCD

Impact: critical

Current behavior:
- Runtime integration later indexes C-cluster text into the generic knowledge base.
- The app then retrieves cluster text through the normal knowledge base search path.
- This is a flattening workaround, not first-class ABCD retrieval.

Evidence:
- `app/scheduler/executor.py:270`
- `app/scheduler/executor.py:320`

Why this matters:
- The system is not reasoning over B/C/D directly.
- It is mostly reasoning over re-embedded flattened text.

### 8. Role chat does not really consume conversation dreaming output

Impact: high

Current behavior:
- Role chat primarily reads:
  - `knowledge_units` archives
  - task reports
- It does not query conversation B/C/D layers as first-class structures.

Evidence:
- `app/scheduler/role_chat.py:518`

Why this matters:
- Conversation dreaming is not yet a strong memory surface for debrief / recall.
- The document path is currently more retrieval-useful than the conversation path.

### 9. D lifecycle metadata is stored but not used as a query layer

Impact: high

Current behavior:
- JSON storage persists:
  - `latest_version`
  - `status`
  - `storage_location`
  - `previous_version`
  - `supersedes_version`
- But retrieval does not use those fields structurally at answer time.

Evidence:
- `src/dreaming/storage/json_backend.py:74`
- `src/dreaming/storage/json_backend.py:128`

Why this matters:
- A real `D` layer should answer:
  - what is active
  - what is stale
  - what superseded what
  - what the current surfaced answer should be

## Data Quality Gaps

### 10. Quality levels are mostly metadata, not meaningfully different pipelines

Impact: medium

Current behavior:
- `basic/good/premium` mainly influence confidence and upgrade flags.
- They do not strongly alter extraction or synthesis strategy.

Evidence:
- `src/dreaming/pipeline.py:34`
- `src/dreaming/chunker.py:286`
- `src/dreaming/synthesizer.py:279`

### 11. Fallbacks are too lossy for a memory engine

Impact: medium

Current behavior:
- Chunker fallback becomes paragraph splits with empty labels/entities.
- Synthesizer fallback becomes label grouping with generic topics.

Evidence:
- `src/dreaming/chunker.py:316`
- `src/dreaming/synthesizer.py:304`

Why this matters:
- A memory system should degrade gracefully without discarding too much structure.

### 12. Embeddings exist in the models but are not produced in the conversation path

Impact: medium

Current behavior:
- `BChunk.embedding` and `CCluster.embedding` exist in the datamodel.
- The conversation pipeline leaves them as `None`.

Evidence:
- `src/dreaming/models.py:62`
- `src/dreaming/models.py:122`
- `src/dreaming/chunker.py:289`

Why this matters:
- Searchability depends on later indexing hacks instead of the ABCD layer owning its own retrievability.

## Integration / Operational Gaps

### 13. The module is still operationally external to the main app

Impact: medium

Current behavior:
- `app/dreaming` is a shim.
- Real implementation lives in the submodule / installed package.
- If the submodule is missing, dreaming is unavailable.

Evidence:
- `app/dreaming/__init__.py:14`

Why this matters:
- This is a core subsystem but still behaves like an optional external component.

### 14. Scheduler integration treats dreaming as storage and indexing, not as a memory contract

Impact: high

Current behavior:
- After dreaming, the executor records counts and archive metadata.
- Then it indexes cluster text into the knowledge base to make anything searchable.

Evidence:
- `app/scheduler/executor.py:259`
- `app/scheduler/executor.py:270`

Why this matters:
- The rest of the product still depends on the generic knowledge base abstraction, not the ABCD contract itself.

## One Important Positive

The document path is materially stronger than the conversation path.

`process_document()` extracts `KnowledgeUnit`s that are much closer to your intended `B` layer:
- atomic proposition
- supporting quote
- links
- entity/label overlap

Evidence:
- `src/dreaming/pipeline.py:311`
- `src/dreaming/models.py:207`

This suggests the right direction may be:
- make conversation dreaming more like the document atomic-fact path
- then build real C and D on top of those units

## Summary Judgment

Current state:

- good archive writing: yes
- useful semantic consolidation: partly
- native ABCD retrieval engine: no
- real D surfaced current-view layer: no
- robust latest / version / contradiction resolution: no
- production-ready long-term memory engine in the intended sense: not yet

## Recommended Next Questions

Before implementation work, the right questions are:

1. Is the intended ABCD estimation correct?
2. Should the method be validated against the current retrieval path before deeper refactor?
3. Should benchmarking continue as the primary refinement loop?

Those should be discussed before changing the module design.

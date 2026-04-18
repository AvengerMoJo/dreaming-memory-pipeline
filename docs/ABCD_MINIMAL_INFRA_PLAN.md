# ABCD Minimal Infra Plan

Date: 2026-04-18

Purpose: define the minimum infrastructure required before benchmarking the full ABCD thesis.

This document is intentionally narrower than the full ABCD vision. It captures the smallest practical architecture that can support:

1. current most likely answer
2. prior historical answers
3. provenance for both

Without this infra, benchmarking will mostly prove that the current system does not yet implement the layer we actually want.

## Current Decision

Benchmarking should not lead implementation blindly right now.

First build the minimum ABCD infra that can express:
- latest/current surfaced value
- historical prior values
- provenance back to source

Then benchmark that.

## Stable Concept Boundaries

### A = Authentic Data

- immutable
- raw, untouched, preserved
- JSON-first
- contains original conversation context and metadata
- may be wrong historically, but must remain authentic

### B = Basic Units

- derived from A
- maintainable, not immutable
- can be soft-deleted / repaired by doctor processes
- should carry:
  - global `bu_id`
  - parent `ad_id`
  - local BU enum/index within AD extraction
  - `snippet` = exact original conversation text fragment
  - `description` = retrieval-oriented digest
- B is not naive chunking
- B may contain overlapping or duplicated semantic units on purpose

### C = Cluster Map

- relationship preparation layer
- many-to-many by default
- built daily from new BUs, but allowed to consult older C snapshots
- old C snapshots are preserved
- daily C final is generated from:
  - new BU processing
  - old C linking / version-awareness
- C should support at minimum:
  - same topic
  - same entity
  - older/newer
  - correction/update
  - duplicate/near-duplicate
- C is disposable and rebuildable
- C can be marked:
  - `active`
  - `archived`

### D = Dynamic Relationship / Daily Ontology View

- high-level surfaced view layer
- generated daily
- not fixed forever
- uses current C plus yesterday's linked D summaries as context
- one ontology identity can have many dated views
- D is not archive in the same sense as C
- old D views remain historically meaningful
- D may contain contradictory views across time
- D should not be treated as canonical truth
- D feeds future knowledge/taste shaping only through later Bonsai processes, not directly as truth

## Minimal Infra Goal

The first implementation target is not “full ontology.”

It is this narrower goal:

### For one tracked entity cluster, the system must support:

1. identify the current most likely active value
2. preserve older values as historical prior states
3. show provenance back to AD and BU

Example:
- site password changed many times
- old passwords remain historically valid for their time
- C understands ordering
- D surfaces the current likely password
- provenance points back to AD + BU

## Minimal C Infra Required

Before full benchmarking, C needs these minimum capabilities:

### 1. Stable cluster identity

Each C cluster should have:
- `cm_id`
- title / description
- status: `active | archived`
- timestamps

### 2. BU membership

Each C cluster should link to many BUs:
- `bu_ids[]`

### 3. Relationship typing

At minimum:
- `same_topic`
- `same_entity`
- `older_newer`
- `correction_update`
- `duplicate_near_duplicate`

### 4. Latest/prior ordering

For clusters that represent evolving values, C must track:
- ordered BU lineage
- latest BU candidate
- prior BU candidates

This does not need perfect ontology. It just needs enough structure to say:
- BU88 older
- BU10234 newer
- BU10234 current best candidate

### 5. Per-cluster JSON + daily manifest

C storage should be:
- per-cluster JSON records
- daily manifest/index

Old C remains preserved.
New C final is produced daily.

## Minimal D Infra Required

Before full benchmarking, D needs these minimum capabilities:

### 1. Ontology identity

Ontology identity should be based on:
- primary: title/description similarity
- secondary: overlapping CM clusters

### 2. D view structure

Each D view should include:
- `ontology_id`
- dated snapshot identity
- surfaced summary
- linked `cm_ids`
- optional top `bu_ids`
- timestamp

### 3. One ontology, many dated views

Do not create a brand new ontology every time.

Instead:
- one ontology identity
- many dated D snapshots/views under it

### 4. Multiple D views allowed

One C cluster may contribute to multiple D views.
One D view may link to many C clusters.

### 5. D history preserved

Old D views are not deleted.
They are historical ontology snapshots.

## Storage Rules

### A

- immutable JSON
- virtual `ad_id`
- physical file path separate from logical identity

### B

- JSON-first
- global `bu_id`
- parent `ad_id`
- local BU enum/index within AD extraction
- soft delete by default
- hard delete only via doctor cleanup if needed

### C

- per-cluster JSON
- daily manifest/index
- old snapshots preserved
- active/archive status allowed

### D

- per-view JSON
- daily manifest/index
- old snapshots preserved
- no archive lifecycle required like C

## Doctor / Maintenance Rules

Only A is truly untouchable.

B is maintainable.
C and D are rebuildable.

The data doctor should eventually detect:
- wrong BU id generation
- orphan BU
- stale C links
- missing BU references in C
- duplicate BU candidates

For now:
- B can be corrected or soft-deleted
- C can be rebuilt
- D can be regenerated

## First Benchmark Target

The first benchmark target should be:

### Latest-value retrieval

Synthetic controlled corpus:
- multiple entities
- multiple updates over time
- wrong intermediate values
- later corrections

Scored outputs:
1. current most likely answer
2. historical prior answers
3. provenance via `ad_id + bu_id`

This is the first benchmark because it directly tests:
- B preservation
- C sequencing
- D surfaced current view

## What Not To Do Yet

Do not try to prove all of ABCD at once.

Do not benchmark:
- broad ontology intelligence
- human-style taste
- open-ended social meaning
- arbitrary cross-domain relational reasoning

before the infra can even express:
- current value
- historical value
- provenance

## Practical Next Step

Implement the smallest C/D layer that can support:

- cluster with ordered BU history
- daily D view with linked CM ids
- answer path:
  - current most likely value
  - prior historical values
  - provenance

Only after that should comparative benchmarking drive further refinement.

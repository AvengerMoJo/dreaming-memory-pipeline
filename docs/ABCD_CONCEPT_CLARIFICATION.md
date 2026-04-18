# ABCD Concept Clarification

Date: 2026-04-18

Purpose: preserve the intended conceptual model behind ABCD before implementation details distort it.

This document is not an implementation spec. It is a conceptual clarification of what ABCD is supposed to mean.

## Why This Exists

ABCD did not start as a clean academic taxonomy. It emerged from implementation work across earlier projects and from dissatisfaction with how conversational memory is usually handled.

The core disagreement is not with decomposition itself.

The disagreement is with **context-destructive chunking**.

Conversation is not the same kind of source material as:
- books
- key-value records
- static documents
- runtime views

Conversation is highly context-sensitive. Cutting too much destroys meaning. Keeping too much blocks reuse and search. The problem is `过犹不及` — both over-cutting and under-cutting are wrong.

ABCD is an attempt to preserve authenticity and context while still allowing decomposition, reuse, clustering, and surfacing over time.

## Core Model

### A = Authentic Data

`A` means **Authentic Data**.

This is the untouched, raw source.

For conversation memory, that means:
- the original JSON
- timestamp metadata
- participant structure
- full sequence and context

The goal of A is not to be “clean” or “correct.”

The goal of A is:
- authentic
- preserved
- auditable
- context-complete

Important:
- A may contain mistakes
- A may contain contradictions
- A may contain partial or emotional or messy statements

That is acceptable.

A is the preserved source of truth in the historical sense, not the final interpreted truth.

This matters because conversational meaning depends on:
- what came before
- what came after
- who said it
- when it happened

Naive chunking destroys this too easily.

### B = Basic Units

`B` means **Basic Units** derived from A.

The purpose of B is not to replace A.

The purpose of B is to create reusable meaningful units while preserving the ability to reconnect them to the original context.

The intended structure is:

- `A`
- `A summary`
- `B units linked to that A summary`

So B is not just random snippets.

It should include:
- a summary of A as a metadata anchor
- meaningful decomposed units
- digested descriptions of those units
- optionally direct snippet-level embeddings

This allows two kinds of retrieval in parallel:

1. whole-picture retrieval
- use the A summary as the fast thumbnail / routing layer

2. fine-grained retrieval
- use the B units themselves for detailed matching

Analogy:
- A = Alex’s full photo
- B = Alex’s face, hand, leg, body

The point is not to cut Alex into nonsense pieces.

The point is to create reusable meaningful units **while retaining the ability to recover the whole picture through A and A-summary linkage**.

So the intended shape is closer to:

- `A -> [Summary -> BUs...]`
- each `BU -> [source snippet + digested description]`

### C = Cluster Map

`C` means **Cluster Map**.

The purpose of C is to create new visibility across B units and their originating A sources.

This is where siloed source memories start becoming connected memory.

Examples:
- same topic across multiple conversations
- repeated references to a person or system
- timeline of a repeated issue
- cross-linked chains such as `x -> y`, `y -> z`, therefore `x -> y -> z`

Important:
- C is not yet the final surfaced answer layer
- C is not necessarily one fixed entity graph
- C should allow multiple possible relationship interpretations

This matters because a single fixed entity worldview can be too rigid.

Depending on the domain and the query, different relationship surfaces may matter:
- timeline
- identity
- technical dependency
- version history
- emotional salience

So C should be thought of as a map-building layer, not a final truth lock.

The intended shape is:

- `C = [clustered BUs + descriptions + linkability back to A]`

### D = Dynamic Relationship

`D` means **Dynamic Relationship**.

This is the original “dreaming” idea in the strongest sense.

D is not just archive.

D is the active surfaced view created from:
- C
- B
- time
- recency
- repetition
- contradiction
- current salience

Its jobs are:

1. consolidate
- reduce repeated or overlapping structures

2. surface active clusters
- what is hot now
- what is most relevant now
- what has changed recently

3. sequence historical versions
- not just latest
- but latest in relation to prior versions

4. create answer-oriented views
- the surface that should be used when someone asks a question

Example:
- A returns 10 files mentioning passwords
- B returns 10 password-related unit versions
- C forms 1 password cluster linking those versions
- D surfaces:
  - the current/latest password
  - while still preserving the historical lineage

That is why D is dynamic.

It is not simply:
- older storage
- colder storage
- archive folder

It is a relationship-aware active view over accumulated memory.

## Important Distinctions

### ABCD is not ordinary chunking

Ordinary chunking often assumes:
- cut text into pieces
- embed pieces
- retrieve pieces

ABCD says that is not enough for conversation memory.

Conversation requires:
- preserved authenticity
- preserved surrounding context
- decomposed but meaningful units
- cross-unit clustering
- surfaced dynamic views

### A is not “truth”

A is authentic source, not perfect truth.

Truth may emerge later through:
- reconciliation
- contradiction handling
- surfacing
- temporal sequencing

That work belongs more to C and D than to A.

### B is not allowed to sever A

The whole point is to decompose without losing the original recoverable context.

### C is not forced into one universal graph worldview

Cluster maps may differ depending on:
- domain
- query type
- perspective
- time window

### D is the surfaced memory layer

If ABCD is implemented correctly, D should feel like:
- the memory surface a human has after sleeping on things
- not just the raw archive of what happened

## Generic, Not MoJo-Specific

This architecture is intended as a generic dreaming-memory model.

It should not be tied too tightly to:
- MoJoAssistant
- any one role system
- one storage backend
- one specific retrieval engine

MoJo is one implementation environment.

ABCD itself is meant to be broader:
- a general architecture for memory consolidation
- especially for conversational and context-sensitive memory systems

## Practical Summary

The intended model is:

- `A` = preserve the raw authentic source
- `B` = create meaningful decomposed units without severing context
- `C` = build relationship maps and cross-source visibility
- `D` = surface the active, consolidated, answer-oriented dynamic view

If implementation drifts into:
- A = raw text
- B = chunks
- C = summaries
- D = archive files

then the concept has been flattened too much.

That may still produce useful software, but it is not the full intended ABCD idea.

## What This Document Is For

This clarification should be used before:
- rewriting the pipeline
- overfitting benchmarks
- replacing ABCD with a simpler graph-only model
- claiming that the current implementation already matches the intended design

It is a conceptual anchor.

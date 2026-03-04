# Dreaming Memory Pipeline

A pluggable memory consolidation pipeline that transforms raw conversations into a structured, versioned knowledge base. Designed to be embedded in any AI assistant, chatbot, or knowledge management system.

## What It Does

The pipeline processes conversations through four stages:

```
A (Raw Text) --> B (Semantic Chunks) --> C (Synthesized Clusters) --> D (Versioned Archive)
```

- **A: Raw input** -- full conversation text, any language
- **B: Chunking** -- LLM breaks the conversation into semantic pieces with labels, entities, and speaker attribution
- **C: Synthesis** -- related chunks are grouped into topic clusters with cross-references
- **D: Archival** -- the result is saved as an immutable versioned archive with lifecycle tracking

Each time the same conversation is re-processed (or upgraded to a higher quality LLM), a new version is created. Old versions are marked `superseded/cold` but never deleted.

## Install

```bash
pip install git+https://github.com/AvengerMoJo/dreaming-memory-pipeline.git

# With local LLM support (llama.cpp bindings)
pip install "dreaming-memory-pipeline[local] @ git+https://github.com/AvengerMoJo/dreaming-memory-pipeline.git"
```

Or clone and install in editable mode:

```bash
git clone https://github.com/AvengerMoJo/dreaming-memory-pipeline.git
cd dreaming-memory-pipeline
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

```python
import asyncio
from dreaming import DreamingPipeline

# Any object with a generate_response(query, context) method works
class MyLLM:
    def generate_response(self, query=None, context=None):
        # Call your LLM here (OpenAI, Anthropic, local, etc.)
        return call_my_llm(query)

pipeline = DreamingPipeline(llm_interface=MyLLM())

result = asyncio.run(pipeline.process_conversation(
    conversation_id="chat_001",
    conversation_text="User: How does caching work?\nAssistant: ...",
    metadata={"original_text": "User: How does caching work?\nAssistant: ..."}
))

print(result["status"])  # "success"
print(result["stages"]["B_chunks"]["count"])  # number of semantic chunks
print(result["stages"]["D_archive"]["version"])  # 1
```

Archives are saved to `~/.memory/dreams/` by default.

## Using the Built-in LLM Interfaces

The package includes ready-made LLM wrappers:

### API-based (OpenAI, Anthropic, any OpenAI-compatible endpoint)

```python
from dreaming.llm.api import APILLMInterface

llm = APILLMInterface(
    api_key="sk-...",
    model="gpt-4o-mini",
    api_url="https://api.openai.com/v1"
)
pipeline = DreamingPipeline(llm_interface=llm, quality_level="good")
```

### Local model (llama.cpp via llama-cpp-python)

```bash
pip install "dreaming-memory-pipeline[local]"
```

```python
from dreaming.llm.local import LocalLLMInterface

llm = LocalLLMInterface(
    model_path="~/.cache/dreaming/models/llama-3.2-3b-instruct-q4_k_m.gguf"
)
pipeline = DreamingPipeline(llm_interface=llm, quality_level="basic")
```

### Bring your own

Any object that implements `generate_response(query: str, context=None) -> str` works. No base class required -- duck typing is sufficient:

```python
class AnthropicLLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def generate_response(self, query=None, context=None):
        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": query}]
        )
        return response.content[0].text
```

## Storage Backends

The pipeline uses a `StorageBackend` interface for persistence. The default `JsonFileBackend` writes archives as JSON files. You can replace it with any backend.

### Default: JSON files

```python
from dreaming import DreamingPipeline

# Uses ~/.memory/dreams/ by default
pipeline = DreamingPipeline(llm_interface=llm)

# Or specify a custom path
pipeline = DreamingPipeline(llm_interface=llm, storage_path=Path("/data/archives"))
```

File layout:

```
~/.memory/dreams/
  chat_001/
    archive_v1.json
    archive_v2.json
    manifest.json
  chat_002/
    archive_v1.json
    manifest.json
```

### Writing a custom backend

Subclass `StorageBackend` and implement 6 methods:

```python
from dreaming.storage.base import StorageBackend

class SQLiteBackend(StorageBackend):

    def __init__(self, db_path: str):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS archives (
                conversation_id TEXT,
                version INTEGER,
                data TEXT,
                PRIMARY KEY (conversation_id, version)
            );
            CREATE TABLE IF NOT EXISTS manifests (
                conversation_id TEXT PRIMARY KEY,
                data TEXT
            );
        """)

    def save_archive(self, conversation_id: str, version: int, data: dict) -> None:
        import json
        self.conn.execute(
            "INSERT OR REPLACE INTO archives VALUES (?, ?, ?)",
            (conversation_id, version, json.dumps(data))
        )
        self.conn.commit()

    def load_archive(self, conversation_id: str, version: int | None = None) -> dict | None:
        import json
        if version is None:
            version = self.get_latest_version(conversation_id)
            if version == 0:
                return None
        row = self.conn.execute(
            "SELECT data FROM archives WHERE conversation_id = ? AND version = ?",
            (conversation_id, version)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list_archives(self) -> list[dict]:
        import json
        rows = self.conn.execute("""
            SELECT a.conversation_id, a.version, a.data
            FROM archives a
            INNER JOIN (
                SELECT conversation_id, MAX(version) AS max_v
                FROM archives GROUP BY conversation_id
            ) latest ON a.conversation_id = latest.conversation_id
                    AND a.version = latest.max_v
        """).fetchall()
        results = []
        for cid, version, raw in rows:
            data = json.loads(raw)
            results.append({
                "conversation_id": cid,
                "latest_version": version,
                "quality_level": data.get("quality_level", "unknown"),
                "created_at": data.get("created_at"),
                "entities_count": len(data.get("entities", [])),
                "chunks_count": len(data.get("b_chunks", [])),
                "clusters_count": len(data.get("c_clusters", [])),
            })
        return results

    def get_manifest(self, conversation_id: str) -> dict | None:
        import json
        row = self.conn.execute(
            "SELECT data FROM manifests WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def update_manifest(self, conversation_id: str, manifest: dict) -> None:
        import json
        self.conn.execute(
            "INSERT OR REPLACE INTO manifests VALUES (?, ?)",
            (conversation_id, json.dumps(manifest))
        )
        self.conn.commit()

    def get_latest_version(self, conversation_id: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(version) FROM archives WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row[0] if row[0] is not None else 0
```

Then pass it to the pipeline:

```python
pipeline = DreamingPipeline(
    llm_interface=llm,
    storage=SQLiteBackend("/data/memories.db")
)
```

The same pattern works for PostgreSQL, S3, Redis, or any other store.

## StorageBackend Interface Reference

```python
class StorageBackend(ABC):

    def save_archive(self, conversation_id: str, version: int, data: dict) -> None:
        """Save a versioned archive snapshot."""

    def load_archive(self, conversation_id: str, version: int | None = None) -> dict | None:
        """Load an archive. version=None returns the latest."""

    def list_archives(self) -> list[dict]:
        """List all conversations with summary metadata."""

    def get_manifest(self, conversation_id: str) -> dict | None:
        """Get the version manifest (lifecycle tracking)."""

    def update_manifest(self, conversation_id: str, manifest: dict) -> None:
        """Save/update the version manifest."""

    def get_latest_version(self, conversation_id: str) -> int:
        """Return the highest version number (0 if none)."""
```

## Pipeline API

### Process a conversation

```python
result = await pipeline.process_conversation(
    conversation_id="chat_001",
    conversation_text="full conversation text",
    metadata={"original_text": "full conversation text", "topic": "caching"}
)
# result["status"] == "success"
# result["stages"]["B_chunks"]["count"] -- number of semantic chunks
# result["stages"]["C_clusters"]["count"] -- number of topic clusters
# result["stages"]["D_archive"]["version"] -- archive version created
# result["stages"]["D_archive"]["path"] -- file path (JsonFileBackend)
```

Always include `original_text` in metadata so quality upgrades can re-process from source.

### Re-process (creates a new version)

```python
# Processing the same conversation_id again creates v2, v3, etc.
result_v2 = await pipeline.process_conversation(
    conversation_id="chat_001",
    conversation_text="updated conversation text",
    metadata={"original_text": "updated conversation text"}
)
# result_v2["stages"]["D_archive"]["version"] == 2
```

### Upgrade quality

```python
result = await pipeline.upgrade_quality("chat_001", target_quality="good")
# Re-processes the original text with a better LLM tier
# result["upgraded_from"] == "basic"
# result["upgraded_to"] == "good"
# Creates the next version; previous version becomes superseded
```

### Retrieve archives

```python
# Latest version
archive = pipeline.get_archive("chat_001")

# Specific version
archive_v1 = pipeline.get_archive("chat_001", version=1)

# List all conversations
archives = pipeline.list_archives()
# [{"conversation_id": "chat_001", "latest_version": 2, "quality_level": "good", ...}]

# Version manifest (lifecycle tracking)
manifest = pipeline.get_manifest("chat_001")
# manifest["versions"]["1"]["status"] == "superseded"
# manifest["versions"]["2"]["status"] == "active"
```

### Archive lifecycle

```python
lifecycle = pipeline.get_archive_lifecycle("chat_001")
# {"version": 2, "status": "active", "storage_location": "hot", "is_latest": True}
```

## Data Model

### BChunk (Semantic Chunk)

Each conversation is broken into semantic pieces:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique chunk ID |
| `parent_id` | str | Source conversation ID |
| `chunk_type` | ChunkType | `SEMANTIC`, `SPEAKER_TURN`, `ENTITY`, `RELATIONSHIP` |
| `content` | str | Original text (preserved language) |
| `labels` | list[str] | Topic tags (`["caching", "architecture"]`) |
| `speaker` | str | `user`, `assistant`, `system` |
| `entities` | list[str] | Named entities mentioned |
| `language` | str | Detected language code (`en`, `zh`, `ja`) |
| `quality_level` | str | `basic`, `good`, `premium` |

### CCluster (Synthesized Cluster)

Related chunks are grouped into clusters:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique cluster ID |
| `cluster_type` | ClusterType | `TOPIC`, `RELATIONSHIP`, `SUMMARY`, `TIMELINE` |
| `content` | str | Synthesized summary |
| `related_chunks` | list[str] | B chunk IDs in this cluster |
| `theme` | str | Cluster theme |
| `entities` | list[str] | Aggregated entities |

### Archive JSON structure

Each `archive_v{N}.json` contains:

```json
{
  "conversation_id": "chat_001",
  "version": 2,
  "quality_level": "good",
  "created_at": "2026-03-04T10:30:00",
  "metadata": {
    "is_latest": true,
    "status": "active",
    "storage_location": "hot",
    "previous_version": 1,
    "original_text": "..."
  },
  "b_chunks": [ ... ],
  "c_clusters": [ ... ],
  "entities": ["caching", "Redis", "API"]
}
```

## Quality Tiers

| Tier | LLM | Cost | Accuracy | Use Case |
|------|-----|------|----------|----------|
| `basic` | Local / small model | Free | 70-80% | Background nightly processing |
| `good` | Haiku / GPT-4o-mini | ~$0.001/conv | 85-95% | Standard quality |
| `premium` | Sonnet / GPT-4o | ~$0.01/conv | 95-99% | Critical conversations |

Quality can be upgraded progressively: `basic` -> `good` -> `premium`. Each upgrade creates a new archive version.

## Multi-language Support

The pipeline preserves the original language of each chunk. Mixed-language conversations are handled naturally -- each chunk retains its detected language:

```python
result = await pipeline.process_conversation(
    conversation_id="bilingual_chat",
    conversation_text="User: Redis 的缓存策略怎么配置？\nAssistant: You can configure Redis caching with...",
    metadata={"original_text": "..."}
)
# Chunks will have language="zh" and language="en" respectively
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

9 tests cover:

- JSON parsing resilience (embedded JSON, LLM repair, fail-fast)
- Pipeline versioning (v1/v2/v3, manifest lifecycle)
- Quality upgrade (creates new version, supersedes previous)
- Archive listing with metadata

## Contributing

### Adding a new storage backend

1. Create a file in `src/dreaming/storage/` (e.g., `sqlite_backend.py`)
2. Subclass `StorageBackend` and implement all 6 methods
3. Add tests that mirror the versioning tests in `tests/test_pipeline_versioning.py` but use your backend
4. Submit a PR

### Improving chunking or synthesis

The LLM prompts live in:
- `src/dreaming/chunker.py` -- A->B prompt (semantic chunking)
- `src/dreaming/synthesizer.py` -- B->C prompt (topic synthesis)

Both use a multi-pass JSON parsing strategy:
1. Try parsing the full LLM response as JSON
2. Extract the first JSON block from surrounding prose
3. Ask the LLM to repair malformed JSON
4. Fail fast with a clear error if all attempts fail

### Adding a new LLM provider

1. Create a class with a `generate_response(query: str, context=None) -> str` method
2. Optionally subclass `BaseLLMInterface` from `dreaming.llm.base`
3. Pass it to `DreamingPipeline(llm_interface=your_llm)`

No registration or adapter needed -- duck typing handles it.

### Project structure

```
src/dreaming/
  __init__.py          # Public API exports
  models.py            # BChunk, CCluster, DArchive dataclasses
  pipeline.py          # DreamingPipeline orchestrator
  chunker.py           # A->B semantic chunking
  synthesizer.py       # B->C topic synthesis
  setup.py             # Local model download CLI
  llm/
    base.py            # BaseLLMInterface ABC
    interface.py       # LLMInterface unified wrapper
    local.py           # LocalLLMInterface (llama.cpp)
    api.py             # APILLMInterface (OpenAI-compatible)
  storage/
    base.py            # StorageBackend ABC
    json_backend.py    # JsonFileBackend (default)
tests/
  test_parsing.py
  test_pipeline_versioning.py
docs/
  SPECIFICATION.md     # Full data model and pipeline spec
  LLM_DESIGN.md        # LLM integration and quality tiers
  IMPLEMENTATION_PLAN.md
```

## License

MIT

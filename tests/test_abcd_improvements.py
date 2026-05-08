"""Unit tests for ABCD Dreaming Pipeline improvements.

Tests the three fixes:
1. Entity extraction wired into auto-dream
2. Multi-cluster decomposition with actionable types
3. Knowledge unit extraction with threshold checks
"""

import unittest
from datetime import datetime

from dreaming.models import BChunk, CCluster, ClusterType, ChunkType
from dreaming.chunker import ConversationChunker
from dreaming.synthesizer import DreamingSynthesizer


class _FakeActiveInterface:
    def __init__(self, model: str = "fake-model"):
        self.model = model


class _FakeLLM:
    """Returns pre-programmed JSON responses for testing."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.active_interface_name = "fake-provider"
        self.active_interface = _FakeActiveInterface("fake-model")

    def generate_response(self, query=None, context=None):
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


class TestEntityExtraction(unittest.IsolatedAsyncioTestCase):
    """Test that entity extraction is properly wired into chunking."""

    async def test_chunker_extracts_entities_from_llm_response(self):
        """LLM response with entities should be preserved in BChunk."""
        llm = _FakeLLM(
            [
                '{"chunks": [{"content": "User asked about Docker", "language": "en", "labels": ["architecture"], "speaker": "user", "entities": ["Docker", "MoJoAssistant"], "summary": "Docker question", "key_facts": ["Docker path was chosen"]}]}',
            ]
        )
        chunker = ConversationChunker(llm_interface=llm)
        chunks = await chunker.chunk_conversation("conv_1", "User: ask about Docker")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].entities, ["Docker", "MoJoAssistant"])
        self.assertEqual(chunks[0].key_facts, ["Docker path was chosen"])

    async def test_chunker_extracts_entities_from_prose_json(self):
        """Entities should survive markdown-wrapped JSON."""
        llm = _FakeLLM(
            [
                '```json\n{"chunks": [{"content": "Rowhammer GPU", "language": "en", "labels": ["security"], "speaker": "assistant", "entities": ["Rowhammer", "GPU", "memory"], "summary": "Rowhammer attack", "key_facts": ["Rowhammer gives complete machine control"]}]}\n```',
            ]
        )
        chunker = ConversationChunker(llm_interface=llm)
        chunks = await chunker.chunk_conversation("conv_2", "Assistant: rowhammer info")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].entities, ["Rowhammer", "GPU", "memory"])


class TestMultiClusterSynthesis(unittest.IsolatedAsyncioTestCase):
    """Test multi-cluster decomposition with actionable types."""

    def test_cluster_type_enum_has_actionable_values(self):
        """ClusterType should include DECISION, OUTCOME, FINDING, TOOL, INCOMPLETE."""
        self.assertTrue(hasattr(ClusterType, "DECISION"))
        self.assertTrue(hasattr(ClusterType, "OUTCOME"))
        self.assertTrue(hasattr(ClusterType, "FINDING"))
        self.assertTrue(hasattr(ClusterType, "TOOL"))
        self.assertTrue(hasattr(ClusterType, "INCOMPLETE"))
        self.assertEqual(ClusterType.DECISION.value, "decision")
        self.assertEqual(ClusterType.OUTCOME.value, "outcome")
        self.assertEqual(ClusterType.FINDING.value, "finding")
        self.assertEqual(ClusterType.TOOL.value, "tool")
        self.assertEqual(ClusterType.INCOMPLETE.value, "incomplete")

    async def test_synthesizer_creates_actionable_clusters(self):
        """Synthesizer should create clusters of actionable types."""
        llm = _FakeLLM(
            [
                '{"clusters": [
                    {"type": "DECISION", "title": "Docker selection", "summary": "Chose Docker over bash_exec", "key_facts": ["Docker path selected"], "chunk_ids": ["b1"], "entities": ["Docker"], "insights": [], "related_clusters": [], "confidence": "high"},
                    {"type": "OUTCOME", "title": "Health endpoint", "summary": "Endpoint tested", "key_facts": ["/api/health returns status"], "chunk_ids": ["b2"], "entities": ["health endpoint"], "insights": [], "related_clusters": [], "confidence": "high"},
                    {"type": "FINDING", "title": "Rowhammer", "summary": "GPU attacks give control", "key_facts": ["Rowhammer GPU attacks give complete machine control"], "chunk_ids": ["b3"], "entities": ["Rowhammer", "GPU"], "insights": [], "related_clusters": [], "confidence": "medium"},
                    {"type": "TOOL", "title": "tmux usage", "summary": "Use tmux for terminal", "key_facts": ["tmux-backed terminal path preferred"], "chunk_ids": ["b4"], "entities": ["tmux"], "insights": [], "related_clusters": [], "confidence": "high"},
                    {"type": "INCOMPLETE", "title": "Pending", "summary": "Need to verify archival", "key_facts": [], "chunk_ids": ["b5"], "entities": [], "insights": [], "related_clusters": [], "confidence": "low"}
                ]}',
            ]
        )
        synthesizer = DreamingSynthesizer(llm_interface=llm)
        chunks = [
            BChunk(id="b1", parent_id="conv", chunk_type=ChunkType.SEMANTIC, content="x", labels=[], speaker="user", entities=["Docker"], confidence=0.9, created_at=datetime.now()),
            BChunk(id="b2", parent_id="conv", chunk_type=ChunkType.SEMANTIC, content="y", labels=[], speaker="assistant", entities=["health"], confidence=0.9, created_at=datetime.now()),
            BChunk(id="b3", parent_id="conv", chunk_type=ChunkType.SEMANTIC, content="z", labels=[], speaker="user", entities=["Rowhammer", "GPU"], confidence=0.9, created_at=datetime.now()),
            BChunk(id="b4", parent_id="conv", chunk_type=ChunkType.SEMANTIC, content="a", labels=[], speaker="assistant", entities=["tmux"], confidence=0.9, created_at=datetime.now()),
            BChunk(id="b5", parent_id="conv", chunk_type=ChunkType.SEMANTIC, content="b", labels=[], speaker="user", entities=[], confidence=0.9, created_at=datetime.now()),
        ]
        clusters = await synthesizer.synthesize_chunks(chunks=chunks, session_id="conv")

        self.assertGreaterEqual(len(clusters), 5)
        types = {c.cluster_type.value for c in clusters}
        self.assertIn("decision", types)
        self.assertIn("outcome", types)
        self.assertIn("finding", types)
        self.assertIn("tool", types)
        self.assertIn("incomplete", types)

    async def test_synthesizer_falls_back_to_topic_when_unknown_type(self):
        """Unknown cluster type should fall back to TOPIC."""
        llm = _FakeLLM(
            [
                '{"clusters": [{"type": "UNKNOWN_TYPE", "title": "T", "summary": "S", "chunk_ids": [], "entities": [], "insights": [], "related_clusters": [], "confidence": "high"}]}',
            ]
        )
        synthesizer = DreamingSynthesizer(llm_interface=llm)
        chunks = [BChunk(id="b1", parent_id="conv", chunk_type=ChunkType.SEMANTIC, content="x", labels=[], speaker="user", entities=[], confidence=0.9, created_at=datetime.now())]
        clusters = await synthesizer.synthesize_chunks(chunks=chunks, session_id="conv")

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].cluster_type, ClusterType.TOPIC)


class TestBChunkKeyFacts(unittest.TestCase):
    """Test that BChunk has key_facts as a proper dataclass field."""

    def test_bchunk_has_key_facts_field(self):
        """BChunk should have key_facts as a field with default_factory=list."""
        chunk = BChunk(id="b1", parent_id="conv", chunk_type=ChunkType.SEMANTIC, content="x")
        self.assertTrue(hasattr(chunk, "key_facts"))
        self.assertEqual(chunk.key_facts, [])

    def test_bchunk_key_facts_from_dict(self):
        """BChunk from_dict should preserve key_facts."""
        data = {
            "id": "b1", "parent_id": "conv", "chunk_type": "semantic", "content": "x",
            "key_facts": ["fact1", "fact2"], "labels": [], "speaker": "user",
            "entities": [], "confidence": 0.9, "created_at": datetime.now().isoformat(),
            "token_range": [0, 0], "position_in_parent": 0.0,
            "quality_level": "basic", "needs_upgrade": True, "llm_used": None,
            "language": "en", "embedding": None,
        }
        chunk = BChunk.from_dict(data)
        self.assertEqual(chunk.key_facts, ["fact1", "fact2"])


class TestCClusterConfidenceLevel(unittest.TestCase):
    """Test that CCluster has confidence_level as a proper field."""

    def test_ccluster_has_confidence_level_field(self):
        """CCluster should have confidence_level as a field."""
        cluster = CCluster(id="c1", cluster_type=ClusterType.DECISION, content="x")
        self.assertTrue(hasattr(cluster, "confidence_level"))
        self.assertEqual(cluster.confidence_level, None)

    def test_ccluster_to_dict_preserves_confidence_level(self):
        """CCluster to_dict should include confidence_level."""
        cluster = CCluster(id="c1", cluster_type=ClusterType.OUTCOME, content="x", confidence_level="high")
        data = cluster.to_dict()
        self.assertIn("confidence_level", data)
        self.assertEqual(data["confidence_level"], "high")


if __name__ == "__main__":
    unittest.main()

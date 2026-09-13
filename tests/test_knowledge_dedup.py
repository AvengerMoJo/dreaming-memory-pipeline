"""Tests for the knowledge-base duplication fix (2026-07-23).

Root cause found live: knowledge_multi_model.json had grown to 7036
documents, 71% (5008) of which were exact-duplicate report-boilerplate
fragments ("**Key Findings:**\n1.", "**Incomplete:** None.", etc.) --
produced by AtomicFactExtractor._fallback_extraction's naive
one-unit-per-paragraph split with no minimum-content filter, and never
caught because MultiModelEmbeddingStorage.store_document had no
deduplication at all.
"""
import tempfile
import unittest

from mojo_memory.memory.multi_model_storage import MultiModelEmbeddingStorage


class _FakeEmbeddingService:
    model_version = "fake-v1"

    def get_text_embedding(self, text):
        return [0.1, 0.2, 0.3]


class TestStoreDocumentDedup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = MultiModelEmbeddingStorage(data_dir=self.tmp.name)
        self.models = {"fake:3": _FakeEmbeddingService()}

    def test_first_document_is_stored(self):
        doc_id = self.storage.store_document("A real fact about the system.", {}, self.models)
        self.assertIsNotNone(doc_id)
        self.assertEqual(len(self.storage.documents), 1)

    def test_exact_duplicate_is_skipped(self):
        self.storage.store_document("A real fact about the system.", {}, self.models)
        second = self.storage.store_document("A real fact about the system.", {}, self.models)
        self.assertIsNone(second)
        self.assertEqual(len(self.storage.documents), 1)

    def test_duplicate_with_different_whitespace_is_still_caught(self):
        self.storage.store_document("A real fact about the system.", {}, self.models)
        second = self.storage.store_document("  A real fact about the system.  \n", {}, self.models)
        self.assertIsNone(second)
        self.assertEqual(len(self.storage.documents), 1)

    def test_empty_content_is_skipped(self):
        result = self.storage.store_document("   \n  ", {}, self.models)
        self.assertIsNone(result)
        self.assertEqual(len(self.storage.documents), 0)

    def test_distinct_content_is_not_deduplicated(self):
        self.storage.store_document("Fact one.", {}, self.models)
        second = self.storage.store_document("Fact two, genuinely different.", {}, self.models)
        self.assertIsNotNone(second)
        self.assertEqual(len(self.storage.documents), 2)

    def test_dedup_index_survives_reload(self):
        self.storage.store_document("Persisted fact.", {}, self.models)
        reloaded = MultiModelEmbeddingStorage(data_dir=self.tmp.name)
        result = reloaded.store_document("Persisted fact.", {}, self.models)
        self.assertIsNone(result)
        self.assertEqual(len(reloaded.documents), 1)


if __name__ == "__main__":
    unittest.main()

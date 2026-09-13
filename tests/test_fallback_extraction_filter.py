"""Tests for AtomicFactExtractor._fallback_extraction's minimum-content
filter (2026-07-23) -- see test_knowledge_dedup.py for the storage-layer
half of this fix.
"""
import unittest

from dreaming.atomic_extractor import AtomicFactExtractor


class TestFallbackExtractionFilter(unittest.TestCase):
    def setUp(self):
        self.extractor = AtomicFactExtractor(llm_interface=None)

    def test_short_boilerplate_paragraphs_are_skipped(self):
        text = (
            "**Key Findings:**\n\n"
            "1.\n\n"
            "**Incomplete:** None.\n\n"
            "This is a genuinely substantive finding that spans well over thirty characters."
        )
        units = self.extractor._fallback_extraction("doc1", text)
        self.assertEqual(len(units), 1)
        self.assertIn("genuinely substantive", units[0].quote)

    def test_all_short_paragraphs_yields_no_units(self):
        text = "**Key Findings:**\n\n1.\n\n**Incomplete:** None."
        units = self.extractor._fallback_extraction("doc1", text)
        self.assertEqual(units, [])

    def test_substantive_paragraphs_all_kept(self):
        text = (
            "This first paragraph is definitely long enough to be a real fact.\n\n"
            "This second paragraph is also long enough to stand on its own as a fact."
        )
        units = self.extractor._fallback_extraction("doc1", text)
        self.assertEqual(len(units), 2)

    def test_boundary_length_paragraph_kept(self):
        # Exactly at the threshold should be kept (filter is strictly "<").
        para = "x" * AtomicFactExtractor._MIN_FALLBACK_PARAGRAPH_CHARS
        units = self.extractor._fallback_extraction("doc1", para)
        self.assertEqual(len(units), 1)


if __name__ == "__main__":
    unittest.main()

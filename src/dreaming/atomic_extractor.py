"""
Atomic Fact Extractor — Document → Knowledge Units

Transforms a document (research report, article, design doc, task output)
into a list of KnowledgeUnits: atomic propositions each anchored to a
direct quote from the source text.

Unlike the ConversationChunker which segments text structurally, this
extractor distills meaning — each unit stands alone, represents exactly
one idea, and is independently embeddable. Links between related units
are computed post-extraction via entity and label overlap, forming a
lightweight knowledge chain across the document.

Use this for document-type inputs. Use ConversationChunker for conversations.
"""

from __future__ import annotations

import json

from dreaming.json_repair import only_objects, repair_json_escapes
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from dreaming.models import KnowledgeUnit


EXTRACTION_PROMPT = """You are a knowledge extraction expert. Extract atomic facts from the following document.

An atomic fact is:
- A single self-contained proposition (one idea only)
- Independently meaningful without surrounding context
- Supported by a direct quote from the source text

DOCUMENT:
{document_text}

For each atomic fact provide:
- core_meaning: One precise sentence stating the fact
- quote: The exact phrase or sentence from the document that supports this fact (copy verbatim)
- labels: Topic tags (e.g. ["security", "supply-chain", "architecture"])
- entities: Named entities mentioned (tools, people, concepts, organizations)

Extract enough facts to fully cover the document's key ideas.
Avoid redundancy — each unit should represent a distinct proposition.

OUTPUT FORMAT (JSON):
{{
  "knowledge_units": [
    {{
      "core_meaning": "<single atomic proposition>",
      "quote": "<exact verbatim quote from source>",
      "labels": ["<tag1>", "<tag2>"],
      "entities": ["<entity1>", "<entity2>"]
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""


class AtomicFactExtractor:
    """Extracts atomic knowledge units from documents using LLM."""

    def __init__(self, llm_interface, quality_level: str = "basic", logger=None):
        self.llm = llm_interface
        self.quality_level = quality_level
        self.logger = logger

    def _log(self, message: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(f"[AtomicExtractor] {message}")

    async def extract_units(
        self,
        doc_id: str,
        document_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeUnit]:
        """
        Extract atomic knowledge units from a document.

        Returns KnowledgeUnits with inter-unit links computed via
        entity/label overlap.
        """
        self._log(f"Extracting knowledge units from {doc_id} ({len(document_text)} chars)")

        try:
            prompt = EXTRACTION_PROMPT.format(document_text=document_text)
            response = self.llm.generate_response(query=prompt, context=None)
            units_data = self._parse_llm_response(response)
            units = self._create_units(doc_id, units_data)
        except Exception as e:
            self._log(f"LLM extraction failed, using fallback: {e}", "warning")
            units = self._fallback_extraction(doc_id, document_text)

        units = self._compute_links(units)
        self._log(f"Extracted {len(units)} knowledge units with {sum(len(u.links) for u in units)} total links")
        return units

    def _compute_links(self, units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """
        Link units that share entities or 2+ labels.

        A single shared entity is a strong signal (named things are specific).
        Two shared labels are needed since labels are broader topic tags.
        """
        for i, unit in enumerate(units):
            unit_entities = set(unit.entities)
            unit_labels = set(unit.labels)
            for j, other in enumerate(units):
                if i == j:
                    continue
                shared_entities = unit_entities & set(other.entities)
                shared_labels = unit_labels & set(other.labels)
                if shared_entities or len(shared_labels) >= 2:
                    if other.id not in unit.links:
                        unit.links.append(other.id)
        return units

    def _create_units(self, doc_id: str, units_data: Dict[str, Any]) -> List[KnowledgeUnit]:
        raw = units_data.get("knowledge_units", [])
        total = len(raw)
        units = []
        for i, item in enumerate(raw):
            unit = KnowledgeUnit(
                id=f"ku_{doc_id}_{i}",
                source_doc_id=doc_id,
                content=item.get("core_meaning", item.get("content", "")),
                quote=item.get("quote", ""),
                labels=item.get("labels", []),
                entities=item.get("entities", []),
                links=[],
                confidence=0.9 if self.quality_level == "good" else 0.7,
                created_at=datetime.now(),
            )
            units.append(unit)
        return units

    # Below this length, a paragraph is essentially never a standalone fact
    # -- it's a markdown header ("**Key Findings:**"), a bare list marker
    # ("1."), or similar structural scaffolding. Found live 2026-07-23: with
    # no filter here, 71% of the entire knowledge base (5008 of 7036
    # documents) ended up being exact-duplicate boilerplate fragments from
    # this fallback path, because task reports share the same section
    # headers and this function turned every one into its own "fact."
    _MIN_FALLBACK_PARAGRAPH_CHARS = 30

    def _fallback_extraction(self, doc_id: str, text: str) -> List[KnowledgeUnit]:
        """Rule-based fallback: one unit per paragraph, skipping paragraphs
        too short to carry standalone meaning. Storage-layer dedup
        (MultiModelEmbeddingStorage.store_document) is the other half of
        this fix -- it catches exact-duplicate content regardless of
        source; this filter reduces how much low-value content is
        generated in the first place.
        """
        self._log("Using rule-based fallback extraction", "warning")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        skipped = 0
        units = []
        for i, para in enumerate(paragraphs):
            if len(para) < self._MIN_FALLBACK_PARAGRAPH_CHARS:
                skipped += 1
                continue
            first_sentence = para.split(".")[0].strip() + "." if "." in para else para[:120]
            unit = KnowledgeUnit(
                id=f"ku_{doc_id}_{i}_fallback",
                source_doc_id=doc_id,
                content=first_sentence,
                quote=para[:400],
                labels=[],
                entities=[],
                links=[],
                confidence=0.4,
                created_at=datetime.now(),
            )
            units.append(unit)
        if skipped:
            self._log(f"Fallback extraction skipped {skipped} paragraph(s) under {self._MIN_FALLBACK_PARAGRAPH_CHARS} chars")
        return units

    # ------------------------------------------------------------------ #
    # JSON parsing — same multi-pass pattern as chunker / synthesizer     #
    # ------------------------------------------------------------------ #

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        clean = response.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        clean = repair_json_escapes(clean)

        try:
            parsed = json.loads(clean)
            normalized = self._normalize(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            pass

        extracted = self._extract_first_json(clean)
        if extracted is not None:
            return extracted

        decoded = self._raw_decode(clean)
        if decoded is not None:
            return decoded

        repaired = self._repair_with_llm(clean)
        if repaired is not None:
            return repaired

        raise ValueError("Failed to parse extraction response as JSON after all passes")

    def _normalize(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            if only_objects(payload.get("knowledge_units")):
                return payload
            if only_objects(payload.get("units")):
                return {"knowledge_units": payload["units"]}
            if only_objects(payload.get("facts")):
                return {"knowledge_units": payload["facts"]}
        if only_objects(payload):
            return {"knowledge_units": payload}
        return None

    def _extract_first_json(self, text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start: i + 1])
                            normalized = self._normalize(parsed)
                            if normalized is not None:
                                return normalized
                        except Exception:
                            break
            start = text.find("{", start + 1)
        return None

    def _raw_decode(self, text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[i:])
                normalized = self._normalize(parsed)
                if normalized is not None:
                    return normalized
            except Exception:
                continue
        return None

    def _repair_with_llm(self, raw_text: str) -> Optional[Dict[str, Any]]:
        repair_prompt = (
            "Convert the following content into STRICT valid JSON with this schema only:\n"
            '{"knowledge_units":[{"core_meaning":"<string>","quote":"<string>",'
            '"labels":["<string>"],"entities":["<string>"]}]}\n'
            "Return JSON only. No prose, no markdown.\n\n"
            f"CONTENT:\n{raw_text}"
        )
        try:
            response = self.llm.generate_response(query=repair_prompt, context=None)
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            parsed = json.loads(clean.strip())
            return self._normalize(parsed)
        except Exception as e:
            self._log(f"LLM repair failed: {e}", "error")
        return None

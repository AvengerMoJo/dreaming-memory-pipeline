"""
Conversation Chunker - A→B Conversion

Transforms raw conversations (A) into semantic chunks (B) using LLM.
Extracts entities, key_facts, labels, and speaker metadata per chunk.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from dreaming.models import BChunk, ChunkType


# Universal chunking prompt (works across languages)
CHUNKING_PROMPT = """You are a semantic analysis expert. Analyze the following conversation and break it into meaningful semantic chunks.

CONVERSATION:
{conversation_text}

INSTRUCTIONS:
1. Identify natural semantic boundaries (topic shifts, speaker turns, logical breaks)
2. Each chunk should be 100-800 tokens
3. Extract metadata for each chunk:
   - labels: List of topic tags (e.g., ["technical", "architecture", "billing"])
   - speaker: Who is speaking (user/assistant/system)
   - entities: Named entities mentioned (people, products, concepts, tools, organizations) — extract ALL named entities
   - summary: One-sentence summary of the chunk
   - key_facts: List of specific factual statements found in this chunk — one per entry (e.g., "Alice moved to Paris in 2020", "Bob is married with two children")

IMPORTANT:
- Preserve the ORIGINAL language of each chunk (do not translate)
- Multi-lingual conversations: Keep each language as-is
- Detect language per chunk: "zh", "en", "ja", etc.
- Extract as many entities as possible — do not omit named concepts
- Extract key_facts even if the chunk is short — every factual statement counts

OUTPUT FORMAT (JSON):
{{
  "chunks": [
    {{
      "content": "<original text, unchanged>",
      "language": "<detected language code>",
      "labels": ["<tag1>", "<tag2>"],
      "speaker": "<user|assistant|system>",
      "entities": ["<entity1>", "<entity2>"],
      "summary": "<one-sentence summary>",
      "key_facts": ["<specific factual statement>", "<another fact>"]
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""


class ConversationChunker:
    """Chunks conversations into semantic pieces using LLM"""

    def __init__(
        self,
        llm_interface,
        quality_level: str = "basic",
        logger=None
    ):
        """
        Initialize chunker

        Args:
            llm_interface: LLM interface instance (any object with generate_response())
            quality_level: Target quality (basic/good/premium)
            logger: Optional logger instance
        """
        self.llm = llm_interface
        self.quality_level = quality_level
        self.logger = logger

    def _log(self, message: str, level: str = "info"):
        """Log message if logger available"""
        if self.logger:
            getattr(self.logger, level)(f"[Chunker] {message}")

    async def chunk_conversation(
        self,
        conversation_id: str,
        conversation_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[BChunk]:
        """
        Chunk a single conversation into B chunks using LLM

        Args:
            conversation_id: A chunk ID (parent)
            conversation_text: Full conversation content
            metadata: Optional metadata from A chunk

        Returns:
            List of B chunks
        """
        self._log(f"Chunking conversation {conversation_id} ({len(conversation_text)} chars)")

        prompt = CHUNKING_PROMPT.format(conversation_text=conversation_text)
        response = self.llm.generate_response(query=prompt, context=None)
        chunks_data = self._parse_llm_response(response)

        b_chunks = self._create_b_chunks(
            parent_id=conversation_id,
            chunks_data=chunks_data,
            original_text=conversation_text
        )

        self._log(f"Created {len(b_chunks)} B chunks")
        return b_chunks

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response with multi-pass fallbacks"""
        # First pass: clean obvious markdown wrappers.
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        # Attempt strict parse first.
        try:
            parsed = json.loads(response_clean)
            normalized = self._normalize_chunk_payload(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            pass

        # Second pass: extract the first JSON object from mixed prose output.
        extracted = self._extract_first_json_object(response_clean)
        if extracted is not None:
            return extracted

        # Third pass: use JSONDecoder raw_decode over all candidate positions.
        decoded = self._extract_json_with_raw_decode(response_clean)
        if decoded is not None:
            return decoded

        # Fourth pass: ask LLM to repair output into strict JSON.
        repaired = self._repair_json_with_llm(response_clean)
        if repaired is not None:
            return repaired

        raise ValueError("Failed to parse chunking response as JSON object after repair")

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract first valid JSON object from free-form model output."""
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
                        candidate = text[start : i + 1]
                        try:
                            parsed = json.loads(candidate)
                            normalized = self._normalize_chunk_payload(parsed)
                            if normalized is not None:
                                return normalized
                        except Exception:
                            break

            start = text.find("{", start + 1)

        return None

    def _extract_json_with_raw_decode(self, text: str) -> Optional[Dict[str, Any]]:
        """Try json raw_decode at each JSON-like start char and normalize payload."""
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, _end = decoder.raw_decode(text[i:])
                normalized = self._normalize_chunk_payload(parsed)
                if normalized is not None:
                    return normalized
            except Exception:
                continue
        return None

    def _normalize_chunk_payload(self, payload: Any) -> Optional[Dict[str, Any]]:
        """Normalize common model payload shapes into {"chunks":[...]}."""
        if isinstance(payload, dict):
            if isinstance(payload.get("chunks"), list):
                return payload
            data = payload.get("data")
            if isinstance(data, dict) and isinstance(data.get("chunks"), list):
                return {"chunks": data.get("chunks", [])}
            results = payload.get("results")
            if isinstance(results, dict) and isinstance(results.get("chunks"), list):
                return {"chunks": results.get("chunks", [])}
            if isinstance(payload.get("output"), dict) and isinstance(payload.get("output").get("chunks"), list):
                return {"chunks": payload.get("output").get("chunks", [])}
            if isinstance(payload.get("items"), list):
                return {"chunks": payload.get("items", [])}
            if isinstance(payload.get("segments"), list):
                return {"chunks": payload.get("segments", [])}
            if isinstance(payload.get("sections"), list):
                return {"chunks": payload.get("sections", [])}
        elif isinstance(payload, list):
            return {"chunks": payload}
        return None

    def _repair_json_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Ask LLM to repair malformed JSON output."""
        self._log("Attempting JSON repair via LLM", "info")
        repair_prompt = f"""The following text contains JSON data but is malformed. Extract the JSON data and return it as a clean JSON object with a 'chunks' key containing a list of chunk objects.

TEXT:
{text}

Return ONLY valid JSON:
{{
  "chunks": [
    {{
      "content": "...",
      "language": "...",
      "labels": [...],
      "speaker": "...",
      "entities": [...],
      "summary": "...",
      "key_facts": [...]
    }}
  ]
}}"""
        try:
            response = self.llm.generate_response(query=repair_prompt, context=None)
            parsed = json.loads(response.strip())
            normalized = self._normalize_chunk_payload(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            pass
        return None

    def _create_b_chunks(
        self,
        parent_id: str,
        chunks_data: Dict[str, Any],
        original_text: str
    ) -> List[BChunk]:
        """Create BChunk objects from LLM output"""
        b_chunks = []
        chunks = chunks_data.get("chunks", [])

        llm_info = self._get_llm_info()

        for i, chunk_data in enumerate(chunks):
            chunk_id = f"b_{parent_id}_{i}"

            token_start = i * 400
            token_end = token_start + len(chunk_data.get("content", "").split())

            # Extract entities from the LLM output
            entities = chunk_data.get("entities", [])
            # Also extract entities from key_facts if entities is empty
            if not entities:
                key_facts = chunk_data.get("key_facts", [])
                for fact in key_facts:
                    # Extract named entities from facts (capitalized words, etc.)
                    fact_entities = self._extract_entities_from_text(fact)
                    entities.extend(fact_entities)

            # Extract key_facts as proper field
            key_facts = chunk_data.get("key_facts", [])
            if not isinstance(key_facts, list):
                key_facts = []

            b_chunk = BChunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.SEMANTIC,
                content=chunk_data.get("content", ""),
                labels=chunk_data.get("labels", []),
                speaker=chunk_data.get("speaker", "unknown"),
                entities=entities,
                key_facts=key_facts,
                confidence=0.9 if self.quality_level == "good" else 0.7,
                token_range=(token_start, token_end),
                position_in_parent=i / len(chunks) if chunks else 0.0,
                embedding=None,
                created_at=datetime.now()
            )

            # Set quality tracking fields
            b_chunk.quality_level = self.quality_level
            b_chunk.needs_upgrade = (self.quality_level == "basic")
            b_chunk.llm_used = llm_info.get("model")
            b_chunk.language = chunk_data.get("language", "unknown")

            b_chunks.append(b_chunk)

        return b_chunks

    def _get_llm_info(self) -> Dict[str, Any]:
        """Get current LLM provider info"""
        try:
            provider = getattr(self.llm, "active_interface_name", "unknown")
            active = getattr(self.llm, "active_interface", None)
            model = getattr(active, "model", "unknown") if active else "unknown"
            return {"provider": provider, "model": model}
        except Exception:
            pass
        return {"provider": "unknown", "model": "unknown"}

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Simple entity extraction from text using capitalized words and patterns"""
        entities = []
        # Extract capitalized words (potential named entities)
        words = text.split()
        for word in words:
            # Clean punctuation
            cleaned = word.strip(".,;:!?()[]{}\"'")
            if cleaned and len(cleaned) >= 2 and cleaned[0].isupper():
                entities.append(cleaned)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique

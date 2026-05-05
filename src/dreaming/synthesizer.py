"""
Dreaming Synthesizer - B→C Conversion

Clusters semantic chunks (B) into synthesized knowledge (C) using LLM.
Creates topic clusters, relationship maps, timelines, and actionable clusters
(DECISION, OUTCOME, FINDING, TOOL, INCOMPLETE) for retrieval.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from dreaming.models import BChunk, CCluster, ClusterType


# Synthesis prompt for clustering B chunks into C clusters
SYNTHESIS_PROMPT = """You are a knowledge synthesis expert. Analyze the following semantic chunks and cluster them into meaningful, actionable knowledge.

CHUNKS:
{chunks_json}

INSTRUCTIONS:
1. Decompose the content into actionable clusters. Aim for at least 5 clusters.
   Use these cluster types:
   - DECISION: Explicit decisions made (e.g., "chose Docker path over bash_exec", "selected gemma-4-3b for inference")
   - OUTCOME: Concrete results (e.g., "briefing saved to ~/.memory/scout_briefing_2026-04-06.md", "health endpoint tested and verified")
   - FINDING: Discovered facts (e.g., "Rowhammer GPU attacks give complete machine control", "cella should be exercised through tmux-backed terminal path")
   - TOOL: Tool usage patterns and recommendations (e.g., "use playwright_browser_snapshot over screenshot for actionable interaction", "scheduler_add_task for async handoff")
   - TOPIC: Thematic groupings (e.g., "scheduler architecture", "error handling")
   - RELATIONSHIP: Connected concepts across chunks
   - TIMELINE: Temporal or sequential patterns
   - SUMMARY: High-level overview of the entire conversation
   - INCOMPLETE: Unresolved items that need follow-up (e.g., "need to verify D stage archival path", "pending investigation of entity extraction failure")

2. For each cluster, provide:
   - type: One of [DECISION, OUTCOME, FINDING, TOOL, TOPIC, RELATIONSHIP, TIMELINE, SUMMARY, INCOMPLETE]
   - title: Concise cluster name
   - summary: 1-2 sentence synthesis of the cluster content
   - key_facts: Complete list of ALL specific facts, events, states, and attributes found in the cluster's chunks — one atomic statement per entry. Do NOT omit facts.
   - chunk_ids: List of chunk IDs in this cluster
   - entities: Key entities/concepts mentioned
   - insights: Novel connections or patterns discovered
   - related_clusters: IDs of clusters that relate to this one
   - confidence: "high", "medium", or "low" — how confident you are in this cluster's accuracy

3. Cross-reference clusters when concepts relate
4. Ensure each cluster is actionable and retrievable — avoid generic meta-descriptions like "A user query about..."

OUTPUT FORMAT (JSON):
{
  "clusters": [
    {
      "type": "DECISION",
      "title": "<cluster name>",
      "summary": "<synthesis of cluster content>",
      "key_facts": ["<specific factual statement>", "<another fact>"],
      "chunk_ids": ["b_xxx_0", "b_xxx_2"],
      "entities": ["<entity1>", "<entity2>"],
      "insights": ["<insight1>", "<insight2>"],
      "related_clusters": [],
      "confidence": "high"
    }
  ]
}

Return ONLY valid JSON, no additional text."""


class DreamingSynthesizer:
    """Synthesizes B chunks into C clusters using LLM"""

    def __init__(
        self,
        llm_interface,
        quality_level: str = "basic",
        logger=None
    ):
        self.llm = llm_interface
        self.quality_level = quality_level
        self.logger = logger

    def _log(self, message: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(f"[Synthesizer] {message}")

    async def synthesize_chunks(
        self,
        chunks: List[BChunk],
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[CCluster]:
        self._log(f"Synthesizing {len(chunks)} chunks into clusters")

        if not chunks:
            self._log("No chunks to synthesize", "warning")
            return []

        try:
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    "id": chunk.id,
                    "content": chunk.content[:200],
                    "labels": chunk.labels,
                    "speaker": chunk.speaker,
                    "entities": chunk.entities,
                    "key_facts": getattr(chunk, 'key_facts', chunk.__dict__.get('key_facts', []))
                })

            chunks_json = json.dumps(chunks_data, indent=2, ensure_ascii=False)
            prompt = SYNTHESIS_PROMPT.format(chunks_json=chunks_json)

            response = self.llm.generate_response(query=prompt, context=None)
            clusters_data = self._parse_llm_response(response)

            c_clusters = self._create_c_clusters(
                session_id=session_id,
                clusters_data=clusters_data,
                source_chunks=chunks
            )

            self._log(f"Created {len(c_clusters)} C clusters")
            return c_clusters

        except Exception as e:
            llm_info = self._get_llm_info()
            self._log(
                f"LLM synthesis failed (provider={llm_info.get('provider')} model={llm_info.get('model')}), "
                f"using rule-based fallback. error={e}",
                "warning"
            )
            return self._fallback_clustering(chunks, session_id)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        try:
            parsed = json.loads(response_clean)
            normalized = self._normalize_cluster_payload(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            pass

        extracted = self._extract_first_json_object(response_clean)
        if extracted is not None:
            return extracted

        decoded = self._extract_json_with_raw_decode(response_clean)
        if decoded is not None:
            return decoded

        repaired = self._repair_json_with_llm(response_clean)
        if repaired is not None:
            return repaired

        raise ValueError("Failed to parse synthesis response as JSON object after repair")

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
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
                            normalized = self._normalize_cluster_payload(parsed)
                            if normalized is not None:
                                return normalized
                        except Exception:
                            break

            start = text.find("{", start + 1)

        return None

    def _extract_json_with_raw_decode(self, text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, _end = decoder.raw_decode(text[i:])
                normalized = self._normalize_cluster_payload(parsed)
                if normalized is not None:
                    return normalized
            except Exception:
                continue
        return None

    def _normalize_cluster_payload(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            if isinstance(payload.get("clusters"), list):
                return payload
            data = payload.get("data")
            if isinstance(data, dict) and isinstance(data.get("clusters"), list):
                return {"clusters": data.get("clusters", [])}
            results = payload.get("results")
            if isinstance(results, dict) and isinstance(results.get("clusters"), list):
                return {"clusters": results.get("clusters", [])}
            if isinstance(payload.get("items"), list):
                return {"clusters": payload.get("items", [])}
            if isinstance(payload.get("output"), dict) and isinstance(payload.get("output").get("clusters"), list):
                return {"clusters": payload.get("output").get("clusters", [])}
        elif isinstance(payload, list):
            return {"clusters": payload}
        return None

    def _repair_json_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        self._log("Attempting JSON repair via LLM", "info")
        repair_prompt = f"""The following text contains JSON data but is malformed. Extract the JSON data and return it as a clean JSON object with a 'clusters' key containing a list of cluster objects.

TEXT:
{text}

Return ONLY valid JSON:
{{
  "clusters": [...]
}}"""
        try:
            response = self.llm.generate_response(query=repair_prompt, context=None)
            parsed = json.loads(response.strip())
            normalized = self._normalize_cluster_payload(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            pass
        return None

    def _create_c_clusters(
        self,
        session_id: str,
        clusters_data: Dict[str, Any],
        source_chunks: List[BChunk]
    ) -> List[CCluster]:
        c_clusters = []
        clusters = clusters_data.get("clusters", [])

        llm_info = self._get_llm_info()

        for i, cluster_data in enumerate(clusters):
            cluster_id = f"c_{session_id}_{i}"

            cluster_type_str = cluster_data.get("type", "TOPIC").upper()
            try:
                cluster_type = ClusterType[cluster_type_str]
            except KeyError:
                cluster_type = ClusterType.TOPIC

            c_cluster = CCluster(
                id=cluster_id,
                cluster_type=cluster_type,
                content=cluster_data.get("summary", ""),
                related_chunks=cluster_data.get("chunk_ids", []),
                related_clusters=cluster_data.get("related_clusters", []),
                theme=cluster_data.get("title", f"Cluster {i}"),
                confidence=0.9 if self.quality_level == "good" else 0.7,
                confidence_level=cluster_data.get("confidence", "medium"),
                created_at=datetime.now()
            )

            if hasattr(c_cluster, '__dict__'):
                c_cluster.__dict__['quality_level'] = self.quality_level
                c_cluster.__dict__['needs_upgrade'] = (self.quality_level == "basic")
                c_cluster.__dict__['llm_used'] = llm_info.get("model")
                raw_facts = cluster_data.get("key_facts", [])
                c_cluster.__dict__['key_facts'] = raw_facts if isinstance(raw_facts, list) else []

            c_clusters.append(c_cluster)

        return c_clusters

    def _get_llm_info(self) -> Dict[str, Any]:
        try:
            provider = getattr(self.llm, "active_interface_name", "unknown")
            active = getattr(self.llm, "active_interface", None)
            model = getattr(active, "model", "unknown") if active else "unknown"
            return {"provider": provider, "model": model}
        except Exception:
            pass
        return {"provider": "unknown", "model": "unknown"}

    def _fallback_clustering(
        self,
        chunks: List[BChunk],
        session_id: str
    ) -> List[CCluster]:
        """Improved rule-based fallback clustering"""
        self._log("Using rule-based fallback clustering", "warning")
        llm_info = self._get_llm_info()

        # Group by labels if available
        label_groups = defaultdict(list)
        for chunk in chunks:
            for label in chunk.labels:
                label_groups[label].append(chunk)

        # If no labels, group by speaker
        if not label_groups:
            speaker_groups = defaultdict(list)
            for chunk in chunks:
                speaker_groups[chunk.speaker].append(chunk)
            label_groups = speaker_groups

        # If still no groups, create one group per chunk (topic-boundary chunking)
        if not label_groups:
            label_groups = {f"segment_{i}": [chunk] for i, chunk in enumerate(chunks)}

        c_clusters = []
        for i, (label, grouped_chunks) in enumerate(label_groups.items()):
            cluster_id = f"c_{session_id}_{i}_fallback"

            # Extract entities from all chunks in this cluster
            all_entities = set()
            for chunk in grouped_chunks:
                all_entities.update(chunk.entities)

            # Extract key facts from chunks
            all_key_facts = []
            for chunk in grouped_chunks:
                key_facts = getattr(chunk, 'key_facts', chunk.__dict__.get('key_facts', []))
                if isinstance(key_facts, list):
                    all_key_facts.extend(key_facts)

            # Determine cluster type
            if i == 0 and len(label_groups) > 1:
                cluster_type = ClusterType.SUMMARY  # First cluster is overview
            elif len(grouped_chunks) == 1:
                cluster_type = ClusterType.FINDING  # Single chunk is a finding
            else:
                cluster_type = ClusterType.TOPIC

            c_cluster = CCluster(
                id=cluster_id,
                cluster_type=cluster_type,
                content=f"Cluster of {len(grouped_chunks)} chunks labeled '{label}'",
                related_chunks=[c.id for c in grouped_chunks],
                related_clusters=[],
                theme=label,
                confidence=0.5,
                confidence_level="low",
                created_at=datetime.now()
            )

            if hasattr(c_cluster, '__dict__'):
                c_cluster.__dict__['quality_level'] = self.quality_level
                c_cluster.__dict__['needs_upgrade'] = True
                c_cluster.__dict__['llm_used'] = llm_info.get("model")
                c_cluster.__dict__['key_facts'] = all_key_facts

            c_clusters.append(c_cluster)

        self._log(f"Created {len(c_clusters)} fallback C clusters")
        return c_clusters

"""
Base LLM Interface

Abstract base class for all LLM implementations.
"""

import logging
from typing import Dict, List, Any
from abc import ABC, abstractmethod


class BaseLLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate_response(self, query: str, context: List[Dict[str, Any]] | None = None) -> str:
        """Generate a response from the LLM"""
        pass

    def shutdown(self) -> None:
        """Clean up resources"""
        pass

    def format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context information for inclusion in prompts

        Args:
            context: List of context items

        Returns:
            str: Formatted context text
        """
        if not context:
            return "No previous context available."

        context_items = []
        for item in context:
            source = item.get('source', 'unknown')

            # Handle different types of content
            content = item.get('content', '')
            if not isinstance(content, str):
                try:
                    if hasattr(content, 'content'):
                        content = content.content
                    elif hasattr(content, 'text'):
                        content = content.text
                    else:
                        content = str(content)
                except Exception:
                    content = "Complex content object"

            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."

            context_items.append(f"- From {source}: {content}")

        return "\n".join(context_items)

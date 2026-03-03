"""
LLM Abstraction Layer

Provider-agnostic LLM interface for the dreaming pipeline.
Supports local models (llama.cpp), API models (OpenAI, Anthropic, etc.),
and custom implementations via the BaseLLMInterface ABC.
"""

from dreaming.llm.base import BaseLLMInterface
from dreaming.llm.interface import LLMInterface, create_llm_interface
from dreaming.llm.local import LocalLLMInterface
from dreaming.llm.api import APILLMInterface

__all__ = [
    'BaseLLMInterface',
    'LLMInterface',
    'create_llm_interface',
    'LocalLLMInterface',
    'APILLMInterface',
]

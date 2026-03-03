"""
Local LLM Interface

Interface for local LLM models using direct subprocess or local API calls.
Supports llama.cpp, GPT4All, and other local model servers.
"""

from typing import Dict, List, Any, Optional
from subprocess import Popen
import os
import json
import subprocess
import time
import requests

from dreaming.llm.base import BaseLLMInterface


class LocalLLMInterface(BaseLLMInterface):
    """
    Interface for local LLM models using direct subprocess or local API calls
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_type: str = "llama",
        server_url: str | None = None,
        server_port: int = 8000,
        context_length: int = 4096,
        timeout: int = 60,
        verbose: bool = True,
    ):
        """
        Initialize the local LLM interface

        Args:
            model_path: Path to the model file
            model_type: Model type (llama, gptj, mistral, etc.)
            server_url: URL of existing local API server (if already running)
            server_port: Port to run local server on (if starting new server)
            context_length: Maximum context length for the model
            timeout: Request timeout in seconds
            verbose: Whether to enable verbose output
        """
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.server_url = server_url or f"http://localhost:{server_port}/v1"
        self.server_port = server_port
        self.context_length = context_length
        self.timeout = timeout
        self.verbose = verbose
        self.server_process: Optional[Popen[bytes]] = None
        self._started = False

        # Initialize server if model path is provided and not external server URL
        if model_path and not server_url:
            self._start_local_server()

    def _start_local_server(self) -> bool:
        """Start a local server using llama-cpp-python or other backend"""
        try:
            # Check if server is already running
            try:
                response = requests.get(
                    f"{self.server_url}/models", timeout=self.timeout
                )
                if response.status_code == 200:
                    self.logger.info(f"Local LLM server already running at {self.server_url}")
                    self._started = True
                    return True
            except requests.RequestException:
                pass

            # Determine server command based on model type
            if self.model_type.lower() in ["llama", "mistral", "phi"]:
                cmd = [
                    "python", "-m", "llama_cpp.server",
                    "--model", self.model_path,
                    "--port", str(self.server_port),
                    "--chat_format", "chatml",
                ]
            elif self.model_type.lower() in ["gptj", "gpt4all"]:
                cmd = [
                    "gpt4all-server",
                    "--model", self.model_path,
                    "--port", str(self.server_port),
                ]
            else:
                cmd = [
                    "python", "-m", "llm_server",
                    "--model", self.model_path,
                    "--port", str(self.server_port),
                ]

            cmd_filtered = [str(arg) for arg in cmd if arg is not None]

            if self.verbose:
                self.logger.info(f"Starting local LLM server: {' '.join(cmd_filtered)}")

            self.server_process = subprocess.Popen(
                cmd_filtered,
                stdout=subprocess.PIPE if not self.verbose else None,
                stderr=subprocess.PIPE if not self.verbose else None,
            )

            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    response = requests.get(
                        f"{self.server_url}/models", timeout=self.timeout
                    )
                    if response.status_code == 200:
                        self._started = True
                        self.logger.info(f"Local LLM server started at {self.server_url}")
                        return True
                except requests.RequestException:
                    time.sleep(1)

            self.logger.warning("Failed to start local LLM server within timeout period")
            return False

        except Exception as e:
            self.logger.error(f"Error starting local LLM server: {e}")
            if self.server_process:
                self.server_process.terminate()
            return False

    def set_model(self, model_path: str, model_type: str = "llama") -> bool:
        """Change the active model"""
        self.shutdown()
        self.model_path = model_path
        self.model_type = model_type
        return self._start_local_server()

    def generate_response(
        self, query: str, context: List[Dict[str, Any]] | None = None
    ) -> str:
        """Generate a response using the local LLM"""
        if not self._started:
            return self._fallback_response(query, context or [])

        context_text = (
            self.format_context(context) if context else "No context available."
        )

        messages = [
            {
                "role": "system",
                "content": f"You are a helpful AI assistant.\n\nCONTEXT INFORMATION:\n{context_text}\n\nRespond based on context when available, otherwise use general knowledge.",
            },
            {"role": "user", "content": query},
        ]

        payload = {
            "model": os.path.basename(self.model_path) if self.model_path else "local-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": min(1024, self.context_length // 4),
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.server_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                content = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                return content
            else:
                self.logger.error(f"Local LLM API error: {response.status_code}")
                return self._fallback_response(query, context or [])

        except Exception as e:
            self.logger.error(f"Error generating local LLM response: {e}")
            return self._fallback_response(query, context or [])

    def generate_chat_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response directly from chat messages"""
        if not self._started:
            last_user_msg = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), "")
            return f"Local LLM server not started. Query: {last_user_msg[:100]}..."

        payload = {
            "model": os.path.basename(self.model_path) if self.model_path else "local-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.server_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                return content
            else:
                self.logger.error(f"Local LLM API error: {response.status_code}")
                return "Failed to generate response from local LLM."

        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            return "Error generating response from local LLM."

    def _fallback_response(
        self, query: str, context: List[Dict[str, Any]] | None = None
    ) -> str:
        """Provide a fallback response when LLM generation fails"""
        context_info = (
            f"(with {len(context)} context items)" if context else "(without context)"
        )
        return f"Could not generate response {context_info}. The local language model appears to be unavailable."

    def shutdown(self) -> None:
        """Shutdown the local server if it was started by this interface"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self._started = False
            self.logger.info("Local LLM server shut down")

    def __del__(self):
        """Clean up resources on destruction"""
        self.shutdown()

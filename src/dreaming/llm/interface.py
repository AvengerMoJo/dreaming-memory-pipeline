"""
Unified LLM Interface

Centralized interface for working with different LLM backends.
"""

from typing import Dict, List, Any, Optional
import os
import json

from dreaming.llm.local import LocalLLMInterface
from dreaming.llm.api import APILLMInterface
from dreaming.llm.base import BaseLLMInterface


class LLMInterface:
    """Unified LLM interface that supports both local and API-based models"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the unified LLM interface

        Args:
            config_file: Path to configuration file
        """
        self.interfaces: Dict[str, BaseLLMInterface] = {}
        self.active_interface: Optional[BaseLLMInterface] = None
        self.active_interface_name: Optional[str] = None

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def load_config(self, config_file: str) -> None:
        """Load configuration from file"""
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            if "local_models" in config:
                for name, model_config in config["local_models"].items():
                    self.add_local_interface(
                        name=name,
                        model_path=model_config.get("path"),
                        model_type=model_config.get("type", "llama"),
                        server_url=model_config.get("server_url"),
                        server_port=model_config.get("server_port", 8000),
                        context_length=model_config.get("context_length", 4096),
                        timeout=model_config.get("timeout", 60),
                    )

            if "api_models" in config:
                for name, api_config in config["api_models"].items():
                    self.add_api_interface(
                        name=name,
                        provider=api_config.get("provider"),
                        api_key=api_config.get("api_key"),
                        model=api_config.get("model"),
                        config=api_config,
                    )

            if (
                "default_interface" in config
                and config["default_interface"] in self.interfaces
            ):
                self.set_active_interface(config["default_interface"])
            elif self.interfaces:
                first_interface = next(iter(self.interfaces.keys()))
                self.set_active_interface(first_interface)

        except Exception as e:
            print(f"Error loading configuration: {e}")

    def add_local_interface(
        self,
        name: str,
        model_path: Optional[str],
        model_type: str = "llama",
        server_url: Optional[str] = None,
        server_port: int = 8000,
        context_length: int = 4096,
        timeout: int = 60,
    ) -> None:
        """Add a local LLM interface"""
        self.interfaces[name] = LocalLLMInterface(
            model_path=model_path,
            model_type=model_type,
            server_url=server_url,
            server_port=server_port,
            context_length=context_length,
            timeout=timeout,
        )
        if len(self.interfaces) == 1:
            self.set_active_interface(name)

    def add_api_interface(
        self,
        name: str,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an API-based LLM interface"""
        self.interfaces[name] = APILLMInterface(
            provider=provider, api_key=api_key, model=model, config=config
        )
        if len(self.interfaces) == 1:
            self.set_active_interface(name)

    def set_active_interface(self, name: str) -> bool:
        """Set the active interface"""
        if name in self.interfaces:
            self.active_interface = self.interfaces[name]
            self.active_interface_name = name
            return True
        return False

    def get_available_interfaces(self) -> List[str]:
        """Get names of available interfaces"""
        return list(self.interfaces.keys())

    def generate_response(
        self, query: str, context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate a response using the active interface"""
        if self.active_interface is None:
            return "No active LLM interface configured."
        return self.active_interface.generate_response(query, context)

    def generate_chat_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from chat messages using the active interface"""
        if self.active_interface is None:
            return "No active LLM interface configured."

        if hasattr(self.active_interface, "generate_chat_response"):
            return self.active_interface.generate_chat_response(messages)
        else:
            last_user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )
            return self.active_interface.generate_response(last_user_msg)

    def shutdown(self) -> None:
        """Shutdown all interfaces and clean up resources"""
        for name, interface in self.interfaces.items():
            if hasattr(interface, "shutdown"):
                interface.shutdown()


def create_llm_interface(
    config_file: Optional[str] = None, model_name: str = "default"
) -> LLMInterface:
    """Factory function to create a unified LLM interface"""
    if config_file and os.path.exists(config_file):
        return LLMInterface(config_file=config_file)

    interface = LLMInterface()

    # Try common model paths
    model_dirs = [
        os.path.expanduser("~/.cache/gpt4all/"),
        os.path.expanduser("~/AppData/Local/nomic.ai/GPT4All/"),
        "/usr/local/share/gpt4all/",
        "./models/",
    ]

    model_path = None
    for dir_path in model_dirs:
        if os.path.exists(dir_path):
            model_files = [
                f for f in os.listdir(dir_path)
                if f.endswith((".bin", ".gguf"))
                and os.path.isfile(os.path.join(dir_path, f))
            ]
            if model_files:
                model_path = os.path.join(dir_path, model_files[0])
                break

    if model_path:
        model_type = "llama" if model_path.endswith(".gguf") else "gptj"
        interface.add_local_interface(
            name=model_name, model_path=model_path, model_type=model_type
        )
    else:
        interface.add_local_interface(name="dummy", model_path=None, model_type="none")

    return interface

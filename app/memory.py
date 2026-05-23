import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Manages conversational memory with a configurable sliding window size.
    """
    def __init__(self, system_prompt: str = "", max_window_size: int = 10):
        self.system_prompt = system_prompt
        self.max_window_size = max_window_size
        self.history: List[Dict[str, str]] = []  # list of {"role": "user"|"assistant", "content": "text"}

    def add_message(self, role: str, content: str):
        """Adds a message to the conversation history."""
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be either 'user' or 'assistant'.")
        self.history.append({"role": role, "content": content})

    def clear(self):
        """Clears the chat history."""
        self.history.clear()

    def get_messages_openai(self) -> List[Dict[str, str]]:
        """
        Formats history for OpenAI/HuggingFace API.
        Includes the system prompt if present, and respects the sliding window size.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Take the last `max_window_size` messages
        window = self.history[-self.max_window_size:] if self.max_window_size > 0 else self.history
        messages.extend(window)
        return messages

    def get_messages_gemini(self) -> List[Dict[str, Any]]:
        """
        Formats history for Google Gemini SDK.
        Gemini SDK format for ChatSession history:
        list of {"role": "user"|"model", "parts": [str]}
        Note: The system prompt is set in generation_config or model initialization,
        so it is NOT included in the history list.
        """
        window = self.history[-self.max_window_size:] if self.max_window_size > 0 else self.history
        gemini_history = []
        for msg in window:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        return gemini_history

    def get_raw_history(self) -> List[Dict[str, str]]:
        """Returns the raw history list."""
        return self.history

    def get_windowed_raw_history(self) -> List[Dict[str, str]]:
        """Returns the raw history within the window."""
        return self.history[-self.max_window_size:] if self.max_window_size > 0 else self.history

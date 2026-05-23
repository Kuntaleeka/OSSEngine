import os
import time
import logging
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Import SDKs
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

logger = logging.getLogger(__name__)
load_dotenv()

class LLMClient:
    """
    Unified client for communicating with Gemini (Frontier) and
    Hugging Face Serverless / Groq (OSS) LLM APIs.
    """
    def __init__(self):
        # API Keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")

        # Configure Gemini (using the new google-genai Client)
        self.gemini_client = None
        if self.gemini_api_key and genai:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info("Gemini API Client configured successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client: {e}")
        else:
            logger.warning("Gemini API Key missing or google-genai package not installed.")

    def get_response(
        self,
        provider: str,
        prompt: str,
        history_messages_openai: list,
        history_messages_gemini: list,
        system_prompt: str,
        model_id: Optional[str] = None,
        gen_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sends query to the selected provider.
        Returns a dict:
        {
            "text": str,          # Model response
            "latency": float,     # API call latency in seconds
            "prompt_tokens": int, # Prompt tokens used (or estimate)
            "completion_tokens": int, # Completion tokens used (or estimate)
            "provider": str,      # Provider used
            "model": str,         # Specific model version
            "error": Optional[str] # Error description if any
        }
        """
        if gen_params is None:
            gen_params = {"temperature": 0.7, "max_tokens": 1024, "top_p": 0.9}

        start_time = time.time()
        result = {
            "text": "",
            "latency": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "provider": provider,
            "model": model_id or "default",
            "error": None
        }

        try:
            if provider == "gemini":
                self._call_gemini(prompt, history_messages_gemini, system_prompt, model_id, gen_params, result)
            elif provider == "huggingface":
                self._call_huggingface(history_messages_openai, model_id, gen_params, result)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            logger.error(f"Error calling {provider}: {e}", exc_info=True)
            result["text"] = f"Error: {str(e)}"
            result["error"] = str(e)
        
        result["latency"] = round(time.time() - start_time, 3)

        # Estimate tokens if not populated
        if result["prompt_tokens"] == 0 and result["text"]:
            # Basic approximation: ~4 characters per token
            total_prompt_len = sum(len(m.get("content", "")) for m in history_messages_openai) + len(prompt) + len(system_prompt)
            result["prompt_tokens"] = max(1, total_prompt_len // 4)
        if result["completion_tokens"] == 0 and result["text"] and not result["error"]:
            result["completion_tokens"] = max(1, len(result["text"]) // 4)

        return result

    def _call_gemini(
        self,
        prompt: str,
        history: list,
        system_prompt: str,
        model_id: Optional[str],
        gen_params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        if not self.gemini_client:
            raise ValueError("Gemini Client is not configured. Check GEMINI_API_KEY.")

        model_name = model_id or "gemini-2.5-flash"
        result["model"] = model_name

        # Construct chat session history in the new format
        contents_history = []
        for msg in history:
            role = msg["role"]
            if role not in ["user", "model"]:
                role = "user"
            parts = [types.Part.from_text(text=p) for p in msg["parts"]]
            contents_history.append(types.Content(role=role, parts=parts))

        # Start chat
        chat = self.gemini_client.chats.create(
            model=model_name,
            history=contents_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=gen_params.get("temperature", 0.7),
                max_output_tokens=gen_params.get("max_tokens", 1024),
                top_p=gen_params.get("top_p", 0.9),
            )
        )

        # Send the latest prompt with exponential backoff on rate limits
        max_retries = 5
        backoff_factor = 2.0
        initial_delay = 5.0

        for attempt in range(max_retries):
            try:
                response = chat.send_message(prompt)
                result["text"] = response.text

                # Extract token counts
                try:
                    if response.usage_metadata:
                        result["prompt_tokens"] = response.usage_metadata.prompt_token_count
                        result["completion_tokens"] = response.usage_metadata.candidates_token_count
                except Exception:
                    pass
                break
            except Exception as e:
                err_str = str(e)
                is_retryable = (
                    "429" in err_str or 
                    "503" in err_str or 
                    "RESOURCE_EXHAUSTED" in err_str or 
                    "UNAVAILABLE" in err_str or 
                    "high demand" in err_str or 
                    "Quota exceeded" in err_str
                )
                if is_retryable and attempt < max_retries - 1:
                    sleep_time = initial_delay * (backoff_factor ** attempt)
                    logger.warning(f"Retryable error hit ({err_str[:80]}). Retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise e

    def _call_huggingface(
        self,
        messages: list,
        model_id: Optional[str],
        gen_params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        if not InferenceClient:
            raise ValueError("huggingface-hub is not installed.")

        model_name = model_id or "Qwen/Qwen2.5-7B-Instruct"
        result["model"] = model_name

        # Initialize Hugging Face InferenceClient
        client = InferenceClient(model=model_name, token=self.hf_token)

        # Call API using chat_completion
        response = client.chat_completion(
            messages=messages,
            max_tokens=gen_params.get("max_tokens", 1024),
            temperature=gen_params.get("temperature", 0.7),
            top_p=gen_params.get("top_p", 0.9),
        )

        result["text"] = response.choices[0].message.content
        if response.usage:
            result["prompt_tokens"] = response.usage.prompt_tokens
            result["completion_tokens"] = response.usage.completion_tokens


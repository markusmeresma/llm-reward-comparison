from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
from llm_schemas.score_response import segment_score_response
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import time
import requests
import logging
import json
from mistralai import Mistral
import httpx
from mistralai.models import MistralError
import os
from typing import Optional

SUPPORTED_MODELS_BY_PROVIDER = {
    "openrouter": {
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "openai/gpt-5.2",
    },
    "mistral": {
        "mistral-large-2512",
    },
}

def _validate_provider_model(provider_name: str, model_name: str) -> None:
    if provider_name not in SUPPORTED_MODELS_BY_PROVIDER:
        supported = ", ".join(sorted(SUPPORTED_MODELS_BY_PROVIDER))
        raise ValueError(f"Unknown LLM provider: {provider_name}. Supported: {supported}")
    
    allowed = SUPPORTED_MODELS_BY_PROVIDER[provider_name]
    if model_name not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(
            f"Model '{model_name}' is not supported for provider '{provider_name}'. "
            f"Allowed: {allowed_str}"
        )

def should_retry(e: Exception) -> bool:
    """Retry predicate for transient API failures across both providers."""
    # OpenRouter (requests-based)
    if isinstance(e, (Timeout, ConnectionError)):
        return True
    if isinstance(e, HTTPError) and e.response is not None:
        return e.response.status_code in {408, 429, 500, 502, 503, 504}
    
    # Mistral SDK (hhtpx-based)
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(e, MistralError):
        return e.status_code in {408, 429, 500, 502, 503, 504}    
    return False
    
@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
@dataclass
class LLMResponse:
    content: str
    usage: Optional[Usage] = None
    
@dataclass
class LLMProviderConfig:
    """Shared config for any LLM provider (model name + API key)."""
    model: str
    api_key: str
    temperature: float = 0.0
   
    
class LLMProvider(ABC):
    """Abstract LLM provider — responsible only for making the API call.
    Concrete implementations handle provider-specific request/response formats."""
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        
    @abstractmethod
    def chat_complete(self, messages: list[dict], response_format: dict) -> LLMResponse:
        pass
    
    
class OpenRouterProvider(LLMProvider):
    """LLM provider using the OpenRouter API (requests-based HTTP calls)."""
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, config: LLMProviderConfig, timeout: tuple[int, int] = (5, 30)):
        super().__init__(config)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        
    def chat_complete(self, messages: list[dict], response_format: dict) -> LLMResponse:
        body = {
            "model": self.config.model,
            "messages": messages,
            "response_format": response_format,
            "temperature": self.config.temperature,
        }
        response = self.session.post(self.BASE_URL, json=body, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        
        usage_data = data.get("usage")
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            usage=Usage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"]
            ) if usage_data else None
        )
    
class MistralProvider(LLMProvider):
    """LLM provider using the Mistral SDK (httpx-based)."""
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.client = Mistral(api_key=self.config.api_key)
        
    def chat_complete(self, messages: list[dict], response_format: dict) -> LLMResponse:
        response = self.client.chat.complete(
            model=self.config.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.config.temperature,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ) if response.usage else None
        )
    
    
class LLMClient:
    """High-level client for LLM-based segment evaluation.
    
    Handles prompt formatting, response parsing, retry logic, and per-call
    JSONL logging. On LLM failure (all retries exhausted), returns score=0.0
    so training continues — the affected steps get zero reward and the policy
    gradient is driven by the value function baseline only.
    """
    def __init__(self, provider: LLMProvider, log_dir: Path, run_id: str) -> None:
        self.provider = provider
        self.logger = logging.getLogger(__name__)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"llm_calls_{run_id}.jsonl"
            
        
    def get_request_body(self, task_prompt: str) -> list[dict]:
        return [{ "role": "user", "content": task_prompt }]
        
    def parse_segment_response(self, llm_response: LLMResponse) -> tuple[float, str]:
        """Parse JSON response into (score, reasoning). Raises ValueError on
        malformed responses (missing fields, out-of-range score, invalid JSON)."""
        content = llm_response.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Response not valid JSON: {content[:200]}")

        score = data.get("score")
        if score is None:
            raise ValueError(f"Missing 'score' in response: {data}")
        if not isinstance(score, (int, float)) or not (0 <= score <= 1):
            raise ValueError(f"Invalid score (must be 0-1): {score}")
        
        reasoning = data.get("reasoning", "")
        return float(score), str(reasoning)
                             
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(should_retry)
    )
    def _call_with_retry(self, messages: list[dict], response_format: dict) -> LLMResponse:
        """Make an LLM API call with exponential backoff retry on transient errors."""
        return self.provider.chat_complete(messages, response_format)
    
    def evaluate_segment(self, prompt: str, segment_length: int) -> tuple[float, str]:
        """Evaluate a segment summary via LLM. Returns (score, reasoning).
        On failure, returns (0.0, "LLM_FAILURE") — neutral score so training
        continues without mixing reward signals from different sources."""
        messages = self.get_request_body(prompt)
        start = time.monotonic()
        
        try:
            llm_response = self._call_with_retry(messages, segment_score_response)
            latency = time.monotonic() - start
            score, reasoning = self.parse_segment_response(llm_response)
            self._log_call(prompt, llm_response, score, latency,
                           error=None, reasoning=reasoning, segment_length=segment_length)
            return score, reasoning
        
        except Exception as e:
            latency = time.monotonic() - start
            self.logger.warning(f"Segment evaluation failed: {e}")
            self._log_call(prompt, None, 0.0, latency,
                           error=str(e), reasoning="LLM_FAILURE", segment_length=segment_length)
            return 0.0, "LLM_FAILURE"
        
    def _log_call(self, prompt, response, score, latency, error,
                  reasoning=None, segment_length=None):
        """Append a structured JSONL record for every LLM call (success or failure)."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.provider.config.model,
            "prompt": prompt,
            "response_content": response.content if response else None,
            "score": score,
            "reasoning": reasoning,
            "segment_length": segment_length,
            "latency_s": round(latency, 3),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response and response.usage else None,
            "error": error,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
            
            
def create_provider(provider_name: str, model_name: str, temperature: float = 0.0) -> LLMProvider:
    _validate_provider_model(provider_name, model_name)
    
    if provider_name == "openrouter":
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPEN_ROUTER_API_KEY is missing")
        return OpenRouterProvider(LLMProviderConfig(model=model_name, api_key=api_key, temperature=temperature))
    
    elif provider_name == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is missing")
        return MistralProvider(LLMProviderConfig(model=model_name, api_key=api_key, temperature=temperature))
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
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

def should_retry(e: Exception) -> bool:
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
    """Shared config fields for any provider."""
    model: str
    api_key: str
   
    
class LLMProvider(ABC):
    """Abstract provider only responsible for making the API call."""
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        
    @abstractmethod
    def chat_complete(self, messages: list[dict]) -> LLMResponse:
        pass
    
    
class OpenRouterProvider(LLMProvider):
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, config: LLMProviderConfig, timeout: tuple[int, int] = (5, 30)):
        super().__init__(config)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        
    def chat_complete(self, messages: list[dict]) -> LLMResponse:
        from llm_schemas.score_response import score_response
        body = {
            "model": self.config.model,
            "messages": messages,
            "response_format": score_response
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
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.client = Mistral(api_key=self.config.api_key)
        
    def chat_complete(self, messages: list[dict]) -> LLMResponse:
        response = self.client.chat.complete(
            model=self.config.model,
            messages=messages,
            response_format={"type": "json_object"}
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
    def __init__(self, provider: LLMProvider, log_dir: Path, run_id: str) -> None:
        self.provider = provider
        self.logger = logging.getLogger(__name__)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"llm_calls_{run_id}.jsonl"
            
        
    def get_request_body(self, task_prompt: str) -> list[dict]:
        return [{ "role": "user", "content": task_prompt }]
        
    def parse_implicit_response(self, llm_response: LLMResponse) -> float:
        """Parse LLM response to scalar value. Raise ValueError on failure."""
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
        
        return float(score)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(should_retry)
    )
    def evaluate_trajectory(self, prompt: str) -> float:
        """Evaluate agent trajectory"""
        messages = self.get_request_body(prompt)
        
        self.logger.info("Making LLM request")
        start = time.monotonic()
        try:
            llm_response = self.provider.chat_complete(messages)
            latency = time.monotonic() - start
            score = self.parse_implicit_response(llm_response)
            
            self._log_call(prompt, llm_response, score, latency, error=None)
            return score
        
        except (RequestException, MistralError, httpx.RequestError) as e:
            latency = time.monotonic() - start
            self._log_call(prompt, None, None, latency, error=str(e))
            self.logger.error("Request failed", exc_info=True)
            raise
        
    def _log_call(self, prompt, response, score, latency, error):
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.provider.config.model,
            "prompt": prompt,
            "response_content": response.content if response else None,
            "score": score,
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
            
            
def create_provider(provider_name: str) -> LLMProvider:
    if provider_name == "openrouter":
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPEN_ROUTER_API_KEY is missing")
        model = os.environ.get("OPENROUTER_MODEL")
        if not model:
            raise ValueError("OPENROUTER_MODEL is missing")
        return OpenRouterProvider(LLMProviderConfig(model=model, api_key=api_key))
    
    elif provider_name == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is missing")
        model = os.environ.get("MISTRAL_MODEL")
        if not model:
            raise ValueError("MISTRAL_MODEL is missing")
        return MistralProvider(LLMProviderConfig(model=model, api_key=api_key))
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
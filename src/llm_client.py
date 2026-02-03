from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
from dataclasses import dataclass
from .llm_schemas.score_response import score_response
from typing import Any
from pathlib import Path
from datetime import datetime
import time
import requests
import os
import logging
import json

def should_retry(e: Exception) -> bool:
    if isinstance(e, (Timeout, ConnectionError)):
        return True
        
    if isinstance(e, HTTPError) and e.response is not None:
        return e.response.status_code in {408, 429, 500, 502, 503, 504}
        
    return False

@dataclass
class ChatMessage:
    role: str
    content: str
    
@dataclass
class ChatChoice:
    message: ChatMessage
    finish_reason: str
    
@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    
@dataclass
class LLMResponse:
    id: str
    model: str
    choices: list[ChatChoice]
    usage: Usage
    
@dataclass
class OpenRouterConfig:
    model: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: tuple[int, int] = (5, 30)
    
    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        model = os.environ.get("MODEL")
        if not api_key:
            raise ValueError("OPEN_ROUTER_API_KEY is missing")
        if not model:
            raise ValueError("MODEL is missing")
        return cls(api_key=api_key, model=model)
    
class LLMClient:
    def __init__(self, config: OpenRouterConfig, log_dir: Path, run_id: str) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        self.logger = logging.getLogger(__name__)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"llm_calls_{run_id}.jsonl"
            
        
    def get_request_body(self, task_prompt: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                { "role": "user", "content": task_prompt }
            ],
            "response_format": score_response
        }
        
    def parse_implicit_response(self, llm_response) -> float:
        """Parse LLM response to scalar value. Raise ValueError on failure."""
        try:
            content = llm_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Malformed response structure: {e}")
        
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
        request_body = self.get_request_body(prompt)
        
        self.logger.info("Making LLM request")
        start = time.monotonic()
        try:
            response = self.session.post(
                self.config.base_url,
                json=request_body,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            llm_response = response.json()
            latency = time.monotonic() - start
            score = self.parse_implicit_response(llm_response)
            
            self._log_call(prompt, llm_response, score, latency, error=None)
            return score
        
        except RequestException as e:
            latency = time.monotonic() - start
            self._log_call(prompt, None, None, latency, error=str(e))
            self.logger.error("Request failed", exc_info=True)
            raise
        
    def _log_call(self, prompt, response, score, latency, error):
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model,
            "prompt": prompt,
            "response_content": response["choices"][0]["message"]["content"] if response else None,
            "score": score,
            "latency_s": round(latency, 3),
            "usage": response.get("usage") if response else None,
            "error": error,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
from dataclasses import dataclass
from .llm_schemas.score_response import score_response
from typing import Any
import requests
import os
import logging

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
    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        self.logger = logging.getLogger(__name__)
        
    def get_request_body(self, task_prompt: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                { "role": "user", "content": task_prompt }
            ],
            "response_format": score_response
        }
        
    def parse_implicit_response(self, llm_response) -> float:
        """Parse LLM response to scalar value"""
        pass
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(should_retry)
    )
    def evaluate_trajectory(self, task_prompt: str) -> float:
        """Evaluate agent trajectory"""
        
        self.logger.info("Making LLM request")
        request_body = self.get_request_body(task_prompt)
        try:
            response = self.session.post(
                self.config.base_url,
                json=request_body,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return self.parse_implicit_response(response.json())
        except RequestException as e:
            self.logger.error("Request failed", exc_info=True)
            raise
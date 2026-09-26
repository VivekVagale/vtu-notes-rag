"""Swappable LLM providers, chosen with LLM_PROVIDER in .env.

anthropic  - Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
ollama     - any local model served by Ollama (free, offline)
extractive - no LLM at all: the answer is composed from the retrieved passages
             (see extractive.py). Useful with no key and no GPU, and it is what
             the tests use.
"""

from __future__ import annotations

from .config import Settings

PROVIDERS = ("anthropic", "ollama", "extractive")


class LLMError(RuntimeError):
    """Raised when a provider is misconfigured or unreachable."""


class LLMProvider:
    name = "base"
    model = "-"

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Put it in .env, or switch "
                "LLM_PROVIDER to ollama or extractive."
            )
        self.model = model
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install anthropic") from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        parts = [b.text for b in message.content if getattr(b, "type", "") == "text"]
        return "\n".join(parts).strip()


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: int = 180) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        import requests

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"Ollama is not reachable at {self.host}. Start it with 'ollama serve' "
                f"and pull the model with 'ollama pull {self.model}'."
            ) from exc
        except Exception as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()


def get_provider(settings: Settings) -> LLMProvider | None:
    """None means 'no external LLM' - the extractive composer handles it."""
    provider = (settings.llm_provider or "extractive").lower()
    if provider == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    if provider == "ollama":
        return OllamaProvider(settings.ollama_host, settings.ollama_model)
    if provider == "extractive":
        return None
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(PROVIDERS)}")

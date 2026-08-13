"""Thread-safe DeepSeek JSON completion client."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class DeepSeekJsonClient:
    """Call DeepSeek with bounded concurrency and non-thinking JSON output."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        concurrency: int = 16,
        max_tokens: int = 3072,
        timeout: float = 180.0,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        if client is None:
            key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY is required; create project-root .env or export it"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("DeepSeek generation requires the openai package") from exc
            client = OpenAI(
                api_key=key,
                base_url=base_url
                or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
                timeout=timeout,
                max_retries=0,
            )
        self._client = client

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int | None = None,
        **_: object,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens or self.max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek returned an empty response")
        return content

    def complete_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_new_tokens: int | None = None,
        **_: object,
    ) -> list[str | Exception]:
        """Complete requests concurrently while preserving input order."""
        if not messages_batch:
            return []

        def invoke(messages: list[dict[str, str]]) -> str:
            return self.complete(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        workers = min(self.concurrency, len(messages_batch))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(invoke, messages) for messages in messages_batch]
            outcomes: list[str | Exception] = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except Exception as exc:
                    outcomes.append(exc)
        return outcomes

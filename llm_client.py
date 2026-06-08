from __future__ import annotations

import json
import re
import time
from typing import Any


def call_anthropic_with_retry(
    client,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int = 2000,
    retries: int = 3,
    backoff: float = 5.0,
) -> str:
    import anthropic

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text
        except anthropic.RateLimitError as exc:
            last_error = exc
            time.sleep(backoff * (attempt + 1))
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                last_error = exc
                time.sleep(backoff * (attempt + 1))
            else:
                raise
        except anthropic.APIConnectionError as exc:
            last_error = exc
            time.sleep(backoff * (attempt + 1))

    raise last_error or RuntimeError("Anthropic API call failed after retries.")


def call_openai_with_retry(
    client,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int = 2000,
    retries: int = 3,
    backoff: float = 5.0,
) -> str:
    import openai

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )
            content = response.choices[0].message.content or ""
            if not content:
                finish_reason = response.choices[0].finish_reason
                usage = getattr(response, "usage", None)
                raise RuntimeError(
                    "OpenAI returned an empty message content. "
                    f"finish_reason={finish_reason}, usage={usage}"
                )
            return content
        except openai.RateLimitError as exc:
            last_error = exc
            time.sleep(backoff * (attempt + 1))
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                last_error = exc
                time.sleep(backoff * (attempt + 1))
            else:
                raise
        except openai.APIConnectionError as exc:
            last_error = exc
            time.sleep(backoff * (attempt + 1))

    raise last_error or RuntimeError("OpenAI API call failed after retries.")


def call_llm_with_retry(
    provider: str,
    api_key: str,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int = 2000,
) -> str:
    if provider == "Anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        return call_anthropic_with_retry(
            client=client,
            model=model,
            system=system,
            user_content=user_content,
            max_tokens=max_tokens,
        )

    if provider == "OpenAI":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        return call_openai_with_retry(
            client=client,
            model=model,
            system=system,
            user_content=user_content,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def parse_llm_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[^\n]*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise

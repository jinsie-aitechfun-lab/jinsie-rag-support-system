from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class ChatCompletionError(RuntimeError):
    pass


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def has_chat_env() -> bool:
    # Minimal gate: only require API key. base_url/model have defaults.
    return bool(_env("OPENAI_API_KEY"))


def _join_chat_completions_url(base_url: str) -> str:
    """
    SiliconFlow / OpenAI-compatible base_url styles:
    - https://api.siliconflow.cn/v1  -> chat completions path should be /chat/completions
    - https://api.siliconflow.cn     -> chat completions path should be /v1/chat/completions

    We normalize both to a correct final URL.
    """
    b = (base_url or "").rstrip("/")
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"


def chat_answer(
    prompt: str,
    *,
    temperature: float = 0.2,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Minimal OpenAI-compatible chat.completions call.

    Env:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (default: https://api.openai.com)
    - OPENAI_CHAT_MODEL (default: gpt-4o-mini)
    """
    api_key = _env("OPENAI_API_KEY")
    base_url = _env("OPENAI_BASE_URL", "https://api.openai.com")

    # ⭐ 兼容项目一：优先 OPENAI_MODEL；没有才读 OPENAI_CHAT_MODEL
    model = _env("OPENAI_MODEL") or _env("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ChatCompletionError("missing env: OPENAI_API_KEY")

    url = _join_chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严谨的企业知识库问答助手。优先依据给定上下文回答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        raise ChatCompletionError(f"chat request failed: {e}") from e

    if r.status_code >= 400:
        raise ChatCompletionError(f"chat http {r.status_code}: {r.text[:300]}")

    try:
        data = r.json()
    except Exception as e:
        raise ChatCompletionError(f"chat response is not json: {e}") from e

    choices = data.get("choices") or []
    if not choices:
        raise ChatCompletionError("chat response missing choices")

    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ChatCompletionError("chat response missing message.content")

    usage: Optional[Dict[str, Any]] = data.get("usage")
    return {
        "text": content.strip(),
        "model": model,
        "usage": usage or {},
        "raw": {"id": data.get("id", ""), "finish_reason": (choices[0] or {}).get("finish_reason", "")},
    }

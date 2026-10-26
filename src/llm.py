import json
import os
from typing import Callable, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv(override=True)

Messages = list[dict]
LLM = Callable[[Messages], str]

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily so tests never need it
        _client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            max_retries=5,  # SDK backs off automatically on 429 rate limits
        )
    return _client


def _call(messages: Messages, json_mode: bool) -> str:
    kwargs = {}
    if json_mode and os.getenv("LLM_JSON_MODE", "1") == "1":
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        messages=messages,
        temperature=0 if json_mode else 0.3,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def groq_chat(messages: Messages) -> str:
    """Plain-text reply (used for chat and for summaries)."""
    return _call(messages, json_mode=False)


def groq_json(messages: Messages) -> str:
    """JSON-only reply (used for fact extraction)."""
    return _call(messages, json_mode=True)


def structured_call(system: str, user: str, schema: type[BaseModel], llm: LLM,
                    retries: int = 2) -> Optional[BaseModel]:
    """Same idea as Project 1: validate, feed the exact error back, retry."""
    messages: Messages = [
        {"role": "system", "content": (
            system + "\nReturn ONLY a JSON object matching this JSON schema. "
            "No prose, no markdown.\n" + json.dumps(schema.model_json_schema()))},
        {"role": "user", "content": user},
    ]
    for _ in range(retries + 1):
        raw = llm(messages)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your output failed validation:\n{e}\n"
                                            "Return ONLY the corrected JSON."},
            ]
    return None
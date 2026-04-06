"""Shared LLM client and helpers for OpenRouter API calls."""

import json
import os
import re
import time

from openai import OpenAI, RateLimitError

MODELS = [
    "qwen/qwen3.6-plus:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openai/gpt-oss-120b:free",
    "minimax/minimax-m2.5:free",
    "stepfun/step-3.5-flash:free",
]


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling thinking tags and markdown fences."""
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:300]}")


def extract_code(text: str) -> str:
    """Extract Python code from LLM response, handling thinking tags and markdown fences."""
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Try extracting from markdown code fences
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no fences, return as-is (might already be raw code)
    return text


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Send a chat completion request with retry and model fallback. Returns the raw response text."""
    client = get_client()

    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except RateLimitError:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"  Rate limited on {model}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Rate limited on {model}, trying next model...")
                    break

    raise RuntimeError("All models rate limited. Please try again in a minute.")

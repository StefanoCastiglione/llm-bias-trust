"""
Shared helper for chat calls to OpenAI-compatible clients.

Some reasoning models place their internal reasoning inside <think>...</think> tags in the same content field as the final answer. This module returns only the final answer for any client and model.

Each call asks the provider to hide reasoning with the reasoning_format parameter, sent through extra_body. If the call fails with a bad request, unprocessable entity or type error, it is repeated without the parameter and the result is remembered for that base_url and model pair, even if the error had another cause. In every case any remaining think blocks are removed from the text, including a block that was cut off before its closing tag.
"""

import re
from openai import BadRequestError, UnprocessableEntityError

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE  = re.compile(r"<think>.*$", re.DOTALL)  # think block that was never closed

# errors treated as a possible rejection of the extra parameter, including unrelated 400 errors
_UNSUPPORTED_PARAM_ERRORS = (BadRequestError, UnprocessableEntityError, TypeError)

# remembers whether reasoning_format is accepted for each base_url and model pair
_reasoning_format_support_cache: dict = {}


def strip_thinking(text: str) -> str:
    """
    Remove think blocks from a text, including a block that was never closed.
    Return the stripped text.
    """
    # nothing to remove when the text has no think tag
    if "<think>" not in text:
        return text.strip()
    # closed blocks go first and then any block left open at the end
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


def _cache_key(client, model: str):
    return (str(getattr(client, "base_url", "")), model)


def chat_completion_text(client, model: str, messages: list, **kwargs) -> str:
    """
    Run a chat completion and return the text of the first choice, stripped and without think blocks.
    It takes the same arguments as client.chat.completions.create, passes extra keyword arguments through and merges extra_body with reasoning_format.
    """
    # copy the caller extra_body so it is never modified
    base_extra_body = dict(kwargs.pop("extra_body", {}) or {})
    key = _cache_key(client, model)
    # reasoning_format is assumed to be accepted until a provider rejects it
    supported = _reasoning_format_support_cache.get(key, True)

    def _call(with_reasoning_format: bool):
        extra_body = dict(base_extra_body)
        # send reasoning_format only when requested
        if with_reasoning_format:
            extra_body["reasoning_format"] = "hidden"
        return client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body or None,
            **kwargs,
        )

    if supported:
        try:
            resp = _call(with_reasoning_format=True)
        except _UNSUPPORTED_PARAM_ERRORS:
            # assume the parameter was rejected, remember it and retry without it
            _reasoning_format_support_cache[key] = False
            resp = _call(with_reasoning_format=False)
    else:
        # the parameter is already known as unsupported for this provider and model
        resp = _call(with_reasoning_format=False)

    text = resp.choices[0].message.content.strip()
    return strip_thinking(text)


# EXAMPLE

if __name__ == "__main__":
    from openai import OpenAI

    BASE_URL = "https://your-endpoint/v1"
    API_KEY = "your-api-key"
    MODEL_NAME = "your-model"

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # the returned text has no think blocks
    text = chat_completion_text(
        client,
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(text)

"""Provider-neutral OpenAI-compatible client. Configure model IDs for your provider."""
import os
import time

# Public display names; API identifiers are explicitly supplied by the user.
MODEL_NAMES = {"deepseek": "DeepSeek V4.1 Flash 1017", "glm": "GLM 5.3 Flash", "qwen38": "Qwen 3.8 27B"}
MODELS = {key: (os.environ.get(f"WSP_{key.upper()}_MODEL", ""),
               os.environ.get(f"WSP_{key.upper()}_EFFORT", effort))
          for key, effort in [("deepseek", "max"), ("glm", "max"), ("qwen38", "xhigh")]}

def client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ["OPENAI_BASE_URL"].rstrip("/"),
                  api_key=os.environ["OPENAI_API_KEY"], timeout=3600, max_retries=0)

def chat(cl, model_key, messages, max_tokens=64000, retries=8):
    """One completion. Returns (content, reasoning, usage dict). Retries 429/5xx/timeouts with backoff.

    Streamed: a long max-effort reasoning turn otherwise sits silent past the gateway's response timeout and
    comes back as a 502 (seen with GLM on the no-tool identification task)."""
    model, effort = MODELS[model_key]
    if not model:
        raise ValueError(f"Set WSP_{model_key.upper()}_MODEL to your provider model ID")
    delay = 20
    for attempt in range(retries):
        try:
            stream = cl.chat.completions.create(model=model, messages=messages, reasoning_effort=effort,
                                                max_tokens=max_tokens, stream=True,
                                                stream_options={"include_usage": True})
            content, reasoning, usage, finish = [], [], {}, None
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                for ch in chunk.choices:
                    d = ch.delta
                    content.append(d.content or "")
                    reasoning.append(getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None) or "")
                    finish = ch.finish_reason or finish
            usage["finish_reason"] = finish  # "length" = cut off at max_tokens; the agent loop must not score it
            return "".join(content), "".join(reasoning), usage
        except Exception as e:  # openai.APIStatusError, timeouts, connection resets
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403) or attempt == retries - 1:
                raise
            print(f"[llm] {model} attempt {attempt}: {type(e).__name__} {status}; retry in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)

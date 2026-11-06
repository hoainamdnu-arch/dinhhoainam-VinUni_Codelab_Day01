"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import sys
import time
from typing import Any, Callable

# Windows terminals default to a legacy codepage (e.g. cp1252) that cannot
# encode Vietnamese diacritics, so printing Vietnamese text raises
# UnicodeEncodeError. Force stdout/stdin to UTF-8 to keep output readable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

# Load API keys from a .env file so they persist across sessions.
# find_dotenv walks up from the current directory, so .env can live in
# starter-code/ or the repository root.
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
 
    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage
    


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start
 
    text = response.text
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return text, latency, usage
   

# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start
 
    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage
    


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------

def _calc_cost(model_key: str, usage: dict) -> float:
    """Chi phí USD = (in*in_rate + out*out_rate) / 1,000,000."""
    rates = PRICING_1M_TOKENS[model_key]
    return (
        usage["input_tokens"] * rates["input"]
        + usage["output_tokens"] * rates["output"]
    ) / 1_000_000
 
def compare_models(prompt: str) -> dict:

    text_4o, lat_4o, u_4o = call_openai(prompt)
    text_mini, lat_mini, u_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    text_gem, lat_gem, u_gem = call_gemini(prompt)
 
    return {
        "gpt4o": {
            "response": text_4o,
            "latency": lat_4o,
            "cost": _calc_cost("gpt-4o", u_4o),
            "input_tokens": u_4o["input_tokens"],
            "output_tokens": u_4o["output_tokens"],
        },
        "gpt4o_mini": {
            "response": text_mini,
            "latency": lat_mini,
            "cost": _calc_cost("gpt-4o-mini", u_mini),
            "input_tokens": u_mini["input_tokens"],
            "output_tokens": u_mini["output_tokens"],
        },
        "gemini_flash": {
            "response": text_gem,
            "latency": lat_gem,
            "cost": _calc_cost("gemini-2.5-flash", u_gem),
            "input_tokens": u_gem["input_tokens"],
            "output_tokens": u_gem["output_tokens"],
        },
    }

# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai
 
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict] = []
 
    print("Gemini chatbot — gõ 'quit' hoặc 'exit' để thoát.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("quit", "exit"):
            print("Tạm biệt!")
            break
 
        # Thêm lượt của người dùng vào lịch sử.
        history.append({"role": "user", "parts": [{"text": user_input}]})
        # Giới hạn 3 lượt gần nhất (6 message).
        history = history[-6:]
 
        print("Gemini: ", end="", flush=True)
        full_reply = ""
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history,
        )
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_reply += chunk.text
        print() 
 
        
        history.append({"role": "model", "parts": [{"text": full_reply}]})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(); nếu lỗi thì thử lại tối đa max_retries lần, mỗi lần chờ
    delay = base_delay * 2^attempt. Thất bại hết thì ném lại exception cuối.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))
            attempt += 1
# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt, gắn thêm khóa "prompt" vào mỗi kết quả.
    """
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng danh sách kết quả batch_compare thành bảng Markdown.
    Cột: | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    labels = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
 
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|--------|-------|----------------------|---------|-----------------|------------|",
    ]
 
    for result in results:
        prompt = result.get("prompt", "")
        for key in ("gpt4o", "gpt4o_mini", "gemini_flash"):
            stats = result[key]
            response = str(stats["response"]).replace("\n", " ")
            if len(response) > 50:
                response = response[:50] + "..."
            lines.append(
                f"| {prompt} | {labels[key]} | {response} "
                f"| {stats['latency']:.2f}s "
                f"| {stats['input_tokens']}/{stats['output_tokens']} "
                f"| ${stats['cost']:.6f} |"
            )
 
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")
 
    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot error: {e}")



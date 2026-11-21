"""Ollama client with timeouts, retries, and usage logging."""

import os
import time
from dataclasses import dataclass

import requests

RETRY_STATUS = (408, 429, 500, 502, 503, 504)
MAX_ATTEMPTS = 3


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class OllamaProvider:

    provider_name = "Ollama"

    def __init__(self, model=None, url=None):

        self.model = model or os.getenv(
            "OLLAMA_MODEL", "llama3.2"
        )

        base = url or os.getenv(
            "OLLAMA_URL", "http://localhost:11434"
        )

        self.url = base.rstrip("/") + "/api/generate"

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=(5, 120)
        )

        response.raise_for_status()
        data = response.json()

        return LLMResult(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0
        )


def is_retryable(error):

    if isinstance(
        error,
        (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        )
    ):
        return True

    if (
        isinstance(error, requests.exceptions.HTTPError)
        and error.response is not None
    ):
        return error.response.status_code in RETRY_STATUS

    return False


class LLMClient:

    def __init__(self, provider, usage_logger, sleep=time.sleep):

        self.provider = provider
        self.usage_logger = usage_logger
        self._sleep = sleep

    def _cost(self, result):

        in_rate = float(
            os.getenv("COMPARE_INPUT_PRICE_PER_MILLION", "0")
        )

        out_rate = float(
            os.getenv("COMPARE_OUTPUT_PRICE_PER_MILLION", "0")
        )

        cost = (
            result.input_tokens * in_rate
            + result.output_tokens * out_rate
        ) / 1_000_000

        kind = (
            "equivalent_hosted_estimate"
            if (in_rate or out_rate)
            else "local_no_api_charge"
        )

        return round(cost, 8), kind

    def generate(self, prompt):

        start = time.perf_counter()

        for attempt in range(1, MAX_ATTEMPTS + 1):

            try:
                result = self.provider.generate(prompt)

            except Exception as error:

                if is_retryable(error) and attempt < MAX_ATTEMPTS:

                    delay = 2 ** (attempt - 1)

                    print(
                        f"Temporary error ({type(error).__name__}). "
                        f"Retry {attempt} of {MAX_ATTEMPTS - 1} "
                        f"in {delay}s."
                    )

                    self._sleep(delay)
                    continue

                self.usage_logger.log_call({
                    "provider": self.provider.provider_name,
                    "model": self.provider.model,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "latency_ms": round(
                        (time.perf_counter() - start) * 1000, 2
                    ),
                    "estimated_cost_usd": None,
                    "status": "failed",
                    "attempts": attempt,
                    "error": str(error)
                })

                raise

            latency_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )

            cost, cost_type = self._cost(result)

            self.usage_logger.log_call({
                "provider": self.provider.provider_name,
                "model": self.provider.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": (
                    result.input_tokens + result.output_tokens
                ),
                "latency_ms": latency_ms,
                "estimated_cost_usd": cost,
                "cost_type": cost_type,
                "status": "success",
                "attempts": attempt
            })

            print(
                f"Latency: {latency_ms} ms | "
                f"Input tokens: {result.input_tokens} | "
                f"Output tokens: {result.output_tokens} | "
                f"Cost (USD): {cost}"
            )

            return result.text
"""Offline tests: no Ollama and no MongoDB needed."""

import unittest
from unittest.mock import patch, MagicMock

import requests

from llm_client import LLMClient, LLMResult, OllamaProvider


class FakeProvider:

    provider_name = "Test"
    model = "fake-model"

    def generate(self, prompt):

        return LLMResult(
            "AI stands for Artificial Intelligence.",
            5,
            8
        )


class FlakyProvider(FakeProvider):

    def __init__(self, error, fail_times):

        self.error = error
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, prompt):

        self.calls += 1

        if self.calls <= self.fail_times:
            raise self.error

        return super().generate(prompt)


class FakeLogger:

    def __init__(self):
        self.records = []

    def log_call(self, record):

        self.records.append(record)
        return True


def http_error(code):

    resp = MagicMock(status_code=code)

    return requests.exceptions.HTTPError(
        f"{code}",
        response=resp
    )


def make_client(provider):

    logger = FakeLogger()

    return (
        LLMClient(provider, logger, sleep=lambda s: None),
        logger
    )


class TestLLMClient(unittest.TestCase):

    def test_success_is_logged(self):

        client, logger = make_client(FakeProvider())

        self.assertEqual(
            client.generate("Explain AI"),
            "AI stands for Artificial Intelligence."
        )

        rec = logger.records[0]

        self.assertEqual(len(logger.records), 1)

        self.assertEqual(
            (
                rec["input_tokens"],
                rec["output_tokens"],
                rec["total_tokens"]
            ),
            (5, 8, 13)
        )

        self.assertEqual(
            (rec["status"], rec["attempts"]),
            ("success", 1)
        )

        self.assertGreaterEqual(rec["latency_ms"], 0)

        self.assertEqual(rec["estimated_cost_usd"], 0.0)

    def test_cost_uses_rates(self):

        client, logger = make_client(FakeProvider())

        env = {
            "COMPARE_INPUT_PRICE_PER_MILLION": "1000000",
            "COMPARE_OUTPUT_PRICE_PER_MILLION": "2000000"
        }

        with patch.dict("os.environ", env):
            client.generate("x")

        self.assertAlmostEqual(
            logger.records[0]["estimated_cost_usd"],
            5 * 1 + 8 * 2
        )

    def test_timeout_is_retried_then_succeeds(self):

        prov = FlakyProvider(
            requests.exceptions.ReadTimeout("slow"),
            2
        )

        client, logger = make_client(prov)

        client.generate("x")

        self.assertEqual(prov.calls, 3)
        self.assertEqual(logger.records[0]["attempts"], 3)

    def test_503_is_retried(self):

        prov = FlakyProvider(http_error(503), 1)

        client, logger = make_client(prov)

        client.generate("x")

        self.assertEqual(prov.calls, 2)

    def test_404_is_not_retried(self):

        prov = FlakyProvider(http_error(404), 5)

        client, logger = make_client(prov)

        with self.assertRaises(requests.exceptions.HTTPError):
            client.generate("x")

        self.assertEqual(prov.calls, 1)

        self.assertEqual(
            logger.records[0]["status"],
            "failed"
        )

    def test_gives_up_after_three_attempts_and_logs_failure(self):

        prov = FlakyProvider(
            requests.exceptions.ConnectionError("down"),
            99
        )

        client, logger = make_client(prov)

        with self.assertRaises(
            requests.exceptions.ConnectionError
        ):
            client.generate("x")

        self.assertEqual(prov.calls, 3)

        rec = logger.records[0]

        self.assertEqual(
            (
                rec["status"],
                rec["attempts"],
                rec["total_tokens"]
            ),
            ("failed", 3, None)
        )


class TestOllamaProvider(unittest.TestCase):

    @patch("llm_client.requests.post")
    def test_reads_token_counts_and_sets_timeout(self, post):

        post.return_value.json.return_value = {
            "response": "hi",
            "prompt_eval_count": 7,
            "eval_count": 3
        }

        r = OllamaProvider(
            "m",
            "http://x:1"
        ).generate("q")

        self.assertEqual(
            (r.text, r.input_tokens, r.output_tokens),
            ("hi", 7, 3)
        )

        self.assertEqual(
            post.call_args.kwargs["timeout"],
            (5, 120)
        )

        self.assertEqual(
            post.call_args.args[0],
            "http://x:1/api/generate"
        )


if __name__ == "__main__":
    unittest.main()
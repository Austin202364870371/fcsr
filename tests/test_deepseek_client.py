import threading
import time
import unittest
from types import SimpleNamespace

from deepseek_client import DeepSeekJsonClient


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def create(self, **kwargs):
        content = kwargs["messages"][-1]["content"]
        with self.lock:
            self.calls.append(kwargs)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        if content == "fail":
            raise RuntimeError("temporary API failure")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f'{{"value":"{content}"}}'),
                    finish_reason="stop",
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class DeepSeekJsonClientTests(unittest.TestCase):
    def test_uses_v4_flash_non_thinking_json_output(self) -> None:
        api = FakeOpenAIClient()
        client = DeepSeekJsonClient(client=api)

        response = client.complete(
            [{"role": "user", "content": "one"}],
            temperature=0.0,
        )

        self.assertEqual(response, '{"value":"one"}')
        self.assertEqual(response.finish_reason, "stop")
        request = api.completions.calls[0]
        self.assertEqual(request["model"], "deepseek-v4-flash")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(request["max_tokens"], 6144)

    def test_complete_many_is_bounded_and_preserves_individual_failures(self) -> None:
        api = FakeOpenAIClient()
        client = DeepSeekJsonClient(client=api, concurrency=2)
        messages = [
            [{"role": "user", "content": value}]
            for value in ("one", "fail", "three")
        ]

        outcomes = client.complete_many(messages, temperature=0.0)

        self.assertEqual(outcomes[0], '{"value":"one"}')
        self.assertIsInstance(outcomes[1], RuntimeError)
        self.assertEqual(outcomes[2], '{"value":"three"}')
        self.assertEqual(api.completions.max_active, 2)


if __name__ == "__main__":
    unittest.main()

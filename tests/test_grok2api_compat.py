from __future__ import annotations

import os
import unittest

from mysearch.clients import MySearchClient
from mysearch.config import MySearchConfig
from proxy import server as proxy_server


class Grok2APICompatibilityTests(unittest.TestCase):
    def test_grok2api_mode_uses_responses_data_plane(self) -> None:
        names = (
            "MYSEARCH_PROXY_BASE_URL",
            "MYSEARCH_PROXY_API_KEY",
            "MYSEARCH_XAI_SEARCH_MODE",
            "MYSEARCH_XAI_BASE_URL",
            "MYSEARCH_XAI_API_KEY",
            "MYSEARCH_XAI_MODEL",
        )
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ.pop(name, None)
            os.environ["MYSEARCH_XAI_SEARCH_MODE"] = "grok2api"
            os.environ["MYSEARCH_XAI_BASE_URL"] = "http://127.0.0.1:18000/v1"
            os.environ["MYSEARCH_XAI_API_KEY"] = "test-client-key"
            os.environ["MYSEARCH_XAI_MODEL"] = "grok-test"

            config = MySearchConfig.from_env()
            client = MySearchClient(config=config)
            payload = client._build_xai_responses_payload(
                query="OpenAI",
                sources=["x"],
                max_results=3,
                include_domains=None,
                exclude_domains=None,
                allowed_x_handles=["OpenAI"],
                excluded_x_handles=None,
                from_date=None,
                to_date=None,
                include_x_images=False,
                include_x_videos=False,
            )

            self.assertEqual(config.xai.search_mode, "grok2api")
            self.assertEqual(config.xai.base_url, "http://127.0.0.1:18000/v1")
            self.assertEqual(config.xai.path("responses"), "/responses")
            self.assertEqual(payload["model"], "grok-test")
            self.assertEqual(payload["tools"], [{"type": "x_search", "allowed_x_handles": ["OpenAI"]}])
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_new_admin_accounts_map_to_social_stats(self) -> None:
        stats = proxy_server.build_social_account_stats(
            {
                "data": {
                    "items": [
                        {
                            "provider": "grok_web",
                            "enabled": True,
                            "authStatus": "active",
                            "quotaWindows": [{"remaining": 7}, {"remaining": 30}],
                        },
                        {
                            "provider": "grok_web",
                            "enabled": False,
                            "authStatus": "disabled",
                            "quotaWindows": [],
                        },
                    ]
                }
            }
        )
        self.assertEqual(stats["token_total"], 2)
        self.assertEqual(stats["token_normal"], 1)
        self.assertEqual(stats["token_invalid"], 1)
        self.assertEqual(stats["chat_remaining"], 37)
        self.assertEqual(stats["pools"], [
            {"pool": "grok_web", "count": 2, "active": 1, "cooling": 0, "invalid": 1},
        ])


if __name__ == "__main__":
    unittest.main()

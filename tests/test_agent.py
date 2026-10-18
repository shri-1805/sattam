import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from human_rights_graphrag.agent import (
    LLMClauseAgent,
    LLMError,
    build_clause_responses,
    build_ssl_context,
    format_upstream_error,
)
from human_rights_graphrag.models import (
    ArticleContext,
    ModelSelection,
    SelectedClause,
    SubClauseContext,
)


class ClauseSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = ArticleContext(
            id="IND-21",
            document="Constitution of India",
            reference="Article 21",
            title="Protection of life and personal liberty",
            text="Full article text containing multiple provisions.",
            score=0.9,
            subclauses=[
                SubClauseContext(id="IND-21-1", text="(1) First provision."),
                SubClauseContext(id="IND-21-2", text="(2) Second provision."),
            ],
            rights=["life and personal liberty"],
        )

    def test_returns_exact_selected_subclause_text(self) -> None:
        selection = ModelSelection(
            clauses=[
                SelectedClause(
                    id="IND-21-2", applicability="The scenario describes this issue."
                )
            ]
        )

        clauses = build_clause_responses([self.candidate], selection)

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].id, "IND-21-2")
        self.assertEqual(clauses[0].article_id, "IND-21")
        self.assertEqual(clauses[0].reference, "Article 21")
        self.assertEqual(clauses[0].text, "(2) Second provision.")

    def test_rejects_ids_not_present_in_retrieved_context(self) -> None:
        selection = ModelSelection(
            clauses=[SelectedClause(id="MADE-UP", applicability="Irrelevant")]
        )

        with self.assertRaises(LLMError):
            build_clause_responses([self.candidate], selection)


class TLSConfigurationTests(unittest.TestCase):
    @patch("human_rights_graphrag.agent.ssl.create_default_context")
    @patch("human_rights_graphrag.agent.certifi.where", return_value="public-roots.pem")
    def test_adds_private_ca_to_default_roots(
        self, certifi_where: MagicMock, create_context: MagicMock
    ) -> None:
        context = MagicMock()
        create_context.return_value = context
        ca_bundle = Path("organization-ca.pem")

        result = build_ssl_context(ca_bundle)

        create_context.assert_called_once_with(cafile="public-roots.pem")
        context.load_verify_locations.assert_called_once_with(
            cafile=str(ca_bundle)
        )
        self.assertIs(result, context)


class UpstreamErrorTests(unittest.TestCase):
    def test_extracts_provider_message_and_redacts_api_key(self) -> None:
        response = httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid request using secret-key",
                    "code": "invalid_request",
                }
            },
        )

        detail = format_upstream_error(response, "secret-key")

        self.assertEqual(detail, "invalid_request: Invalid request using [redacted]")

    def test_handles_non_json_upstream_errors(self) -> None:
        response = httpx.Response(502, text="unavailable")

        self.assertEqual(
            format_upstream_error(response, "secret-key"),
            "No error message was provided",
        )


class _MockAsyncClient:
    def __init__(self, **options: object) -> None:
        self.options = options
        self.url: str | None = None
        self.request_body: dict[str, object] | None = None

    async def __aenter__(self) -> _MockAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> httpx.Response:
        self.url = url
        self.request_body = json
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"clauses":[{"id":"IND-21-2",'
                                '"applicability":"The facts may engage liberty."}]}'
                            )
                        }
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )


class GroqProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_groq_chat_completion_and_returns_graph_text(self) -> None:
        candidate = ArticleContext(
            id="IND-21",
            document="Constitution of India",
            reference="Article 21",
            title="Protection of liberty",
            text="Full article text.",
            score=0.9,
            subclauses=[
                SubClauseContext(id="IND-21-2", text="(2) No person shall be detained.")
            ],
            rights=["life and personal liberty"],
        )
        client = _MockAsyncClient()
        agent = LLMClauseAgent(
            provider="groq",
            api_key="test-groq-key",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            timeout_seconds=10,
        )

        with patch(
            "human_rights_graphrag.agent.httpx.AsyncClient", return_value=client
        ):
            result = await agent.analyze(
                "A person was detained.", ["detained"], [candidate]
            )

        self.assertEqual(
            client.url, "https://api.groq.com/openai/v1/chat/completions"
        )
        self.assertIsNotNone(client.request_body)
        assert client.request_body is not None
        self.assertEqual(client.request_body["model"], "openai/gpt-oss-120b")
        self.assertIn("messages", client.request_body)
        self.assertEqual(
            client.request_body["response_format"], {"type": "json_object"}
        )
        self.assertNotIn("store", client.request_body)
        self.assertEqual(result.clauses[0].id, "IND-21-2")
        self.assertEqual(result.clauses[0].text, "(2) No person shall be detained.")

    async def test_can_disable_tls_verification_for_local_prototype(self) -> None:
        client = _MockAsyncClient()
        agent = LLMClauseAgent(
            provider="groq",
            api_key="test-groq-key",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            timeout_seconds=10,
            verify_tls=False,
        )
        candidate = ArticleContext(
            id="IND-21",
            document="Constitution of India",
            reference="Article 21",
            title="Protection of liberty",
            text="Full article text.",
            score=0.9,
            subclauses=[
                SubClauseContext(id="IND-21-2", text="(2) No person shall be detained.")
            ],
            rights=["life and personal liberty"],
        )

        with patch(
            "human_rights_graphrag.agent.httpx.AsyncClient", return_value=client
        ) as async_client:
            await agent.analyze("A person was detained.", ["detained"], [candidate])

        async_client.assert_called_once_with(timeout=10, verify=False)


if __name__ == "__main__":
    unittest.main()

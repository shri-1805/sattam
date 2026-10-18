"""LLM-backed clause selection and applicability explanations."""

import json
import logging
import ssl
from pathlib import Path
from typing import Any

import certifi
import httpx

from human_rights_graphrag.models import (
    AnalysisResponse,
    ArticleContext,
    ClauseResponse,
    ModelSelection,
)


class LLMError(RuntimeError):
    """Raised when the configured language-model API cannot return an answer."""


logger = logging.getLogger(__name__)


def format_upstream_error(response: httpx.Response, api_key: str) -> str:
    """Extract a concise, credential-redacted message from an API error."""
    try:
        body = response.json()
    except ValueError:
        body = {}

    error = body.get("error", body) if isinstance(body, dict) else {}
    if isinstance(error, str):
        message = error
    elif isinstance(error, dict):
        message = str(
            error.get("message")
            or error.get("detail")
            or error.get("code")
            or "No error message was provided"
        )
        code = error.get("code")
        if code and code not in message:
            message = f"{code}: {message}"
    else:
        message = "No error message was provided"

    message = message.replace(api_key, "[redacted]").strip()
    if len(message) > 600:
        message = f"{message[:597]}..."
    return message


def build_ssl_context(ca_bundle: Path | None = None) -> ssl.SSLContext:
    """Create a verifying TLS context, optionally trusting an additional CA."""
    context = ssl.create_default_context(cafile=certifi.where())
    if ca_bundle is not None:
        try:
            context.load_verify_locations(cafile=str(ca_bundle))
        except (OSError, ssl.SSLError) as exc:
            raise ValueError(
                f"Could not load the configured LLM_CA_BUNDLE file {ca_bundle}: {exc}"
            ) from exc
    return context


def build_clause_responses(
    candidates: list[ArticleContext], selection: ModelSelection
) -> list[ClauseResponse]:
    """Map selected IDs back to exact graph text, rejecting invented IDs."""
    by_id: dict[str, tuple[ArticleContext, str]] = {}
    for candidate in candidates:
        options = (
            [(subclause.id, subclause.text) for subclause in candidate.subclauses]
            or [(candidate.id, candidate.text)]
        )
        by_id.update(
            {
                clause_id: (candidate, clause_text)
                for clause_id, clause_text in options
            }
        )

    chosen: list[ClauseResponse] = []
    seen: set[str] = set()
    for selected in selection.clauses:
        match = by_id.get(selected.id)
        if match is None:
            raise LLMError(
                f"The model selected clause {selected.id!r} that was not retrieved"
            )
        if selected.id in seen:
            continue
        seen.add(selected.id)
        candidate, text = match
        chosen.append(
            ClauseResponse(
                id=selected.id,
                article_id=candidate.id,
                document=candidate.document,
                reference=candidate.reference,
                title=candidate.title,
                text=text,
                applicability=selected.applicability,
                rights=candidate.rights,
            )
        )
    return chosen


class LLMClauseAgent:
    """Use Groq or xAI to select retrieved clauses and explain their relevance."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: float,
        ca_bundle: Path | None = None,
        verify_tls: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError(f"{provider.upper()}_API_KEY must be configured")
        if provider not in {"groq", "xai"}:
            raise ValueError(f"Unsupported language-model provider: {provider}")
        self.api_key = api_key
        self.provider = provider
        endpoint_suffix = "/chat/completions" if provider == "groq" else "/responses"
        self.endpoint = f"{base_url.rstrip('/')}{endpoint_suffix}"
        self.model = model
        self.timeout_seconds = timeout_seconds
        if ca_bundle is not None and not verify_tls:
            logger.warning(
                "LLM_CA_BUNDLE is ignored because TLS verification is disabled"
            )
        self.verify_tls = verify_tls
        self._ssl_context = (
            build_ssl_context(ca_bundle) if verify_tls and ca_bundle else None
        )
        if not verify_tls:
            logger.warning(
                "TLS certificate verification is disabled for %s; "
                "use this only for local development",
                self.provider.upper(),
            )

    async def analyze(
        self,
        scenario: str,
        concepts: list[str],
        candidates: list[ArticleContext],
    ) -> AnalysisResponse:
        if not candidates:
            return AnalysisResponse(scenario=scenario, clauses=[])

        payload = {
            "scenario": scenario,
            "key_concepts": concepts,
            "retrieved_clauses": [
                {
                    "id": item.id,
                    "document": item.document,
                    "reference": item.reference,
                    "title": item.title,
                    "clauses": [
                        {"id": sub.id, "text": sub.text}
                        for sub in item.subclauses
                    ]
                    or [{"id": item.id, "text": item.text}],
                    "rights": item.rights,
                }
                for item in candidates
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal-information clause selector. Assess only the "
                    "provided scenario and retrieved fundamental-rights clauses. "
                    "Return only clauses that directly apply; do not add outside "
                    "law, invent clause IDs, or rewrite legal text. For each "
                    "selected clause, provide a concise, cautious explanation "
                    "grounded in the scenario and that clause. If none applies, "
                    'return {"clauses": []}. Output one JSON object with this '
                    'shape: {"clauses":[{"id":"exact retrieved id",'
                    '"applicability":"short explanation"}]}.'
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        if self.provider == "groq":
            request_body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_completion_tokens": 1800,
                "response_format": {"type": "json_object"},
            }
        else:
            request_body = {
                "model": self.model,
                "input": messages,
                "store": False,
                "temperature": 0,
                "max_output_tokens": 1800,
                "text": {"format": {"type": "json_object"}},
            }
        try:
            client_options: dict[str, Any] = {"timeout": self.timeout_seconds}
            if self._ssl_context is not None:
                client_options["verify"] = self._ssl_context
            elif not self.verify_tls:
                client_options["verify"] = False
            async with httpx.AsyncClient(**client_options) as client:
                response = await client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request_body,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = format_upstream_error(exc.response, self.api_key)
            raise LLMError(
                f"{self.provider.upper()} API rejected the request (HTTP "
                f"{exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.provider.upper()} API request failed: {exc}") from exc
        try:
            body = response.json()
            if self.provider == "groq":
                content = body["choices"][0]["message"]["content"]
            else:
                content = "".join(
                    part["text"]
                    for item in body["output"]
                    if item.get("type") == "message"
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                )
            if not isinstance(content, str):
                raise ValueError("Response content was not text")
            if not content:
                raise ValueError("Model response contained no output text")
            selection = ModelSelection.model_validate_json(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(
                f"{self.provider.upper()} returned an invalid clause-selection response"
            ) from exc

        return AnalysisResponse(
            scenario=scenario,
            clauses=build_clause_responses(candidates, selection),
        )

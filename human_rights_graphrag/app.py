"""FastAPI and command-line entry points for Human Rights GraphRAG."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from human_rights_graphrag.agent import LLMClauseAgent, LLMError
from human_rights_graphrag.config import Settings
from human_rights_graphrag.db import DatabaseUnavailableError, FalkorGraph
from human_rights_graphrag.embeddings import LocalEmbedder
from human_rights_graphrag.ingest import ingest_corpus
from human_rights_graphrag.models import AnalysisResponse, ScenarioRequest
from human_rights_graphrag.retriever import ClauseRetriever


class HealthResponse(BaseModel):
    """Service and graph health status."""

    status: str
    database: str


def _make_graph(settings: Settings) -> FalkorGraph:
    return FalkorGraph(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        graph_name=settings.falkordb_graph,
        password=settings.falkordb_password,
    )


def _make_agent(settings: Settings) -> LLMClauseAgent | None:
    if not settings.llm_api_key:
        return None
    return LLMClauseAgent(
        provider=settings.ai_provider,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        ca_bundle=settings.llm_ca_bundle,
        verify_tls=settings.llm_verify_tls,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    graph = _make_graph(settings)
    embedder = LocalEmbedder(settings.embedding_model, token=settings.hf_token)
    app.state.settings = settings
    app.state.graph = graph
    app.state.retriever = ClauseRetriever(graph, embedder, settings.retrieval_limit)
    app.state.agent = _make_agent(settings)
    yield


def create_app() -> FastAPI:
    """Construct the HTTP API without opening a database connection."""
    api = FastAPI(
        title="Human Rights GraphRAG",
        description="Pinpointed retrieval of fundamental human-rights provisions.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @api.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        try:
            healthy = await run_in_threadpool(request.app.state.graph.health_check)
        except DatabaseUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not healthy:
            raise HTTPException(status_code=503, detail="FalkorDB health check failed")
        return HealthResponse(status="ok", database="connected")

    @api.post("/analyze", response_model=AnalysisResponse)
    async def analyze(
        payload: ScenarioRequest, request: Request
    ) -> AnalysisResponse:
        if request.app.state.agent is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{request.app.state.settings.ai_provider.upper()} is not configured; "
                    f"set {request.app.state.settings.ai_provider.upper()}_API_KEY "
                    "to enable analysis"
                ),
            )
        try:
            concepts, candidates = await run_in_threadpool(
                request.app.state.retriever.retrieve, payload.scenario
            )
            return await request.app.state.agent.analyze(
                payload.scenario, concepts, candidates
            )
        except DatabaseUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return api


app = create_app()
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "web", html=True),
    name="web",
)


def main() -> None:
    """Run the API server or execute graph operations from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="start the FastAPI application")
    subparsers.add_parser("ingest", help="load legal rights into FalkorDB")
    subparsers.add_parser("health", help="check the FalkorDB connection")
    query_parser = subparsers.add_parser("query", help="analyze a scenario")
    query_parser.add_argument("scenario")
    args = parser.parse_args()
    settings = Settings()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("human_rights_graphrag.app:app", host="0.0.0.0", port=8000)
        return

    graph = _make_graph(settings)
    if args.command == "health":
        if not graph.health_check():
            raise SystemExit("FalkorDB health check failed")
        print("FalkorDB connected")
        return
    if args.command == "ingest":
        count = ingest_corpus(
            graph, LocalEmbedder(settings.embedding_model, token=settings.hf_token)
        )
        print(f"Ingested {count} articles/provisions")
        return
    if args.command == "query":
        if not settings.llm_api_key:
            raise SystemExit(
                f"Set {settings.ai_provider.upper()}_API_KEY to analyze scenarios"
            )
        retriever = ClauseRetriever(
            graph,
            LocalEmbedder(settings.embedding_model, token=settings.hf_token),
            settings.retrieval_limit,
        )
        concepts, candidates = retriever.retrieve(args.scenario)
        agent = _make_agent(settings)
        if agent is None:
            raise SystemExit(
                f"Set {settings.ai_provider.upper()}_API_KEY to analyze scenarios"
            )
        result = asyncio.run(agent.analyze(args.scenario, concepts, candidates))
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

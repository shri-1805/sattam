# Human Rights GraphRAG

A Python 3.10+ application that retrieves narrowly relevant fundamental-rights
provisions from FalkorDB and uses Groq or xAI to explain their applicability. It
ingests Fundamental Rights in Part III (Articles 12–35 and selected inserted
articles) from `indianconstitution`, plus selected UDHR and ICCPR provisions.
It is an informational aid, not legal advice.

## Requirements

- Python 3.10+
- Docker Compose
- A Groq API key (or an xAI API key) to use `/analyze`

The first embedding operation downloads the configured sentence-transformer
model and may take a few minutes. Embeddings run locally on CPU. Set `HF_TOKEN`
in `.env` if you need authenticated Hugging Face downloads or higher Hub limits.

## Run

```powershell
docker compose up -d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env
python -m human_rights_graphrag.app ingest
python -m human_rights_graphrag.app serve
```

Open `http://localhost:8000` for the chat interface. It shares the API origin,
so no separate frontend server or CORS configuration is needed. Interactive
API documentation is available at `/docs`.

```http
POST /analyze
Content-Type: application/json

{"scenario": "A peaceful protester was detained without being told the reason."}
```

The response includes only subclause IDs selected from retrieved graph
candidates (and their parent `article_id`), verbatim indexed text, and
applicability notes. The model cannot supply or rewrite the returned legal
text.

Other commands:

```powershell
python -m human_rights_graphrag.app health
python -m human_rights_graphrag.app query "A peaceful protester was detained without reason"
```

## Configuration

Copy `.env.example` to `.env`. `FALKORDB_HOST`, `FALKORDB_PORT`, and
`FALKORDB_GRAPH` configure the graph connection. Groq is the default AI
provider: set `GROQ_API_KEY` and optionally configure `GROQ_MODEL` and
`GROQ_BASE_URL`. The selected Groq model must be enabled for your account and
support JSON mode. When switching an existing `.env` from xAI, set
`AI_PROVIDER=groq`; an existing `XAI_API_KEY` will not be used. Alternatively,
set `AI_PROVIDER=xai` and use `XAI_API_KEY`,
`XAI_MODEL`, and `XAI_BASE_URL`. `EMBEDDING_MODEL` must remain the same between
ingestion and query: the FalkorDB vector index dimension is derived from that
model. `HF_TOKEN` is passed to the Hugging Face sentence-transformer loader.

If your network uses an HTTPS inspection proxy or private certificate
authority, export its trusted CA certificate chain in PEM format and set
`LLM_CA_BUNDLE` to that file's path in `.env` (for example,
`LLM_CA_BUNDLE=C:\certs\organization-ca-bundle.pem`). The application adds
that CA to the standard trusted roots while keeping TLS certificate and
hostname verification enabled. Obtain the CA certificate from your
organization's IT/security team. For a throwaway local prototype only, set
`LLM_VERIFY_TLS=false` to bypass certificate checks if your local network
prevents a verified connection. This disables protection against man-in-the-
middle attacks; do not use it on public networks or in production. TLS
verification remains enabled by default.

The selected treaty provisions are packaged in
`human_rights_graphrag/data/international_rights.json`; these are not a
comprehensive statement of either instrument. Verify current official texts
and jurisdictional status before relying on any result. The corpus is based on
the [UDHR](https://www.un.org/en/about-us/universal-declaration-of-human-rights)
and the
[ICCPR](https://www.ohchr.org/en/instruments-mechanisms/instruments/international-covenant-civil-and-political-rights).

## Graph schema and retrieval

- `(:Document {name})-[:HAS_ARTICLE]->(:Article {id, reference, title, text, embedding})`
- `(:Article)-[:HAS_SUBCLAUSE]->(:SubClause {id, text})`
- `(:Article)-[:PROTECTS]->(:Right {category})`
- A cosine vector index on `Article.embedding`

The retriever performs vector nearest-neighbor search and expands only
`HAS_SUBCLAUSE` and `PROTECTS` neighbors for the candidate articles. The
configured language model selects applicable candidate IDs; the API sources
legal text directly from the graph rather than from model-generated output.

## HTTP health

`GET /health` checks FalkorDB connectivity and returns HTTP 503 if unavailable.
`POST /analyze` accepts a scenario string and reports graph/LLM service failures
as HTTP 503/502 respectively. Upstream provider errors include the provider's
response detail, with API keys redacted.

## Tests

```powershell
python -m unittest discover -s tests -v
```

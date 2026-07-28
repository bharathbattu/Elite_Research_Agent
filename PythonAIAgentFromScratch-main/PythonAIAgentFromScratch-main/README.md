# Elite Research Assistant v2

An evidence-first research application that retrieves source documents, ranks evidence,
generates a structured report through OpenRouter, validates inline citations, persists history,
and exports Markdown, text, and PDF.

## Why v2 is stronger

- Deterministic research pipeline instead of an unconstrained file-writing agent
- Actual document extraction rather than synthesis from search snippets alone
- Typed Pydantic data from retrieval through API responses
- Inline source IDs (`[S1]`) checked against retrieved evidence
- SSRF-resistant public URL fetching and bounded document sizes
- SQLite report history with parameterized queries
- Streamlit UI, CLI, and FastAPI interfaces sharing one service
- Live OpenRouter web search with current-date grounding and source annotations
- Configurable current web, news, academic, and background research modes
- Freshness, language, region, allow-domain, and exclude-domain controls
- Search caching, source metadata, credibility/freshness ranking, and retrieval metrics
- Public-URL document extraction with a guarded OpenRouter web-fetch fallback
- Offline tests, CI, Docker, health endpoint, and optional API-key protection

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -e ".[dev]"
copy sample.env .env
```

Edit `.env` and add a valid `OPENROUTER_API_KEY`. The default model is
`nvidia/nemotron-3-ultra-550b-a55b:free`. That key enables both report generation and
live OpenRouter web search. Google Custom Search is optional and can be used as an
additional provider; Wikipedia remains the reference-source fallback.

## Run

Streamlit:

```powershell
# One-command Windows launcher (creates/repairs .venv when needed):
.\start.ps1

# Or launch directly:
# Works without activating the virtual environment:
.\.venv\Scripts\python.exe -m streamlit run app.py
```

CLI:

```bash
python main.py "How is AI changing early cancer detection?"
```

API:

```bash
uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/docs`. If `APP_API_KEY` is configured, pass it in the
`X-API-Key` header for `/v1/*` endpoints.

## Test

```bash
ruff check .
pytest --cov=elite_research --cov-report=term-missing
```

Tests use fake retrieval and synthesis components, so they do not consume API credits.

## Architecture

1. Expand the question into targeted searches.
2. Search the current web through OpenRouter, plus Google (when configured) and Wikipedia.
3. Cache search results and convert provider annotations into application-owned evidence.
4. Fetch public documents with URL and size controls; use a one-request server-fetch fallback
   for thin or inaccessible non-search results.
5. Deduplicate and rank evidence by relevance, credibility, freshness, and depth.
6. Ask the configured OpenRouter model to synthesize only from numbered evidence records.
7. Reject unknown citations and flag uncited report fields.
8. Persist the validated report, retrieval metrics, and user-controlled exports.

## Live internet research

Open **Internet research settings** in the sidebar:

- **Current web** — current facts, releases, prices, schedules, or changing information
- **News** — recent events and official announcements
- **Academic** — studies, reviews, and research reports
- **Background** — stable explainers and primary-source context

Use freshness to prefer the last day, week, month, or year. Domain controls accept comma-separated
host names such as `who.int, cdc.gov`; do not include `https://` or page paths.

Search results are cached locally in `data/search-cache.db` for
`WEB_SEARCH_CACHE_MINUTES` (30 minutes by default). Set `WEB_SEARCH_ENABLED=false` to disable live
search or `WEB_FETCH_FALLBACK_ENABLED=false` to disable server-side fetch fallback.
`WEB_FETCH_MAX_FALLBACKS` caps deeper fetches per report (2 by default) to control latency
and model-token usage.

## Production notes

- Rotate any API key that was present in an earlier copy of `sample.env`.
- Put the API behind HTTPS and configure `APP_API_KEY` or an identity-aware proxy.
- SQLite is appropriate for a single instance. Use PostgreSQL and a job queue for
  multi-instance or high-volume deployments.
- Free OpenRouter endpoints may log prompts for provider improvement. Do not submit confidential,
  personal, or regulated information unless the selected provider policy permits it.
- Web content is untrusted evidence. The synthesizer prompt treats it as data, not
  instructions, and the model has no mutation tools.
- OpenRouter server tools are beta and may change. Web-search and model token usage may incur
  OpenRouter charges; the configured `openrouter` web-fetch engine is free apart from model tokens.

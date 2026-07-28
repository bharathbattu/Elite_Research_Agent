# Elite Research Assistant v2

An evidence-first AI research application that searches the current web, retrieves and ranks
source material, generates structured research reports, validates inline citations, preserves
research history, and exports results in multiple formats.

The v2 rebuild replaces the original unconstrained LangChain/Mistral agent with a deterministic,
typed research pipeline powered by OpenRouter, Streamlit, FastAPI, Pydantic, and SQLite.

> The application source is located in
> `PythonAIAgentFromScratch-main/PythonAIAgentFromScratch-main`.

## Table of contents

- [Highlights](#highlights)
- [Technology stack](#technology-stack)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
- [Environment variables](#environment-variables)
- [Using the application](#using-the-application)
- [FastAPI interface](#fastapi-interface)
- [Command-line interface](#command-line-interface)
- [Testing and quality checks](#testing-and-quality-checks)
- [Docker](#docker)
- [Security model](#security-model)
- [Data and privacy](#data-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Production recommendations](#production-recommendations)
- [License](#license)

## Highlights

- **Live internet research** through OpenRouter Web Search with current-date grounding
- **Four research modes**: current web, news, academic, and background
- **Freshness controls** for the last day, week, month, year, or unrestricted history
- **Domain controls** to allow or exclude specific websites across every search provider
- **Multiple evidence providers**: OpenRouter Web Search, optional Google Custom Search, and
  Wikipedia reference material
- **Evidence extraction** from public HTML and text documents
- **Guarded deep-fetch fallback** through OpenRouter Web Fetch when a normal fetch is too thin
- **Source ranking** based on relevance, credibility, freshness, and document depth
- **Traceable citations** using stable source identifiers such as `[S1]` and `[S2]`
- **Citation validation** that rejects unknown source identifiers before saving a report
- **Transparent retrieval metrics** showing raw results, unique URLs, selected evidence, provider
  counts, and fallback usage
- **Modern Streamlit interface** inspired by calm, document-focused research tools
- **Research archive** persisted locally in SQLite
- **Markdown, text, and PDF exports**
- **Three interfaces** sharing one pipeline: Streamlit, FastAPI, and CLI
- **Automated tests, Ruff linting, coverage enforcement, Docker, and health checks**

## Technology stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11+ |
| Web UI | Streamlit |
| API | FastAPI and Uvicorn |
| Model gateway | OpenRouter |
| Default model | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| Live search | OpenRouter Web Search |
| Deep-fetch fallback | OpenRouter Web Fetch |
| Optional search | Google Custom Search |
| Reference source | Wikipedia API |
| Validation | Pydantic v2 |
| HTTP and extraction | HTTPX and Beautiful Soup |
| Persistence and cache | SQLite |
| PDF export | ReportLab |
| Testing | Pytest and pytest-cov |
| Code quality | Ruff |
| Containerization | Docker |

## How it works

```text
Research question
      |
      v
Mode-aware query planning
      |
      v
OpenRouter Web Search + optional Google + Wikipedia
      |
      v
Domain filtering -> normalization -> deduplication
      |
      v
Public document extraction -> guarded fetch fallback
      |
      v
Relevance + credibility + freshness + depth ranking
      |
      v
Evidence-only model synthesis with [S1] citations
      |
      v
Citation validation -> SQLite archive -> exports
```

The language model does not browse independently or write arbitrary files. Search, extraction,
ranking, synthesis, validation, persistence, and export are separate application-owned stages.
This makes failures easier to diagnose and reports easier to audit.

## Project structure

```text
Elite_Research_Agent/
├── README.md
└── PythonAIAgentFromScratch-main/
    └── PythonAIAgentFromScratch-main/
        ├── app.py                  # Streamlit user interface
        ├── api.py                  # FastAPI service
        ├── main.py                 # Command-line interface
        ├── start.ps1               # One-command Windows launcher
        ├── sample.env              # Safe environment template
        ├── pyproject.toml          # Package and tooling configuration
        ├── requirements.txt        # Runtime and development dependencies
        ├── Dockerfile
        ├── elite_research/
        │   ├── config.py           # Typed environment configuration
        │   ├── models.py           # Request, evidence, metrics, and report schemas
        │   ├── retrieval.py        # Search, cache, extraction, filtering, and ranking
        │   ├── synthesis.py        # Evidence-grounded OpenRouter synthesis
        │   ├── pipeline.py         # End-to-end research orchestration
        │   ├── storage.py          # SQLite report repository
        │   ├── exporters.py        # Markdown, text, and PDF output
        │   └── errors.py           # User-safe domain errors
        ├── tests/
        └── docs/elite-research-v2/
```

## Prerequisites

- Python 3.11 or newer
- An [OpenRouter](https://openrouter.ai/) API key
- Internet access for live research
- Git, if cloning the repository
- Docker, only if using the container workflow

Google Custom Search credentials are optional. OpenRouter live search and Wikipedia work without
Google credentials.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/bharathbattu/Elite_Research_Agent.git
cd Elite_Research_Agent/PythonAIAgentFromScratch-main/PythonAIAgentFromScratch-main
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the application

Runtime dependencies:

```bash
python -m pip install -e .
```

Runtime plus testing tools:

```bash
python -m pip install -e ".[dev]"
```

### 4. Configure the environment

Windows:

```powershell
Copy-Item sample.env .env
```

macOS or Linux:

```bash
cp sample.env .env
```

Open `.env` and replace:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Never commit `.env`. It is already excluded by `.gitignore`.

### 5. Start Streamlit

Windows users can use the self-repairing launcher:

```powershell
.\start.ps1
```

Or launch directly on any platform:

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Environment variables

### Required

| Variable | Description |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter key used for live search and report synthesis |

### Model and service configuration

| Variable | Default | Description |
| --- | --- | --- |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter model identifier |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter-compatible API endpoint |
| `OPENROUTER_APP_TITLE` | `Elite Research Assistant` | Application attribution title |
| `APP_API_KEY` | empty | Optional `X-API-Key` protection for `/v1/*` endpoints |
| `DATABASE_PATH` | `data/research.db` | SQLite report database |
| `MAX_SOURCES` | `10` | Default evidence limit, from 3 to 20 |
| `MAX_QUERY_LENGTH` | `500` | Maximum research question length |
| `MAX_DOCUMENT_CHARS` | `8000` | Maximum evidence characters retained per document |
| `REQUEST_TIMEOUT_SECONDS` | `20` | Standard HTTP timeout |
| `MODEL_TIMEOUT_SECONDS` | `240` | Model and server-tool timeout |

### Live search and retrieval

| Variable | Default | Description |
| --- | --- | --- |
| `WEB_SEARCH_ENABLED` | `true` | Enables OpenRouter live web search |
| `WEB_SEARCH_ENGINE` | `exa` | OpenRouter search engine |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Maximum results requested per search call |
| `WEB_SEARCH_CACHE_MINUTES` | `30` | Local search-cache lifetime |
| `WEB_FETCH_FALLBACK_ENABLED` | `true` | Enables deep-fetch fallback for thin sources |
| `WEB_FETCH_ENGINE` | `openrouter` | Server-side fetch engine |
| `WEB_FETCH_MAX_FALLBACKS` | `2` | Maximum deep-fetch calls per report |
| `GOOGLE_API_KEY` | empty | Optional Google Custom Search key |
| `GOOGLE_CSE_ID` | empty | Optional Google Programmable Search Engine ID |

## Using the application

1. Enter a focused research question.
2. Select the maximum number of evidence sources.
3. Open **Internet research settings**.
4. Choose a search mode and freshness window.
5. Optionally provide a language, region, allowed domains, or excluded domains.
6. Select **Begin research**.
7. Review the generated report, evidence ledger, methodology, and retrieval metrics.
8. Export the validated result as Markdown, text, or PDF.

### Research modes

| Mode | Best for |
| --- | --- |
| Current web | Software releases, changing facts, schedules, policies, and recent information |
| News | Recent events, announcements, and current developments |
| Academic | Peer-reviewed studies, reviews, and institutional research |
| Background | Stable explainers, historical context, and primary sources |

### Domain filters

Enter comma-separated host names without page paths:

```text
who.int, cdc.gov, nature.com
```

Allowed and excluded domains are enforced across all providers. Subdomains are supported, and
invalid host names are rejected before research starts.

## FastAPI interface

Start the API:

```bash
python -m uvicorn api:app --reload
```

Open:

- API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Create a research report

```bash
curl -X POST http://127.0.0.1:8000/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What changed in the latest stable Python release?",
    "max_sources": 6,
    "search_options": {
      "mode": "current_web",
      "freshness": "month",
      "language": "English",
      "region": "",
      "allowed_domains": ["python.org"],
      "excluded_domains": []
    }
  }'
```

If `APP_API_KEY` is configured, include:

```text
X-API-Key: your_api_key
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Runtime and provider configuration health |
| `POST` | `/v1/research` | Run and persist a research report |
| `GET` | `/v1/reports` | List archived reports |
| `GET` | `/v1/reports/{report_id}` | Retrieve one report |

## Command-line interface

Print a report as Markdown:

```bash
python main.py "How is AI changing early cancer detection?"
```

Limit evidence and save to a new file:

```bash
python main.py \
  "What evidence supports universal basic income?" \
  --max-sources 8 \
  --output ubi-report.md
```

The CLI intentionally refuses to overwrite an existing output file.

## Testing and quality checks

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run linting:

```bash
python -m ruff check .
```

Run tests with coverage:

```bash
python -m pytest --cov=elite_research --cov-report=term-missing
```

The test suite uses fake retrieval and synthesis components by default, so it does not consume
OpenRouter credits. CI targets Python 3.11, 3.12, and 3.13 and enforces at least 75% coverage.

## Docker

Run these commands from the application directory:

```bash
docker build -t elite-research-assistant .
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v elite-research-data:/app/data \
  elite-research-assistant
```

Open [http://localhost:8501](http://localhost:8501).

The image:

- uses Python 3.12 slim
- runs as a non-root user
- stores application data under `/app/data`
- exposes Streamlit on port 8501
- includes a Streamlit health check

## Security model

- Secrets are loaded from environment variables and excluded from Git.
- The API can be protected with constant-time `X-API-Key` verification.
- Direct document retrieval accepts only public HTTP and HTTPS destinations.
- Private, loopback, link-local, and non-public addresses are rejected.
- Redirects are checked before they are followed.
- Domain filters are normalized and validated.
- Documents, search results, queries, timeouts, and fallback calls are bounded.
- Retrieved web pages are treated as untrusted evidence, not application instructions.
- The synthesis model receives no filesystem, shell, database mutation, or deployment tools.
- Unknown citation identifiers cause report validation to fail.

## Data and privacy

- Reports are stored in `data/research.db`.
- Search results are cached in `data/search-cache.db`.
- Both files are excluded from Git.
- Export files are created only when the user requests them.
- Search queries and retrieved evidence are sent to configured external providers.
- Provider privacy, retention, and logging policies apply.

Do not submit confidential, regulated, personal, or proprietary information unless the selected
providers and your deployment controls are appropriate for that data.

OpenRouter server tools are currently beta. Their API behavior and pricing may change; review the
official [Web Search](https://openrouter.ai/docs/guides/features/server-tools/web-search) and
[Web Fetch](https://openrouter.ai/docs/guides/features/server-tools/web-fetch) documentation before
production deployment.

## Troubleshooting

### `streamlit` is not recognized

Use the virtual environment’s Python module launcher:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or use:

```powershell
.\start.ps1
```

### The application says OpenRouter is not configured

Verify that `.env` exists in the application directory and contains:

```dotenv
OPENROUTER_API_KEY=your_real_key
```

Restart Streamlit after changing `.env`.

### Live search returns too few sources

- Widen the freshness window.
- Remove or broaden allowed-domain restrictions.
- Use a more specific question.
- Confirm the OpenRouter account has available credits.
- Optionally configure Google Custom Search.

### A source could not be fetched

Some websites block automated retrieval or require JavaScript. The system keeps search excerpts
and can use a limited OpenRouter Web Fetch fallback. A report can still complete when enough other
evidence is available.

### Port 8501 is already in use

Start Streamlit on another port:

```bash
python -m streamlit run app.py --server.port 8502
```

### Reset local research data

Stop the application and move or remove the database files under `data/`. This permanently removes
local report history and cached searches, so back them up first if needed.

## Production recommendations

- Place Streamlit and FastAPI behind HTTPS and an authenticated reverse proxy.
- Configure `APP_API_KEY` or use an identity-aware gateway.
- Use explicit OpenRouter budgets and monitor provider costs.
- Retain the deep-fetch cap to prevent unexpected latency and token usage.
- Replace SQLite with PostgreSQL for multi-instance deployments.
- Move long-running research jobs to a queue for high traffic.
- Add centralized logs, metrics, error reporting, backups, and retention policies.
- Pin and regularly update dependencies after testing.
- Run the automated test suite before every deployment.

## License

This project is licensed under the MIT License. See the application
[`LICENSE`](PythonAIAgentFromScratch-main/PythonAIAgentFromScratch-main/LICENSE) file.

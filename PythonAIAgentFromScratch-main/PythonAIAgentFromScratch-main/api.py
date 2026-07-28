import hmac

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from elite_research.config import Settings
from elite_research.errors import ResearchError
from elite_research.models import ReportSummary, ResearchReport, ResearchRequest
from elite_research.pipeline import ResearchPipeline
from elite_research.storage import ReportRepository

settings = Settings()
repository = ReportRepository(settings.database_path)
app = FastAPI(
    title="Elite Research Assistant API",
    version="2.0.0",
    description="Evidence-first research API with traceable inline citations.",
)


def authorize(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.app_api_key:
        return
    expected = settings.app_api_key.get_secret_value()
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_configured": settings.model_enabled,
        "live_web_configured": settings.live_web_enabled,
        "google_search_configured": settings.google_search_enabled,
    }


@app.post(
    "/v1/research",
    response_model=ResearchReport,
    dependencies=[Depends(authorize)],
)
async def research(request: ResearchRequest) -> ResearchReport:
    try:
        pipeline = ResearchPipeline(settings, repository=repository)
        return await run_in_threadpool(
            pipeline.run,
            request.query,
            request.max_sources,
            None,
            request.search_options,
        )
    except ResearchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/v1/reports",
    response_model=list[ReportSummary],
    dependencies=[Depends(authorize)],
)
def list_reports(limit: int = 50) -> list[ReportSummary]:
    return repository.list(max(1, min(limit, 100)))


@app.get(
    "/v1/reports/{report_id}",
    response_model=ResearchReport,
    dependencies=[Depends(authorize)],
)
def get_report(report_id: str) -> ResearchReport:
    report = repository.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

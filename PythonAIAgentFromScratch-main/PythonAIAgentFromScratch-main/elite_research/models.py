import re
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    snippet: str = ""
    provider: str
    published_at: datetime | None = None
    author: str | None = None
    language: str | None = None
    source_type: str = "web"


class SourceRecord(BaseModel):
    id: str
    title: str
    url: HttpUrl
    snippet: str = ""
    content: str = ""
    provider: str
    publisher: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    author: str | None = None
    language: str | None = None
    source_type: str = "web"
    retrieval_method: str = "direct"
    freshness_score: float = Field(default=0, ge=0, le=1)
    credibility_score: float = Field(default=0, ge=0, le=1)
    quality_score: float = Field(default=0, ge=0, le=1)


class SearchOptions(BaseModel):
    mode: Literal["current_web", "news", "academic", "background"] = "current_web"
    freshness: Literal["day", "week", "month", "year", "any"] = "month"
    language: str = Field(default="English", max_length=40)
    region: str = Field(default="", max_length=80)
    allowed_domains: list[str] = Field(default_factory=list, max_length=20)
    excluded_domains: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("allowed_domains", "excluded_domains")
    @classmethod
    def normalize_domains(cls, values: list[str]) -> list[str]:
        domains = []
        for value in values:
            domain = (
                value.strip()
                .lower()
                .removeprefix("https://")
                .removeprefix("http://")
                .split("/", 1)[0]
                .split(":", 1)[0]
                .strip(".")
            )
            if domain and re.fullmatch(
                r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}",
                domain,
            ):
                if domain not in domains:
                    domains.append(domain)
            elif domain:
                raise ValueError(f"Invalid domain: {value}")
        return domains


class RetrievalMetrics(BaseModel):
    query_count: int = 0
    raw_results: int = 0
    unique_results: int = 0
    fetched_sources: int = 0
    selected_sources: int = 0
    fallback_fetches: int = 0
    provider_counts: dict[str, int] = Field(default_factory=dict)
    provider_errors: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    heading: str = Field(min_length=2, max_length=120)
    content: str = Field(min_length=20)


class GeneratedResearch(BaseModel):
    topic: str = Field(min_length=5, max_length=200)
    abstract: str = Field(min_length=80)
    sections: list[ReportSection] = Field(min_length=3)
    key_insights: list[str] = Field(min_length=3, max_length=8)


class ResearchReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    topic: str
    abstract: str
    sections: list[ReportSection]
    sources: list[SourceRecord]
    key_insights: list[str]
    methodology: list[str]
    model: str
    research_mode: str = "background"
    information_current_at: datetime = Field(default_factory=utc_now)
    retrieval_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    created_at: datetime = Field(default_factory=utc_now)
    status: Literal["completed"] = "completed"
    warnings: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Research query cannot be blank")
        return value


class ReportSummary(BaseModel):
    id: str
    query: str
    topic: str
    created_at: datetime
    model: str


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    max_sources: int | None = Field(default=None, ge=3, le=20)
    search_options: SearchOptions = Field(default_factory=SearchOptions)

from pathlib import Path

import pytest

from elite_research.config import Settings
from elite_research.errors import ResearchQualityError
from elite_research.models import (
    GeneratedResearch,
    ReportSection,
    SourceRecord,
)
from elite_research.pipeline import ResearchPipeline, validate_citations
from elite_research.storage import ReportRepository


def source(identifier: str, title: str) -> SourceRecord:
    return SourceRecord(
        id=identifier,
        title=title,
        url=f"https://example.com/{identifier.lower()}",
        snippet=f"Evidence summary for {title}",
        content=f"Long-form evidence about {title}. " * 20,
        provider="test",
        quality_score=0.8,
    )


class FakeRetriever:
    def retrieve(self, query: str, max_sources: int, options=None) -> list[SourceRecord]:
        return [source("S1", "Clinical study"), source("S2", "Systematic review")]


class FakeSynthesizer:
    model_name = "fake-model"

    def synthesize(self, query: str, sources: list[SourceRecord]) -> GeneratedResearch:
        return GeneratedResearch(
            topic="Evidence-Based Test Research Report",
            abstract=(
                "The available evidence provides a testable synthesis of the research "
                "question and identifies consistent findings across two independent sources "
                "[S1][S2]. This abstract is intentionally detailed enough for schema validation."
            ),
            sections=[
                ReportSection(
                    heading="Background",
                    content="The background is established by the clinical evidence [S1].",
                ),
                ReportSection(
                    heading="Current evidence",
                    content="The systematic review supports the central result [S2].",
                ),
                ReportSection(
                    heading="Limitations",
                    content="The source set is small, so generalization is limited [S1][S2].",
                ),
                ReportSection(
                    heading="Outlook",
                    content="Future studies should validate the observed result [S2].",
                ),
            ],
            key_insights=[
                "Evidence supports the main conclusion [S1].",
                "Independent synthesis provides corroboration [S2].",
                "More diverse evidence would improve confidence [S1][S2].",
            ],
        )


def test_pipeline_runs_and_persists_report(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "reports.db")
    repository = ReportRepository(settings.database_path)
    pipeline = ResearchPipeline(
        settings,
        retriever=FakeRetriever(),
        synthesizer=FakeSynthesizer(),
        repository=repository,
    )

    report = pipeline.run("Does the test pipeline work?", max_sources=5)

    assert report.model == "fake-model"
    assert len(report.sources) == 2
    assert repository.get(report.id) == report
    assert repository.list()[0].id == report.id


def test_unknown_citation_is_rejected() -> None:
    generated = FakeSynthesizer().synthesize("query", [])
    generated.sections[0].content += " Unsupported claim [S9]."

    with pytest.raises(ResearchQualityError, match="unknown sources"):
        validate_citations(generated, [source("S1", "One"), source("S2", "Two")])

from pathlib import Path

from elite_research.config import Settings
from elite_research.exporters import to_markdown, to_pdf, to_text
from elite_research.pipeline import ResearchPipeline
from elite_research.storage import ReportRepository

from .test_pipeline import FakeRetriever, FakeSynthesizer


def test_all_export_formats(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "reports.db")
    report = ResearchPipeline(
        settings,
        retriever=FakeRetriever(),
        synthesizer=FakeSynthesizer(),
        repository=ReportRepository(settings.database_path),
    ).run("Export this research")

    markdown = to_markdown(report)
    assert report.topic in markdown
    assert "[S1]" in markdown
    assert report.topic in to_text(report)
    assert to_pdf(report).startswith(b"%PDF")

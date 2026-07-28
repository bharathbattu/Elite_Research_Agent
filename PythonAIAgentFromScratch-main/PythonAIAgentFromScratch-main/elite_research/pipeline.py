import logging
import re
from collections.abc import Callable

from .config import Settings
from .errors import ResearchQualityError, RetrievalError
from .models import GeneratedResearch, ResearchReport, SearchOptions, SourceRecord, utc_now
from .retrieval import EvidenceRetriever
from .storage import ReportRepository
from .synthesis import OpenRouterSynthesizer, ResearchSynthesizer

ProgressCallback = Callable[[str, float], None]
logger = logging.getLogger(__name__)


class ResearchPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: EvidenceRetriever | None = None,
        synthesizer: ResearchSynthesizer | None = None,
        repository: ReportRepository | None = None,
    ):
        self.settings = settings
        self.retriever = retriever or EvidenceRetriever(settings)
        self.synthesizer = synthesizer or OpenRouterSynthesizer(settings)
        self.repository = repository or ReportRepository(settings.database_path)

    def run(
        self,
        query: str,
        max_sources: int | None = None,
        progress: ProgressCallback | None = None,
        search_options: SearchOptions | None = None,
    ) -> ResearchReport:
        query = self._validate_query(query)
        search_options = search_options or SearchOptions()
        source_limit = max_sources or self.settings.max_sources
        source_limit = max(3, min(source_limit, 20))
        notify = progress or (lambda _message, _value: None)

        notify("Planning targeted searches", 0.08)
        sources = self.retriever.retrieve(query, source_limit, search_options)
        if len(sources) < 2:
            raise RetrievalError(
                "Not enough usable evidence was retrieved. Configure Google Search or refine "
                "the query."
            )

        notify(f"Analyzing {len(sources)} evidence sources", 0.55)
        generated = self.synthesizer.synthesize(query, sources)

        notify("Validating claims and citations", 0.82)
        warnings = validate_citations(generated, sources)
        report = ResearchReport(
            query=query,
            topic=generated.topic,
            abstract=generated.abstract,
            sections=generated.sections,
            sources=sources,
            key_insights=generated.key_insights,
            methodology=[
                "Expanded the question into targeted evidence searches.",
                (
                    f"Searched live web and reference sources in "
                    f"{search_options.mode.replace('_', ' ')} mode."
                ),
                "Retrieved, normalized, and deduplicated evidence with source metadata.",
                "Ranked evidence using relevance, credibility, freshness, and document depth.",
                "Synthesized only from retrieved evidence using inline source identifiers.",
                "Validated citation identifiers and measured citation coverage before saving.",
            ],
            model=self.synthesizer.model_name,
            research_mode=search_options.mode,
            information_current_at=utc_now(),
            retrieval_metrics=getattr(
                self.retriever, "last_metrics", None
            ) or {},
            warnings=warnings,
        )
        self.repository.save(report)
        logger.info(
            "Research report completed report_id=%s mode=%s sources=%d warnings=%d",
            report.id,
            report.research_mode,
            len(report.sources),
            len(report.warnings),
        )
        notify("Research report completed", 1.0)
        return report

    def _validate_query(self, query: str) -> str:
        query = re.sub(r"\s+", " ", query).strip()
        if len(query) < 3:
            raise ValueError("Research query must contain at least 3 characters.")
        if len(query) > self.settings.max_query_length:
            raise ValueError(
                f"Research query cannot exceed {self.settings.max_query_length} characters."
            )
        return query


def validate_citations(
    generated: GeneratedResearch,
    sources: list[SourceRecord],
) -> list[str]:
    valid_ids = {source.id for source in sources}
    fields = [
        ("abstract", generated.abstract),
        *[(f"section '{section.heading}'", section.content) for section in generated.sections],
        *[(f"insight {index}", text) for index, text in enumerate(generated.key_insights, 1)],
    ]
    unknown: set[str] = set()
    uncited: list[str] = []
    cited_ids: set[str] = set()
    for field_name, text in fields:
        citations = set(re.findall(r"\[(S\d+)\]", text))
        unknown.update(citations - valid_ids)
        cited_ids.update(citations & valid_ids)
        if not citations:
            uncited.append(field_name)

    if unknown:
        raise ResearchQualityError(
            f"The generated report cited unknown sources: {', '.join(sorted(unknown))}."
        )
    if len(cited_ids) < min(2, len(valid_ids)):
        raise ResearchQualityError("The report did not use enough retrieved evidence.")

    warnings = []
    if uncited:
        warnings.append("Some report fields have no inline citation: " + ", ".join(uncited))
    unused = sorted(valid_ids - cited_ids)
    if unused:
        warnings.append("Retrieved but not cited: " + ", ".join(unused))
    return warnings

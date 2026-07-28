import httpx
import pytest

from elite_research.config import Settings
from elite_research.models import SearchOptions, SearchResult, SourceRecord
from elite_research.retrieval import (
    DocumentFetcher,
    EvidenceRetriever,
    OpenRouterWebSearchProvider,
    SearchCache,
    WikipediaSearchProvider,
)


class FakeProvider:
    name = "fake"

    def search(self, query: str, limit: int, options=None) -> list[SearchResult]:
        return [
            SearchResult(
                title="Artificial intelligence evidence",
                url="https://example.com/evidence",
                snippet="Clinical artificial intelligence evidence",
                provider=self.name,
            ),
            SearchResult(
                title="Artificial intelligence evidence duplicate",
                url="https://example.com/evidence",
                snippet="Duplicate",
                provider=self.name,
            ),
        ]


class FakeFetcher:
    def fetch(self, result: SearchResult) -> SourceRecord:
        return SourceRecord(
            id="pending",
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            content="Artificial intelligence clinical evidence " * 100,
            provider=result.provider,
            publisher="example.edu",
        )


def test_retriever_deduplicates_ranks_and_assigns_ids() -> None:
    settings = Settings()
    retriever = EvidenceRetriever(
        settings,
        providers=[FakeProvider()],
        fetcher=FakeFetcher(),
    )

    sources = retriever.retrieve("artificial intelligence clinical evidence", 5)

    assert len(sources) == 1
    assert sources[0].id == "S1"
    assert sources[0].quality_score > 0.5
    assert len(retriever._query_variants("topic")) == 3


def test_html_extraction_removes_navigation_and_scripts() -> None:
    html = """
    <html><head><title>Study</title></head><body><nav>Menu</nav>
    <main><h1>Study</h1><p>Useful evidence.</p>
    <script>bad()</script></main><footer>Footer</footer></body></html>
    """
    text, metadata = DocumentFetcher._extract_html(html)

    assert text == "Study Useful evidence."
    assert metadata["title"] == "Study"


class FakeWikipediaResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "query": {
                "pages": {
                    "1": {
                        "title": "Research",
                        "fullurl": "https://en.wikipedia.org/wiki/Research",
                        "extract": "Research is systematic investigation.",
                    }
                }
            }
        }


def test_wikipedia_search_identifies_its_http_client(monkeypatch) -> None:
    captured: dict = {}

    def fake_get(*args, **kwargs):
        captured.update(kwargs)
        return FakeWikipediaResponse()

    monkeypatch.setattr(httpx, "get", fake_get)
    results = WikipediaSearchProvider(Settings()).search("research", 3)

    assert len(results) == 1
    assert captured["headers"]["User-Agent"].startswith("EliteResearchAssistant/")


def test_openrouter_annotations_become_owned_search_results() -> None:
    results = OpenRouterWebSearchProvider._annotations_to_results(
        [
            {
                "type": "url_citation",
                "url_citation": {
                    "title": "Official release",
                    "url": "https://example.gov/news/release",
                    "content": "The official release was published today.",
                },
            }
        ]
    )

    assert len(results) == 1
    assert results[0].provider == "OpenRouter Web Search"
    assert results[0].source_type == "official"
    assert "published today" in results[0].snippet


def test_search_cache_round_trip(tmp_path) -> None:
    settings = Settings(database_path=tmp_path / "reports.db")
    cache = SearchCache(settings)
    options = SearchOptions(mode="news", freshness="day")
    key = cache.key("provider", "current topic", 3, options)
    expected = [
        SearchResult(
            title="Current source",
            url="https://example.com/news",
            snippet="Current evidence",
            provider="provider",
        )
    ]

    assert cache.get(key) is None
    cache.set(key, expected)

    assert cache.get(key) == expected


def test_source_scoring_uses_freshness_and_credibility() -> None:
    source = SourceRecord(
        id="pending",
        title="Official current evidence",
        url="https://example.gov/news",
        snippet="Official current evidence about the topic",
        content="Current evidence " * 100,
        provider="OpenRouter Web Search",
        publisher="example.gov",
        source_type="official",
    )
    options = SearchOptions(mode="news", freshness="week")
    source.credibility_score = EvidenceRetriever._credibility(source)
    source.freshness_score = EvidenceRetriever._freshness(source, options)
    source.quality_score = EvidenceRetriever._score_source(
        "current evidence topic", source, options
    )

    assert source.credibility_score >= 0.9
    assert source.freshness_score > 0
    assert source.quality_score > 0.5


def test_web_fetch_fallback_is_used_for_thin_non_search_content(monkeypatch) -> None:
    settings = Settings(
        openrouter_api_key="test-key",
        web_fetch_fallback_enabled=True,
    )
    fetcher = DocumentFetcher(settings)
    monkeypatch.setattr(
        "elite_research.retrieval._public_url",
        lambda _url: True,
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("failed")),
    )
    monkeypatch.setattr(
        fetcher,
        "_openrouter_fetch",
        lambda _url: "Deep content recovered through the server fetch tool. " * 50,
    )
    result = SearchResult(
        title="Thin source",
        url="https://example.com/article",
        snippet="Short snippet",
        provider="Google Custom Search",
    )

    source = fetcher.fetch(result)

    assert source.retrieval_method == "openrouter_web_fetch"
    assert "Deep content recovered" in source.content


def test_retrieval_metrics_capture_provider_and_deduplication() -> None:
    settings = Settings(openrouter_api_key=None)
    retriever = EvidenceRetriever(
        settings,
        providers=[FakeProvider()],
        fetcher=FakeFetcher(),
    )

    retriever.retrieve("artificial intelligence clinical evidence", 5)

    assert retriever.last_metrics.raw_results == 6
    assert retriever.last_metrics.unique_results == 1
    assert retriever.last_metrics.selected_sources == 1
    assert retriever.last_metrics.provider_counts == {"fake": 6}


def test_search_options_normalize_and_validate_domains() -> None:
    options = SearchOptions(
        allowed_domains=["https://WWW.Example.com/research", "www.example.com"]
    )

    assert options.allowed_domains == ["www.example.com"]
    with pytest.raises(ValueError, match="Invalid domain"):
        SearchOptions(excluded_domains=["not a host"])


def test_domain_filters_apply_to_every_provider_and_include_subdomains() -> None:
    options = SearchOptions(
        allowed_domains=["python.org"],
        excluded_domains=["docs.python.org"],
    )

    assert EvidenceRetriever._domain_allowed(
        "https://www.python.org/downloads/", options
    )
    assert not EvidenceRetriever._domain_allowed(
        "https://docs.python.org/3/", options
    )
    assert not EvidenceRetriever._domain_allowed(
        "https://en.wikipedia.org/wiki/Python", options
    )

import json
from typing import Protocol

import httpx

from .config import Settings
from .errors import ConfigurationError, ModelError
from .models import GeneratedResearch, SourceRecord


class ResearchSynthesizer(Protocol):
    model_name: str

    def synthesize(self, query: str, sources: list[SourceRecord]) -> GeneratedResearch: ...


class OpenRouterSynthesizer:
    def __init__(self, settings: Settings):
        if not settings.openrouter_api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required. Copy sample.env to .env and add your key."
            )
        self.settings = settings
        self.model_name = settings.openrouter_model

    def synthesize(self, query: str, sources: list[SourceRecord]) -> GeneratedResearch:
        evidence = self._format_evidence(sources)
        system_prompt = """
You are an evidence-first research analyst. Write a balanced, precise report using ONLY the
evidence supplied by the user. Every factual claim must be followed by one or more source IDs
such as [S1] or [S1][S3]. Never invent authors, dates, statistics, publications, or URLs.
If evidence is limited or conflicting, state that limitation explicitly.

Return one JSON object with exactly these keys:
- topic: concise research title
- abstract: 120-220 word synthesis containing citations
- sections: array of at least 4 objects with "heading" and "content"; each content should be
  substantive and citation-rich
- key_insights: array of 3-6 evidence-backed conclusions, each containing citations

Do not return Markdown fences or a sources list. Source metadata is managed by the application.
""".strip()
        user_prompt = f"RESEARCH QUESTION:\n{query}\n\nEVIDENCE:\n{evidence}"
        payload: dict = {
            "model": self.model_name,
            "temperature": 0.15,
            "max_tokens": 6_000,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = self._post(payload)
        try:
            content = response["choices"][0]["message"]["content"]
            return GeneratedResearch.model_validate(self._parse_json(content))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ModelError(
                "OpenRouter returned an invalid structured research response."
            ) from exc

    def _post(self, payload: dict) -> dict:
        try:
            response = httpx.post(
                f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": (
                        f"Bearer {self.settings.openrouter_api_key.get_secret_value()}"
                    ),
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": self.settings.openrouter_app_title,
                },
                json=payload,
                timeout=self.settings.model_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = self._safe_error_detail(exc.response)
            suffix = f": {detail}" if detail else "."
            raise ModelError(f"OpenRouter request failed with HTTP {status}{suffix}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError("OpenRouter request failed or returned malformed JSON.") from exc

    @staticmethod
    def _parse_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```")
            content = content.removesuffix("```").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end <= start:
                raise
            return json.loads(content[start : end + 1])

    @staticmethod
    def _safe_error_detail(response: httpx.Response) -> str:
        try:
            message = response.json().get("error", {}).get("message", "")
            return str(message)[:300]
        except (ValueError, AttributeError):
            return ""

    @staticmethod
    def _format_evidence(sources: list[SourceRecord]) -> str:
        blocks = []
        for source in sources:
            content = source.content or source.snippet
            blocks.append(
                "\n".join(
                    [
                        f"[{source.id}] {source.title}",
                        f"URL: {source.url}",
                        f"Publisher: {source.publisher or source.provider}",
                        f"Source type: {source.source_type}",
                        "Published: "
                        + (
                            source.published_at.isoformat()
                            if source.published_at
                            else "unknown"
                        ),
                        f"Retrieved: {source.retrieved_at.isoformat()}",
                        f"Evidence: {content[:6_000]}",
                    ]
                )
            )
        return "\n\n".join(blocks)

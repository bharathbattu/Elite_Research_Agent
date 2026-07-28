import json

import httpx
import pytest

from elite_research.config import Settings
from elite_research.errors import ModelError
from elite_research.synthesis import OpenRouterSynthesizer

from .test_pipeline import source


class FakeResponse:
    status_code = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def generated_payload() -> dict:
    return {
        "topic": "A valid evidence based topic",
        "abstract": (
            "This sufficiently detailed abstract summarizes the supplied evidence and its "
            "implications while remaining bounded by the records included in the prompt [S1]."
        ),
        "sections": [
            {"heading": "Background", "content": "Substantive background evidence [S1]."},
            {"heading": "Evidence", "content": "Substantive current evidence [S1]."},
            {"heading": "Limitations", "content": "Substantive limitations evidence [S1]."},
        ],
        "key_insights": [
            "First evidence-backed conclusion [S1].",
            "Second evidence-backed conclusion [S1].",
            "Third evidence-backed conclusion [S1].",
        ],
    }


def test_openrouter_synthesizer_parses_structured_response(monkeypatch) -> None:
    response = {
        "choices": [{"message": {"content": json.dumps(generated_payload())}}]
    }
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: FakeResponse(response))
    synthesizer = OpenRouterSynthesizer(Settings(openrouter_api_key="test-key"))

    result = synthesizer.synthesize("question", [source("S1", "Evidence")])

    assert result.topic == "A valid evidence based topic"
    assert "[S1]" in result.abstract


def test_openrouter_synthesizer_rejects_invalid_json(monkeypatch) -> None:
    response = {"choices": [{"message": {"content": "not json"}}]}
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: FakeResponse(response))
    synthesizer = OpenRouterSynthesizer(Settings(openrouter_api_key="test-key"))

    with pytest.raises(ModelError, match="invalid structured"):
        synthesizer.synthesize("question", [source("S1", "Evidence")])

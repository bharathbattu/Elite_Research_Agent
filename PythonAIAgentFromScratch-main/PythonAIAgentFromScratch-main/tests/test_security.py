from elite_research.retrieval import _public_url


def test_non_web_and_private_urls_are_blocked() -> None:
    assert not _public_url("file:///etc/passwd")
    assert not _public_url("http://127.0.0.1/private")
    assert not _public_url("http://localhost/admin")

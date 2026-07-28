# Elite Research v2 Research

## Problem

The legacy snapshot could not install, contained a committed secret, exposed filesystem writes
to an agent, synthesized from short snippets, and had no working UI or tests.

## Selected approach

Use a deterministic evidence-first pipeline:

1. Search multiple targeted query variants.
2. Retrieve bounded text from public URLs.
3. Deduplicate and rank evidence.
4. Assign application-owned source IDs.
5. Generate typed research from the evidence.
6. validate inline citations before persistence.

This approach was selected over a general autonomous agent because the workflow is known,
auditability matters, and deterministic stages are easier to secure and evaluate.

## Product interfaces

- Streamlit for the interactive local application
- FastAPI for integrations
- CLI for automation and scripting

All interfaces use the same pipeline and SQLite repository.

## Risks and mitigations

- Prompt injection in web content: the model receives evidence as untrusted data and has no tools.
- SSRF: only globally routable HTTP(S) destinations are fetched, including redirects.
- Hallucinated citations: source metadata is application-owned and IDs are validated.
- Dependency drift: a small dependency surface and compatible version ranges replace LangChain.
- Missing external credentials: health and UI diagnostics expose configuration status.

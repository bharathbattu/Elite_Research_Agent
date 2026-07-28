import argparse
import sys

from elite_research.config import Settings
from elite_research.errors import ResearchError
from elite_research.exporters import to_markdown
from elite_research.pipeline import ResearchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elite Research Assistant CLI")
    parser.add_argument("query", nargs="+", help="Research question")
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--output", help="Optional Markdown output path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    query = " ".join(args.query)
    try:
        report = ResearchPipeline(Settings()).run(
            query,
            max_sources=args.max_sources,
            progress=lambda message, _: print(f"• {message}", file=sys.stderr),
        )
    except (ResearchError, ValueError) as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1

    markdown = to_markdown(report)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as file:
            file.write(markdown)
        print(f"Saved {args.output}", file=sys.stderr)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

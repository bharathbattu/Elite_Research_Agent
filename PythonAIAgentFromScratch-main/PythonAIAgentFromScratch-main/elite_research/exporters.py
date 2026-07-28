import io
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from .models import ResearchReport


def to_markdown(report: ResearchReport) -> str:
    lines = [
        f"# {report.topic}",
        "",
        f"_Generated {report.created_at:%Y-%m-%d %H:%M UTC} with {report.model}_",
        "",
        "## Abstract",
        "",
        report.abstract,
    ]
    for section in report.sections:
        lines.extend(["", f"## {section.heading}", "", section.content])
    lines.extend(["", "## Key insights", ""])
    lines.extend(f"- {insight}" for insight in report.key_insights)
    lines.extend(["", "## Sources", ""])
    lines.extend(
        f"- [{source.id}] [{source.title}]({source.url}) — {source.publisher or source.provider}"
        for source in report.sources
    )
    lines.extend(["", "## Methodology", ""])
    lines.extend(f"- {item}" for item in report.methodology)
    if report.warnings:
        lines.extend(["", "## Quality warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines).strip() + "\n"


def to_text(report: ResearchReport) -> str:
    text = to_markdown(report)
    text = text.replace("# ", "").replace("## ", "")
    return text


def to_pdf(report: ResearchReport) -> bytes:
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=report.topic,
    )
    story = [
        Paragraph(escape(report.topic), styles["Title"]),
        Spacer(1, 12),
        Paragraph("Abstract", styles["Heading1"]),
        Paragraph(escape(report.abstract), styles["BodyText"]),
    ]
    for section in report.sections:
        story.extend(
            [
                Spacer(1, 10),
                Paragraph(escape(section.heading), styles["Heading1"]),
                Paragraph(escape(section.content).replace("\n", "<br/>"), styles["BodyText"]),
            ]
        )
    story.extend([PageBreak(), Paragraph("Key insights", styles["Heading1"])])
    story.extend(
        Paragraph(f"• {escape(insight)}", styles["BodyText"]) for insight in report.key_insights
    )
    story.append(Paragraph("Sources", styles["Heading1"]))
    story.extend(
        Paragraph(
            f"[{source.id}] {escape(source.title)} — {escape(str(source.url))}",
            styles["BodyText"],
        )
        for source in report.sources
    )
    document.build(story)
    return buffer.getvalue()

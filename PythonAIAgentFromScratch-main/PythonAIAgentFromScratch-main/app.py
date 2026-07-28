from html import escape

import streamlit as st

from elite_research.config import Settings
from elite_research.errors import ResearchError
from elite_research.exporters import to_markdown, to_pdf, to_text
from elite_research.models import ResearchReport, SearchOptions
from elite_research.pipeline import ResearchPipeline
from elite_research.storage import ReportRepository

st.set_page_config(
    page_title="Elite Research Assistant",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600&family=Poppins:wght@400;500;600&display=swap');

    :root {
        --ink: #141413;
        --ink-soft: #262624;
        --paper: #faf9f5;
        --paper-soft: #f0eee6;
        --accent: #d97757;
        --blue: #6a9bcc;
        --green: #788c5d;
        --mist: #77756e;
        --line: #e2e0d7;
        --line-dark: rgba(20, 20, 19, 0.12);
    }

    html, body, [class*="css"] {
        font-family: "Lora", Georgia, serif;
    }

    .stApp {
        color: var(--ink);
        background: var(--paper);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMain"] .block-container {
        max-width: 1040px;
        padding: 2.2rem 3rem 5rem;
    }

    [data-testid="stSidebar"] {
        background: var(--paper-soft);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.75rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: "Poppins", Arial, sans-serif;
    }

    h1, h2, h3 {
        color: var(--ink);
        font-family: "Poppins", Arial, sans-serif;
        letter-spacing: -0.025em;
    }

    h1 {font-size: clamp(2.3rem, 5vw, 4.4rem) !important; line-height: .98 !important;}
    h2 {font-size: 2rem !important;}
    h3 {font-size: 1.35rem !important;}

    p, label, .stCaption {color: #56544f;}

    code, .mono, [data-testid="stMetricLabel"] {
        font-family: "Poppins", Arial, sans-serif !important;
    }

    .editorial-hero {
        position: relative;
        overflow: hidden;
        min-height: 230px;
        padding: 3.6rem 1rem 2.8rem;
        margin-bottom: 1rem;
        text-align: center;
        border: 0;
        background: transparent;
        animation: reveal .65s ease-out both;
    }

    .editorial-hero::after {
        content: "";
        position: absolute;
        width: 7px;
        height: 7px;
        top: 2.1rem;
        left: calc(50% - 3px);
        background: var(--accent);
        border-radius: 50%;
    }

    .hero-index {
        color: var(--mist);
        font-family: "Poppins", Arial, sans-serif;
        font-size: .7rem;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 1.05rem auto 1.15rem;
        color: var(--ink);
        font-family: "Poppins", Arial, sans-serif;
        font-size: clamp(2.5rem, 6vw, 4.8rem);
        font-weight: 500;
        line-height: 1.04;
        letter-spacing: -.055em;
    }

    .hero-title em {
        color: var(--accent);
        font-family: "Lora", Georgia, serif;
        font-weight: 500;
    }

    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 650px;
        margin: 0 auto;
        color: #66645e;
        font-size: 1rem;
        line-height: 1.7;
    }

    .section-label {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: 1.4rem 0 .7rem;
        color: var(--accent);
        font-family: "Poppins", Arial, sans-serif;
        font-size: .7rem;
        letter-spacing: .15em;
        text-transform: uppercase;
    }

    .section-label::after {
        content: "";
        width: 38px;
        height: 1px;
        background: currentColor;
    }

    .system-mark {
        display: inline-flex;
        align-items: center;
        gap: .6rem;
        margin-bottom: 1.4rem;
        color: var(--ink);
        font-family: "Poppins", Arial, sans-serif;
        font-size: .73rem;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .system-mark::before {
        content: "";
        width: 18px;
        height: 18px;
        border: 0;
        border-radius: 45% 55% 55% 45%;
        background: var(--accent);
        transform: rotate(18deg);
    }

    .status-pill {
        display: flex;
        align-items: center;
        gap: .65rem;
        padding: .75rem .85rem;
        margin: .5rem 0;
        border: 1px solid var(--line);
        border-radius: 10px;
        color: #56544f;
        background: rgba(255,255,255,.42);
        font-size: .78rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        flex: 0 0 auto;
        border-radius: 50%;
        background: var(--green);
    }

    .status-dot.muted {
        background: var(--accent);
        box-shadow: none;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 16px !important;
        background: rgba(255,255,255,.56);
        box-shadow: 0 1px 2px rgba(20,20,19,.025);
    }

    .stTextArea textarea {
        min-height: 145px;
        padding: 1.15rem !important;
        border: 1px solid #d6d3ca !important;
        border-radius: 14px !important;
        color: var(--ink) !important;
        background: #fff !important;
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(217,119,87,.10) !important;
    }

    .stButton > button, .stDownloadButton > button {
        min-height: 2.85rem;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        color: var(--ink) !important;
        background: transparent !important;
        font-family: "Poppins", Arial, sans-serif;
        font-weight: 500;
        transition: transform .18s ease, border-color .18s ease, background .18s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        border-color: #bbb8ae !important;
        color: var(--ink) !important;
        background: #f3f1ea !important;
    }

    .stButton > button[kind="primary"] {
        border-color: var(--ink) !important;
        color: var(--paper) !important;
        background: var(--ink) !important;
        box-shadow: none;
    }

    .stButton > button[kind="primary"]:hover {
        color: var(--paper) !important;
        background: #302f2c !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        min-height: 2.45rem;
        justify-content: flex-start;
        border-color: transparent !important;
        color: #64615b !important;
        font-size: .78rem;
        font-weight: 500;
        text-align: left;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: none;
        border-color: var(--line) !important;
        color: var(--ink) !important;
        background: rgba(255,255,255,.65) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        height: 3.4rem;
        padding: 0;
        color: var(--mist);
        font-family: "Poppins", Arial, sans-serif;
        font-size: .74rem;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }

    [data-testid="stExpander"] {
        margin-bottom: .65rem;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,.48);
    }

    .report-rule {
        height: 1px;
        margin: 1.8rem 0;
        background: linear-gradient(90deg, var(--accent) 0 12%, var(--line) 12%);
    }

    .step-number {
        color: var(--accent);
        font-family: "Poppins", Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 400;
        line-height: 1;
    }

    .step-copy {
        color: #66645e;
        font-size: .9rem;
        line-height: 1.55;
    }

    [data-testid="stProgress"] > div > div {
        background-color: var(--accent) !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    *:focus-visible {
        outline: 2px solid var(--accent) !important;
        outline-offset: 3px;
    }

    @keyframes reveal {
        from {opacity: 0; transform: translateY(14px);}
        to {opacity: 1; transform: translateY(0);}
    }

    @media (max-width: 800px) {
        [data-testid="stMain"] .block-container {padding: 1rem 1rem 3rem;}
        .editorial-hero {min-height: 280px; padding: 3rem 1rem 1.5rem;}
        .editorial-hero::after {top: 1.5rem;}
        .hero-title {font-size: 2.9rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = (
    (
        "Clinical evidence",
        "How is artificial intelligence changing early cancer detection, and what evidence "
        "supports its clinical impact?",
    ),
    (
        "Technology outlook",
        "What are the established benefits and limitations of retrieval-augmented generation?",
    ),
    (
        "Policy analysis",
        "What evidence exists on the effectiveness and risks of universal basic income?",
    ),
)


@st.cache_resource
def runtime() -> tuple[Settings, ReportRepository]:
    settings = Settings()
    return settings, ReportRepository(settings.database_path)


def load_example(query: str) -> None:
    st.session_state.research_query = query


def render_report(report: ResearchReport) -> None:
    st.markdown(
        '<div class="section-label">Validated research dossier</div>',
        unsafe_allow_html=True,
    )
    st.title(report.topic)

    metric_columns = st.columns([1, 1, 1, 2])
    metric_columns[0].metric("Sources", len(report.sources))
    metric_columns[1].metric("Sections", len(report.sections))
    metric_columns[2].metric("Insights", len(report.key_insights))
    metric_columns[3].metric(
        "Information current at",
        report.information_current_at.strftime("%d %b %Y · %H:%M UTC"),
    )
    st.markdown('<div class="report-rule"></div>', unsafe_allow_html=True)

    report_tab, evidence_tab, method_tab, export_tab = st.tabs(
        ["01 / Report", "02 / Evidence", "03 / Method", "04 / Export"]
    )

    with report_tab:
        st.markdown("### Abstract")
        with st.container(border=True):
            st.markdown(report.abstract)

        for index, section in enumerate(report.sections, 1):
            st.markdown(
                f'<div class="section-label">{index:02d} / Analysis</div>',
                unsafe_allow_html=True,
            )
            st.subheader(section.heading)
            st.markdown(section.content)

        st.markdown('<div class="section-label">Decision layer</div>', unsafe_allow_html=True)
        st.subheader("Key insights")
        for index, insight in enumerate(report.key_insights, 1):
            with st.container(border=True):
                number_col, content_col = st.columns([0.7, 8])
                number_col.markdown(
                    f'<div class="step-number">{index:02d}</div>',
                    unsafe_allow_html=True,
                )
                content_col.markdown(insight)

    with evidence_tab:
        st.markdown("### Source ledger")
        st.caption(
            "Every source ID below maps directly to the inline citations used in the report."
        )
        for source in report.sources:
            with st.expander(f"[{source.id}]  {source.title}"):
                publisher_col, score_col = st.columns([3, 1])
                publisher_col.markdown(
                    f"**Publisher**  \n{source.publisher or source.provider}"
                )
                score_col.metric("Evidence score", f"{source.quality_score:.0%}")
                st.progress(source.quality_score)
                detail_columns = st.columns(4)
                detail_columns[0].caption(f"Type · {source.source_type.title()}")
                detail_columns[1].caption(
                    "Published · "
                    + (
                        source.published_at.strftime("%d %b %Y")
                        if source.published_at
                        else "Not supplied"
                    )
                )
                detail_columns[2].caption(
                    f"Freshness · {source.freshness_score:.0%}"
                )
                detail_columns[3].caption(
                    "Retrieved · "
                    + source.retrieval_method.replace("_", " ").title()
                )
                st.markdown(f"**Original source:** [{source.url}]({source.url})")
                st.markdown("**Retrieved evidence**")
                st.write((source.content or source.snippet)[:1_500])

    with method_tab:
        st.markdown("### Research protocol")
        st.caption("A transparent record of how this dossier was assembled and checked.")
        metrics = report.retrieval_metrics
        metric_columns = st.columns(4)
        metric_columns[0].metric("Search results", metrics.raw_results)
        metric_columns[1].metric("Unique URLs", metrics.unique_results)
        metric_columns[2].metric("Evidence used", metrics.selected_sources)
        metric_columns[3].metric("Fetch fallbacks", metrics.fallback_fetches)
        if metrics.provider_counts:
            providers = " · ".join(
                f"{provider}: {count}"
                for provider, count in sorted(metrics.provider_counts.items())
            )
            st.caption(f"Provider results · {providers}")
        if metrics.provider_errors:
            st.warning("Provider issues: " + ", ".join(metrics.provider_errors))
        for index, item in enumerate(report.methodology, 1):
            with st.container(border=True):
                number_col, content_col = st.columns([0.7, 8])
                number_col.markdown(
                    f'<div class="step-number">{index:02d}</div>',
                    unsafe_allow_html=True,
                )
                content_col.markdown(item)
        if report.warnings:
            st.warning("\n\n".join(report.warnings))
        else:
            st.success("Citation integrity checks passed without critical warnings.")

    with export_tab:
        st.markdown("### Take the dossier with you")
        st.caption("Exports preserve citations, evidence references, and report structure.")
        stem = f"research-{report.id[:8]}"
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "↓ Markdown",
            to_markdown(report),
            f"{stem}.md",
            "text/markdown",
            use_container_width=True,
        )
        col2.download_button(
            "↓ Plain text",
            to_text(report),
            f"{stem}.txt",
            "text/plain",
            use_container_width=True,
        )
        col3.download_button(
            "↓ PDF dossier",
            to_pdf(report),
            f"{stem}.pdf",
            "application/pdf",
            use_container_width=True,
        )


def _domains(value: str) -> list[str]:
    return [
        domain.strip().lower().removeprefix("https://").removeprefix("http://")
        for domain in value.split(",")
        if domain.strip()
    ]


def render_sidebar(
    settings: Settings,
    repository: ReportRepository,
) -> tuple[int, SearchOptions]:
    with st.sidebar:
        st.markdown('<div class="system-mark">ERA / Research OS</div>', unsafe_allow_html=True)
        st.markdown("### System")

        model_status = (
            f'<div class="status-pill"><span class="status-dot"></span>'
            f'{escape(settings.openrouter_model)}</div>'
            if settings.model_enabled
            else '<div class="status-pill"><span class="status-dot muted"></span>'
            "OpenRouter key missing</div>"
        )
        st.markdown(model_status, unsafe_allow_html=True)

        if settings.live_web_enabled:
            search_label = "Live web + Wikipedia"
        elif settings.google_search_enabled:
            search_label = "Google + Wikipedia"
        else:
            search_label = "Wikipedia evidence mode"
        st.markdown(
            f'<div class="status-pill"><span class="status-dot"></span>'
            f"{search_label}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Research depth")
        max_sources = st.slider(
            "Maximum evidence sources",
            min_value=3,
            max_value=20,
            value=settings.max_sources,
            help="More sources improve breadth but increase research time.",
        )
        with st.expander("Internet research settings", expanded=True):
            mode_label = st.selectbox(
                "Search mode",
                ["Current web", "News", "Academic", "Background"],
                help="Changes query planning and how freshness is weighted.",
            )
            freshness_label = st.selectbox(
                "Freshness",
                ["24 hours", "7 days", "30 days", "1 year", "Any time"],
                index=2,
            )
            language = st.text_input("Language", value="English")
            region = st.text_input(
                "Region",
                placeholder="Global, India, United States…",
            )
            allowed = st.text_input(
                "Only these domains",
                placeholder="who.int, nature.com",
                help="Optional comma-separated allowlist.",
            )
            excluded = st.text_input(
                "Exclude domains",
                placeholder="example.com",
                help="Optional comma-separated blocklist.",
            )

        st.markdown("---")
        history = repository.list(30)
        st.markdown(f"### Archive · {len(history):02d}")
        if not history:
            st.caption("Your completed dossiers will appear here.")
        for item in history:
            label = f"{item.created_at:%d %b}  ·  {item.topic[:38]}"
            if st.button(
                label,
                key=f"history-{item.id}",
                help=item.query,
                use_container_width=True,
            ):
                st.session_state.current_report_id = item.id

        st.caption("Evidence-first · Citation-checked · Locally archived")
    mode_map = {
        "Current web": "current_web",
        "News": "news",
        "Academic": "academic",
        "Background": "background",
    }
    freshness_map = {
        "24 hours": "day",
        "7 days": "week",
        "30 days": "month",
        "1 year": "year",
        "Any time": "any",
    }
    return max_sources, SearchOptions(
        mode=mode_map[mode_label],
        freshness=freshness_map[freshness_label],
        language=language.strip() or "English",
        region=region.strip(),
        allowed_domains=_domains(allowed),
        excluded_domains=_domains(excluded),
    )


def render_empty_state() -> None:
    st.markdown('<div class="section-label">How the system thinks</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    steps = (
        ("01", "Retrieve", "Expand the question and collect focused public evidence."),
        ("02", "Synthesize", "Ask Nemotron to reason only over numbered source records."),
        ("03", "Verify", "Reject unknown citations before saving the final dossier."),
    )
    for column, (number, title, copy) in zip(columns, steps, strict=True):
        with column, st.container(border=True):
            st.markdown(
                f'<div class="step-number">{number}</div>',
                unsafe_allow_html=True,
            )
            st.subheader(title)
            st.markdown(
                f'<div class="step-copy">{copy}</div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    settings, repository = runtime()
    max_sources, search_options = render_sidebar(settings, repository)

    st.markdown(
        """
        <section class="editorial-hero">
            <div class="hero-index">Elite Research Assistant</div>
            <div class="hero-title">What will we<br><em>understand today?</em></div>
            <div class="hero-copy">
                Explore a difficult question through public evidence, careful synthesis,
                and citations you can inspect for yourself.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Open a new inquiry</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("What do you need to understand?")
        st.caption(
            "Frame a focused question. Include a population, timeframe, region, or comparison "
            "when those details matter."
        )

        example_columns = st.columns(3)
        for column, (label, query) in zip(example_columns, EXAMPLES, strict=True):
            column.button(
                f"↗ {label}",
                key=f"example-{label}",
                on_click=load_example,
                args=(query,),
                use_container_width=True,
            )

        st.text_area(
            "Research question",
            key="research_query",
            placeholder=(
                "Example: How is artificial intelligence changing early cancer detection, "
                "and what evidence supports its clinical impact?"
            ),
            height=145,
            max_chars=settings.max_query_length,
            label_visibility="collapsed",
        )

        action_col, note_col = st.columns([1.4, 2.6], vertical_alignment="center")
        run_research = action_col.button(
            "Begin research  →",
            type="primary",
            disabled=not settings.model_enabled,
            use_container_width=True,
        )
        note_col.caption(
            f"Up to {max_sources} sources · Live {search_options.mode.replace('_', ' ')} "
            "· Citation validation enabled"
        )

    if run_research:
        progress_bar = st.progress(0, text="Opening the research protocol…")

        def update_progress(message: str, value: float) -> None:
            progress_bar.progress(value, text=message)

        try:
            pipeline = ResearchPipeline(settings, repository=repository)
            report = pipeline.run(
                st.session_state.research_query,
                max_sources=max_sources,
                progress=update_progress,
                search_options=search_options,
            )
            st.session_state.current_report_id = report.id
            progress_bar.empty()
            st.toast("Dossier complete and archived.", icon="✓")
        except (ResearchError, ValueError) as exc:
            progress_bar.empty()
            st.error(str(exc))
        except Exception:
            progress_bar.empty()
            st.error("Unexpected research failure. Check the server logs for details.")

    report_id = st.session_state.get("current_report_id")
    if report_id:
        report = repository.get(report_id)
        if report:
            st.markdown("<br>", unsafe_allow_html=True)
            render_report(report)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()

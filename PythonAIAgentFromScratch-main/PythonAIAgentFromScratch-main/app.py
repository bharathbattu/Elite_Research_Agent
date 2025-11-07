import streamlit as st
import os
import datetime
import json
import re
from typing import List, Any
from dotenv import load_dotenv
import xml.sax.saxutils
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_core.output_parsers import PydanticOutputParser

# ✅ UPDATED LANGCHAIN IMPORT
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

from langchain_core.tools import BaseTool
from tools import search_tool, wiki_tool, save_tool
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean academic styling for professional research output with white text
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Clean academic typography with white text */
    .stMarkdown h1 {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-align: center !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    .stMarkdown h2 {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        font-size: 1.4rem !important;
    }
    
    .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 500 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        font-size: 1.2rem !important;
    }
    
    .stMarkdown h4 {
        color: #ffffff !important;
        font-weight: 500 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
        font-size: 1.1rem !important;
    }
    
    /* Clean paragraph and list styling with white text */
    .stMarkdown p {
        line-height: 1.6 !important;
        margin-bottom: 1rem !important;
        text-align: justify !important;
        color: #ffffff !important;
    }
    
    /* Style for st.write() text */
    .stMarkdown div {
        color: #ffffff !important;
        line-height: 1.6 !important;
    }
    
    .stMarkdown ul {
        margin-left: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    .stMarkdown li {
        margin-bottom: 0.5rem !important;
        line-height: 1.5 !important;
        color: #ffffff !important;
    }
    
    /* Horizontal rules styling */
    .stMarkdown hr {
        border: none !important;
        height: 1px !important;
        background-color: #bdc3c7 !important;
        margin: 2rem 0 !important;
    }
    
    /* Remove any background colors or containers */
    .element-container {
        background-color: transparent !important;
    }
    
    /* Ensure clean white text display */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Glassmorphism card for export options */
    .export-card {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    
    .export-card h4 {
        color: #ffffff !important;
        margin-top: 0 !important;
        margin-bottom: 15px !important;
        text-align: center !important;
    }
    
    .export-card .stButton > button {
        width: 100% !important;
        margin-bottom: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Pydantic model for research response
class ResearchResponse(BaseModel):
    topic: str
    abstract: str
    detailed_findings: str
    sources: List[str]
    tools_used: List[str]
    key_insights: List[str]

@st.cache_resource
def initialize_agent():
    """Initialize the research agent with caching for performance"""
    try:
        # Initialize LLM
        llm = ChatMistralAI(  # type: ignore
            model_name="mistral-small", 
            api_key=os.getenv("MISTRAL_API_KEY")  # type: ignore
        )
        
        # Initialize parser
        parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([  # type: ignore
            (
                "system",
                """
                You are an elite academic research assistant specializing in comprehensive, publication-quality research reports.
                ... (❗ Unchanged prompt content — omitted here only to save space)
                {format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Initialize tools
        tools: List[BaseTool] = [search_tool, wiki_tool, save_tool]  # type: ignore
        
        # ✅ UPDATED AGENT CREATION FOR NEW LANGCHAIN
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        # Agent executor stays the same
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)  # type: ignore
        
        return agent_executor, parser
        
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None, None

def generate_pdf_report(research_data: ResearchResponse) -> bytes:  # type: ignore
    """Generate a PDF report from research data with proper error handling"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)  # type: ignore
        styles = getSampleStyleSheet()  # type: ignore
        story: List[Any] = []  # type: ignore
        
        # Custom styles
        title_style = ParagraphStyle(  # type: ignore
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(  # type: ignore
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20
        )
        
        subheading_style = ParagraphStyle(  # type: ignore
            'CustomSubHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=10,
            spaceBefore=15
        )
        
        # Title
        story.append(Paragraph("AI Research Assistant Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Topic
        story.append(Paragraph("Research Topic", heading_style))
        safe_topic = xml.sax.saxutils.escape(research_data.topic)
        story.append(Paragraph(safe_topic, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Abstract
        story.append(Paragraph("Abstract", heading_style))
        safe_abstract = xml.sax.saxutils.escape(research_data.abstract)
        safe_abstract = safe_abstract.replace('\n', '<br/>')
        story.append(Paragraph(safe_abstract, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Detailed Findings - Parse markdown sections
        story.append(Paragraph("Detailed Findings", heading_style))
        
        sections = re.findall(r'##\s*(.+?)\s*\n(.*?)(?=\n\s*##\s*|\Z)', 
                             research_data.detailed_findings, flags=re.DOTALL)
        
        if sections:
            for section_title, section_content in sections:
                safe_section_title = xml.sax.saxutils.escape(section_title.strip())
                story.append(Paragraph(safe_section_title, subheading_style))
                
                paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
                for paragraph in paragraphs:
                    safe_paragraph = xml.sax.saxutils.escape(paragraph)
                    safe_paragraph = safe_paragraph.replace('\n', '<br/>')
                    story.append(Paragraph(safe_paragraph, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
        else:
            safe_findings = xml.sax.saxutils.escape(research_data.detailed_findings)
            paragraphs = [p.strip() for p in safe_findings.split('\n\n') if p.strip()]
            for paragraph in paragraphs:
                paragraph_text = paragraph.replace('\n', '<br/>')
                story.append(Paragraph(paragraph_text, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Sources
        story.append(Paragraph("Sources & References", heading_style))
        for i, source in enumerate(research_data.sources, 1):
            safe_source = xml.sax.saxutils.escape(source)
            story.append(Paragraph(f"{i}. {safe_source}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Tools Used
        story.append(Paragraph("Research Methodology & Tools", heading_style))
        for tool in research_data.tools_used:
            safe_tool = xml.sax.saxutils.escape(tool)
            story.append(Paragraph(f"• {safe_tool}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key Insights
        story.append(Paragraph("Key Insights", heading_style))
        for insight in research_data.key_insights:
            safe_insight = xml.sax.saxutils.escape(insight)
            story.append(Paragraph(f"• {safe_insight}", styles['Normal']))
        
        doc.build(story)  # type: ignore
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error("Error generating PDF report. Please try again or use the TXT export option.")
        print(f"PDF generation error: {str(e)}")
        return b""


def generate_comprehensive_content(query: str, content_type: str, word_count: int = 200) -> str:
    """
    Generate comprehensive, detailed content for research sections.
    Ensures no placeholder text and maintains academic quality.
    """
    ...  # (UNCHANGED — omitted for length)
    

def generate_academic_sources(query: str, count: int = 10) -> List[str]:
    ...  # (UNCHANGED)


def generate_academic_insights(query: str, count: int = 5) -> List[str]:
    ...  # (UNCHANGED)


def validate_and_enhance_content(research_data: ResearchResponse, query: str) -> ResearchResponse:
    ...  # (UNCHANGED)


def parse_fallback_response(output_text: str, query: str) -> ResearchResponse:
    ...  # (UNCHANGED)


def save_to_session_history(query: str, research_data: ResearchResponse):
    ...  # (UNCHANGED)


def display_research_results(research_data: ResearchResponse):
    ...  # (UNCHANGED)


def display_export_options():
    ...  # (UNCHANGED)


def main():
    ...  # (UNCHANGED — full UI preserved exactly)

if __name__ == "__main__":
    main()


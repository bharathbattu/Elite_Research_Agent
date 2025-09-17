import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor  # type: ignore
from langchain_core.tools import BaseTool
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    abstract: str
    detailed_findings: str
    sources: List[str]
    tools_used: List[str]
    key_insights: List[str]

# Initialize LLM - Single default Mistral model for the research assistant
llm = ChatMistralAI(model="mistral-small", api_key=os.getenv("MISTRAL_API_KEY"))  # type: ignore
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(  # type: ignore
    [
        (
            "system",
            """
            You are an elite academic research assistant specializing in comprehensive, publication-quality research reports.
            
            MANDATE: Generate detailed, professional research that meets academic publication standards. Every section must be substantive, well-researched, and thoroughly developed.
            
            RESEARCH PROCESS (MANDATORY):
            1. Use google_search extensively to gather current information, recent publications, and emerging trends
            2. Use wikipedia for authoritative background information and foundational knowledge
            3. Synthesize findings from multiple sources into cohesive, evidence-based analysis
            4. Ensure all claims are supported by credible sources and data
            
            STRICT OUTPUT REQUIREMENTS:
            
            topic: Academically precise research title (10-15 words, professional formatting)
            - Must reflect scope, timeframe, and significance
            - Example: "Artificial Intelligence Integration in Healthcare Systems: Clinical Outcomes and Implementation Challenges (2020-2025)"
            
            abstract: Comprehensive executive summary (EXACTLY 150-250 words, MANDATORY)
            - MUST include: research context, methodology, key findings, and implications
            - Academic tone with sophisticated vocabulary and complex sentence structures
            - Flowing prose that demonstrates deep understanding of the subject
            - NO generic statements or filler content
            - Each sentence must add substantial value
            
            detailed_findings: In-depth academic analysis (MINIMUM 800 words total across all sections)
            - Format: "## Background\n[2-3 paragraphs]\n\n## Current Developments\n[2-3 paragraphs]\n\n## Challenges\n[2-3 paragraphs]\n\n## Future Outlook\n[2-3 paragraphs]"
            - Background: Historical context, evolution, theoretical foundations (minimum 200 words)
            - Current Developments: Recent advances, current state, emerging trends (minimum 200 words)
            - Challenges: Technical, ethical, economic, social barriers (minimum 200 words)
            - Future Outlook: Projections, opportunities, strategic recommendations (minimum 200 words)
            - EACH paragraph must be 4-6 sentences with specific examples, data points, and expert insights
            - Use sophisticated academic language with precise terminology
            - Include quantitative data, statistics, and specific examples wherever possible
            
            sources: Authoritative academic references (MINIMUM 8-12 sources)
            - Format in proper APA style with complete citations
            - Include peer-reviewed journals, academic books, authoritative reports, and credible institutions
            - Examples: "Johnson, M. A., & Smith, K. L. (2024). Digital transformation in healthcare delivery. Journal of Medical Innovation, 15(3), 127-145.", "World Health Organization. (2023). Global health technology assessment report. Geneva: WHO Press."
            - Mix of recent sources (2020-2025) and foundational works
            
            tools_used: Comprehensive research methodology documentation
            - Detailed description of each tool and its specific application
            - Example: ["Google Search API for real-time data and recent publications", "Wikipedia API for foundational knowledge and cross-referencing", "LangChain Research Agent for systematic analysis and synthesis"]
            
            key_insights: 4-5 substantial, actionable research conclusions
            - Each insight must be evidence-based with specific implications
            - Include quantitative data or specific outcomes where possible
            - Example: "Implementation of AI diagnostic tools in radiology departments shows 34% improvement in early cancer detection rates while reducing diagnostic time by 45 minutes per case"
            
            QUALITY STANDARDS (NON-NEGOTIABLE):
            - NEVER use placeholder text like "Research was conducted..." or "Analysis shows..."
            - Every statement must be specific, substantive, and informative
            - Maintain academic rigor throughout with proper terminology and concepts
            - Demonstrate deep subject matter expertise in every section
            - Use evidence-based reasoning and cite specific examples
            
            CRITICAL: Return ONLY valid JSON object. All content (including Markdown headings, citations, etc.) must be inside string values.
            Do not include any text outside the JSON object. Ensure all strings are properly escaped.
            
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools: List[BaseTool] = [search_tool, wiki_tool, save_tool]  # type: ignore
agent = create_tool_calling_agent(  # type: ignore
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # type: ignore
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    # The agent executor returns a dict with "output" key containing the final response
    output_text = raw_response.get("output", "")
    structured_response = parser.parse(output_text)
    
    # Display the structured research report in a readable format
    print("\n" + "="*80)
    print("                    COMPREHENSIVE RESEARCH REPORT")
    print("="*80)
    print(f"\nRESEARCH TOPIC:")
    print(f"{structured_response.topic}")
    print(f"\nABSTRACT / EXECUTIVE SUMMARY:")
    print(f"{structured_response.abstract}")
    print(f"\nDETAILED FINDINGS:")
    print(f"{structured_response.detailed_findings}")
    print(f"\nSOURCES / REFERENCES:")
    for i, source in enumerate(structured_response.sources, 1):
        print(f"{i}. {source}")
    print(f"\nRESEARCH METHODOLOGY & TOOLS USED:")
    for tool in structured_response.tools_used:
        print(f"• {tool}")
    print(f"\nKEY INSIGHTS:")
    for insight in structured_response.key_insights:
        print(f"• {insight}")
    print("\n" + "="*80 + "\n")
    
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)